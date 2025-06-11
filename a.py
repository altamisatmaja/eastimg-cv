#!/usr/bin/env python
# Improved Text Recognition with EAST, Tesseract OCR, and Custom Vocabulary
# python a.py --east=frozen_east_text_detection_copy.pb --image=whitebg1.jpg --vocab=my_vocab.txt
# python a.py --east=frozen_east_text_detection_copy.pb --image=1131w-qyven2ZczO8.webp --vocab=my_vocab.txt
# python a.py --east=frozen_east_text_detection_copy.pb --image=38ddfc2c-1c96-479a-bda5-868ace95cf66.jpg --vocab=my_vocab.txt

import cv2
import numpy as np
import pytesseract
import argparse
import re
import difflib
from imutils.object_detection import non_max_suppression

# Text detection functions
def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < args["min_confidence"]:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# Image enhancement
def preprocess_image(image):
    # Contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# Text cleaning
def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)  # Keep alphanumeric and basic punctuation
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.upper()

# Vocabulary processing
def load_vocab(vocab_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return [word.strip().upper() for word in f.readlines() if word.strip()]
    except FileNotFoundError:
        print(f"[WARNING] Vocabulary file {vocab_path} not found. Continuing without vocabulary.")
        return []

def correct_with_vocab(text, vocab):
    if not vocab:
        return text
    
    words = text.split()
    corrected = []
    
    for word in words:
        # Skip short words and numbers
        if len(word) <= 2 or word.isdigit():
            corrected.append(word)
            continue
            
        matches = difflib.get_close_matches(word, vocab, n=1, cutoff=0.7)
        corrected.append(matches[0] if matches else word)
    
    return ' '.join(corrected)

# Main processing
def main():
    # Load and preprocess image
    image = cv2.imread(args["image"])
    if image is None:
        print(f"[ERROR] Could not load image {args['image']}")
        return
    
    orig = preprocess_image(image.copy())
    (origH, origW) = orig.shape[:2]

    # Resize image
    (newW, newH) = (args["width"], args["height"])
    rW = origW / float(newW)
    rH = origH / float(newH)
    resized = cv2.resize(orig, (newW, newH))
    (H, W) = resized.shape[:2]

    # Load vocabulary
    vocab = load_vocab(args["vocab"])

    # Load EAST text detector
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    print("[INFO] Loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # Run text detection
    blob = cv2.dnn.blobFromImage(resized, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Decode predictions
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Process detections
    results = []
    for (startX, startY, endX, endY) in boxes:
        # Scale coordinates
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # Apply padding
        dX = int((endX - startX) * args["padding"])
        dY = int((endY - startY) * args["padding"])
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        
        # Extract and preprocess ROI
        roi = orig[startY:endY, startX:endX]
        if roi.size == 0:
            continue
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Multiple OCR configurations
        configs = [
            "--oem 3 --psm 6 -l eng",  # Uniform block
            "--oem 3 --psm 7 -l eng",  # Single line
            "--oem 3 --psm 8 -l eng"   # Single word
        ]
        
        texts = []
        for cfg in configs:
            if args["vocab"]:
                cfg += f" --user-words {args['vocab']}"
            text = pytesseract.image_to_string(gray, config=cfg)
            text = clean_text(text)
            if text:
                texts.append(text)
        
        # Get best result
        if texts:
            text = max(set(texts), key=texts.count)
            text = correct_with_vocab(text, vocab)
            results.append(((startX, startY, endX, endY), text))

    # Sort results top-to-bottom, left-to-right
    results = sorted(results, key=lambda r: (r[0][1], r[0][0]))

    # Display results
    output = orig.copy()
    final_text = []
    for ((startX, startY, endX, endY), text) in results:
        final_text.append(text)
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(output, text, (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("\nDetected Text:")
    print("-" * 30)
    print("\n".join(final_text))
    print("-" * 30)

    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    ap.add_argument("-east", "--east", type=str, required=True, help="path to EAST text detector")
    ap.add_argument("-v", "--vocab", type=str, default="", help="path to custom vocabulary file")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.7, help="min confidence for text detection")
    ap.add_argument("-w", "--width", type=int, default=640, help="resized image width")
    ap.add_argument("-e", "--height", type=int, default=640, help="resized image height")
    ap.add_argument("-p", "--padding", type=float, default=0.05, help="padding around ROI")
    args = vars(ap.parse_args())

    main()