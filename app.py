from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"   

# ================================
# Konfigurasi
# ================================
MIN_CONFIDENCE = 0.5
WIDTH = 640
HEIGHT = 640
MODEL_PATH = "frozen_east_text_detection_copy.pb"
# IMAGE_PATH = "61078fe3ef303.jpg"
IMAGE_PATH = "61078fe3ef303.jpg"
# ================================
# Load dan pra-pemrosesan gambar
# ================================
image = cv2.imread(IMAGE_PATH)
orig = image.copy()
(H, W) = image.shape[:2]

# Tingkatkan kontras dan hapus noise (CLAHE + Bilateral Filter)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
image = cv2.bilateralFilter(image, 9, 75, 75)

# Resize gambar
rW = W / float(WIDTH)
rH = H / float(HEIGHT)
image = cv2.resize(image, (WIDTH, HEIGHT))
(H, W) = image.shape[:2]

# ================================
# Load model EAST
# ================================
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(MODEL_PATH)

# ================================
# Blob dan Forward Pass
# ================================
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward([
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
])
end = time.time()
print(f"[INFO] text detection took {end - start:.6f} seconds")

# ================================
# Decode hasil deteksi
# ================================
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
        if scoresData[x] < MIN_CONFIDENCE:
            continue

        offsetX, offsetY = x * 4.0, y * 4.0
        angle = anglesData[x]
        cos, sin = np.cos(angle), np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(float(scoresData[x]))

# ================================
# Non-Max Suppression
# ================================
boxes = non_max_suppression(np.array(rects), probs=confidences)

def is_valid_text(text):
    text = text.strip()
    if len(text) < 2:
        return False
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < 1:
        return False
    return True

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return [word.strip().upper() for word in f.readlines()]

def preprocess_with_vocab(text, vocab):
    text = clean_text(text)
    words = text.split()
    corrected = []
    
    for word in words:
        matches = difflib.get_close_matches(word, vocab, n=1, cutoff=0.6)
        corrected.append(matches[0] if matches else word)
    
    return ' '.join(corrected)

# Load vocab sekali di awal
VOCAB = load_vocab("vocab.txt")  # Ganti dengan path sesuai lokasi file



for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    
    # Filter based on box characteristics
    width = endX - startX
    height = endY - startY
    aspect_ratio = width / float(height)
    area = width * height
    
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        continue
    if area < 100 or area > (W*H)/4:  # Too small or too large
        continue
        
    roi = orig[startY:endY, startX:endX]
    
    # Contrast check
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contrast_ratio = np.sum(binary == 255) / float(np.sum(binary == 0) + 1e-6)
    if contrast_ratio < 0.1 or contrast_ratio > 10:
        continue
    
    # Preprocess for OCR
    gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # OCR with stricter config
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?()[]{}<>@#$%^&*+-=/"
    text = pytesseract.image_to_string(gray_roi, config=config)
    
    if is_valid_text(text):
        print(f"[TEXT] Detected: {text.strip()}")
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(orig, text.strip(), (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# ================================
# Tampilkan hasil
# ================================
cv2.imshow("Text Detection", orig)
cv2.imwrite("text_detection_output.jpg", orig)  # Simpan hasil
cv2.waitKey(0)
cv2.destroyAllWindows()
