#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAST Text Detector - From Training to Inference (Single File Version)
"""

import os
import cv2
import numpy as np
import tensorflow as tf

Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Concatenate = tf.keras.layers.Concatenate


# ====================== 1. EAST Model Architecture ======================
def build_east_model(input_shape=(512, 512, 3)):
    """Build EAST model for text detection"""
    inputs = Input(shape=input_shape, name='input_image')
    
    # Feature extractor (simplified version)
    h = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    h = Conv2D(32, (3,3), padding='same', activation='relu')(h)
    h = Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu')(h)  # Downsample
    
    # U-Net like decoder
    merge = Concatenate()([h, h])  # Simplified skip connection
    
    # Output layers
    score_map = Conv2D(1, (1,1), activation='sigmoid', name='score_map')(merge)
    geo_map = Conv2D(4, (1,1), activation='sigmoid', name='geo_map')(merge)
    
    return Model(inputs=inputs, outputs=[score_map, geo_map])

# ====================== 2. Data Preparation ======================
class TextDatasetGenerator:
    """Simplified data generator for demo purposes"""
    def __init__(self, image_dir, label_dir, batch_size=4):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __iter__(self):
        while True:
            for i in range(0, len(self.image_paths), self.batch_size):
                batch_images = []
                batch_scores = []
                batch_geos = []
                
                for img_path in self.image_paths[i:i+self.batch_size]:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (512, 512))
                    img = img.astype(np.float32) / 255.0
                    
                    # Dummy ground truth (replace with real annotations)
                    score_map = np.random.rand(128, 128, 1)  # 1/4 scale of input
                    geo_map = np.random.rand(128, 128, 4)
                    
                    batch_images.append(img)
                    batch_scores.append(score_map)
                    batch_geos.append(geo_map)
                
                yield np.array(batch_images), [np.array(batch_scores), np.array(batch_geos)]

# ====================== 3. Training Setup ======================
def dice_loss(y_true, y_pred):
    """Custom loss function for text/non-text classification"""
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def train_model():
    """Train the EAST model"""
    # Initialize model
    model = build_east_model()
    model.compile(
        optimizer='adam',
        loss=[dice_loss, 'mse'],  # Loss for score map and geometry map
        loss_weights=[1.0, 10.0]   # Weight geometry prediction higher
    )
    
    # Dummy data generator (replace with your dataset)
    train_gen = TextDatasetGenerator('/Users/apple/Documents/eastimg-cv/data/ch4_end_to_end/training_set/ch4_training_images', '/Users/apple/Documents/eastimg-cv/data/ch4_end_to_end/training_set/ch4_training_localization_transcription_gt')
    val_gen = TextDatasetGenerator('dataset/val', 'labels/val')
    
    # Train
    model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=10
    )
    
    # Save model
    model.save('east_model.h5')
    print("Model saved to east_model.h5")

# ====================== 4. Text Detection ======================
class EASTTextDetector:
    """EAST text detector for inference"""
    def __init__(self, model_path):
        # Load model
        self.net = cv2.dnn.readNet(model_path)
        self.input_size = (320, 320)  # Standard EAST input size
        self.conf_threshold = 0.5
        
    def detect(self, image):
        """Detect text regions in image"""
        # Preprocess
        blob = cv2.dnn.blobFromImage(
            image, 1.0, self.input_size,
            (123.68, 116.78, 103.94), True, False
        )
        
        # Forward pass
        self.net.setInput(blob)
        scores, geometry = self.net.forward(['score_map', 'geo_map'])
        
        # Post-process
        boxes, confidences = self._decode_predictions(scores, geometry)
        
        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxesRotated(
            boxes, confidences, self.conf_threshold, 0.4
        )
        
        return [boxes[i[0]] for i in indices]
    
    def _decode_predictions(self, scores, geometry):
        """Convert model outputs to text boxes"""
        # Implementation simplified for demo
        # Full version should handle geometry decoding
        boxes = []
        confidences = []
        
        # Dummy decoding (replace with actual implementation)
        for y in range(scores.shape[2]):
            for x in range(scores.shape[3]):
                if scores[0][0][y][x] > self.conf_threshold:
                    boxes.append([x*4, y*4, 40, 20, 0])  # x,y,w,h,angle
                    confidences.append(float(scores[0][0][y][x]))
        
        return boxes, confidences

# ====================== 5. Main Program ======================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'detect'], required=True)
    parser.add_argument('--image', help='Path to input image for detection')
    parser.add_argument('--model', default='east_model.pb', help='Path to model file')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("[INFO] Training EAST model...")
        train_model()
    else:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        
        print("[INFO] Running text detection...")
        detector = EASTTextDetector(args.model)
        image = cv2.imread(args.image)
        
        boxes = detector.detect(image)
        
        # Draw detections
        for box in boxes:
            cv2.polylines(image, [np.int0(box)], True, (0, 255, 0), 2)
        
        cv2.imshow('Text Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()