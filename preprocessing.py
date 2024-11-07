import cv2
import numpy as np
import random

def sharpening_img(img):
    img_blur = cv2.blur(img, (3, 3))
    diff = cv2.subtract(img, img_blur)
    final = cv2.addWeighted(img, 1.5, diff, -0.5, 0)
    return final

def extract_character(img):
    # Threshold the image to binary
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the character
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return img[y:y+h, x:x+w]  # Crop the character
    return img  # Return original if no contour is found

def augment_data(img):
    # Randomly apply augmentation techniques
    if random.random() < 0.5:  # Random rotation
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        img = cv2.warpAffine(img, M, (28, 28))
    
    if random.random() < 0.5:  # Random translation
        tx = random.randint(-3, 3)
        ty = random.randint(-3, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (28, 28))
    
    return img

def preprocess_data(X):
    processed_images = []
    for x in X:
        # Step 1: Sharpen the image
        sharpened_img = sharpening_img(x)
        # Step 2: Extract the character from the sharpened image
        character_img = extract_character(sharpened_img)
        # Step 3: Resize to 28x28 if necessary
        character_img = cv2.resize(character_img, (28, 28))
        # Step 4: Apply data augmentation
        augmented_img = augment_data(character_img)
        processed_images.append(augmented_img)
    
    return np.array(processed_images)
