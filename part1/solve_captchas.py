import os
import pickle
import sys
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "captcha_model.keras"))
with open(os.path.join(os.path.dirname(__file__), "model_labels.pkl"), "rb") as f:
    lb = pickle.load(f)

def solve(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue
        if w / h > 1.25:
            half = w // 2
            regions.append((x, gray[y:y+h, x:x+half]))
            regions.append((x + half, gray[y:y+h, x+half:x+w]))
        else:
            regions.append((x, gray[y:y+h, x:x+w]))

    regions.sort(key=lambda t: t[0])
    if len(regions) != 4:
        return None

    result = ""
    for _, crop in regions:
        crop = cv2.resize(crop, (20, 20)).astype("float32") / 255.0
        crop = crop[np.newaxis, ..., np.newaxis]
        pred = model.predict(crop, verbose=0)
        result += lb.classes_[np.argmax(pred)]

    return result

if len(sys.argv) > 1:
    result = solve(sys.argv[1])
    print(f"predicted: {result}")
    sys.exit()

CAPTCHA_DIR = os.path.join(os.path.dirname(__file__), "generated_captcha_images")

correct = 0
total = 0

for fname in os.listdir(CAPTCHA_DIR):
    if not fname.endswith(".png"):
        continue
    true_label = os.path.splitext(fname)[0].upper()
    predicted = solve(os.path.join(CAPTCHA_DIR, fname))
    if predicted is None:
        continue
    total += 1
    if predicted == true_label:
        correct += 1
    else:
        print(f"wrong: expected {true_label}, got {predicted}")

print(f"\naccuracy: {correct}/{total} ({correct/total*100:.1f}%)")