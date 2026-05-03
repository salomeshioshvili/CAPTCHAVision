import os
import cv2
import uuid

CAPTCHA_DIR = os.path.join(os.path.dirname(__file__), "generated_captcha_images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_letter_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(CAPTCHA_DIR):
    if not fname.endswith(".png"):
        continue

    answer = os.path.splitext(fname)[0].upper()
    img = cv2.imread(os.path.join(CAPTCHA_DIR, fname))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue
        if w / h > 1.25:
            half = w // 2
            letter_regions.append((x, gray[y:y+h, x:x+half]))
            letter_regions.append((x + half, gray[y:y+h, x+half:x+w]))
        else:
            letter_regions.append((x, gray[y:y+h, x:x+w]))

    letter_regions.sort(key=lambda t: t[0])

    if len(letter_regions) != len(answer):
        continue

    for char, (_, crop) in zip(answer, letter_regions):
        save_dir = os.path.join(OUTPUT_DIR, char)
        os.makedirs(save_dir, exist_ok=True)
        resized = cv2.resize(crop, (20, 20))
        cv2.imwrite(os.path.join(save_dir, f"{uuid.uuid4().hex}.png"), resized)

print("done")