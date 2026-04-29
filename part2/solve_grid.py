import pickle
import sys
import os
import cv2
import numpy as np
import tensorflow as tf

model_path = os.path.join(os.path.dirname(__file__), "part2_model.keras")
labels_path = os.path.join(os.path.dirname(__file__), "part2_labels.pkl")

model = tf.keras.models.load_model(model_path)
with open(labels_path, "rb") as f:
    lb = pickle.load(f)

IMG_SIZE = 96

def split_grid(image_path, grid_size=3):
    img = cv2.imread(image_path)
    if img is None:
        print(f"could not read {image_path}")
        return []

    h, w = img.shape[:2]
    tile_h = h // grid_size
    tile_w = w // grid_size

    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * tile_h
            y2 = y1 + tile_h
            x1 = col * tile_w
            x2 = x1 + tile_w
            tiles.append((row, col, img[y1:y2, x1:x2]))

    return tiles


def classify_tile(tile_bgr):
    img = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    img = img / 127.5 - 1.0
    img = img[np.newaxis, ...]
    probs = model.predict(img, verbose=0)[0]
    idx = np.argmax(probs)
    return lb.classes_[idx], probs[idx]


def solve_grid(image_path, prompt_category, grid_size=3, threshold=0.5):
    tiles = split_grid(image_path, grid_size)
    if not tiles:
        return []

    # normalise prompt the same way labels were normalised during training
    prompt_category = prompt_category.lower().replace(" ", "_")

    matches = []
    print(f"\nclassifying {len(tiles)} tiles for: '{prompt_category}'")
    print("-" * 45)

    for row, col, tile in tiles:
        pred_class, confidence = classify_tile(tile)
        is_match = pred_class == prompt_category and confidence >= threshold
        status = "MATCH" if is_match else "-"
        print(f"  tile ({row},{col}): {pred_class} ({confidence:.2f})  {status}")
        if is_match:
            matches.append((row, col))

    print(f"tiles to click: {matches}")
    return matches


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python solve_grid.py <grid_image.png> <category> [grid_size]")
        print("example: python solve_grid.py captcha.png traffic_light 3")
        sys.exit(1)

    solve_grid(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)