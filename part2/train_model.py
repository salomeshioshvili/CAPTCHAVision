import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Google_Recaptcha_V2_Images_Dataset", "images")
IMG_SIZE = 96  # MobileNetV2 works at 96x96 — much faster than 224x224

X, y = [], []

for label in sorted(os.listdir(DATASET_DIR)):
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"{label}: {len(files)} images")

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label.lower().replace(" ", "_"))

print(f"\ntotal images loaded: {len(X)}")

if len(X) == 0:
    print("no images found — check DATASET_DIR path")
    exit()

X = np.array(X, dtype="float32") / 127.5 - 1.0

lb = LabelBinarizer()
y_enc = lb.fit_transform(y)
num_classes = len(lb.classes_)
print(f"classes: {list(lb.classes_)}")

# fix class imbalance — penalises mistakes on rare categories more
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = dict(zip(range(len(classes)), weights))
print("class weights computed")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=0, stratify=y
)

# MobileNetV2 — 3.4M params vs ResNet50's 25M, trains ~7x faster
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# phase 1: train top layers only — 5 epochs is enough
print("\nphase 1: training top layers...")
model.fit(X_train, y_train, validation_split=0.1, epochs=5,
          batch_size=64, class_weight=class_weight_dict)

# phase 2: fine-tune last 20 layers
print("\nphase 2: fine-tuning...")
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, validation_split=0.1, epochs=5,
          batch_size=64, class_weight=class_weight_dict)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\ntest accuracy: {acc:.4f}")

model_path = os.path.join(os.path.dirname(__file__), "part2_model.keras")
labels_path = os.path.join(os.path.dirname(__file__), "part2_labels.pkl")

model.save(model_path)
with open(labels_path, "wb") as f:
    pickle.dump(lb, f)

print("saved model and labels")