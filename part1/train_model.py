import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(__file__)
LETTER_DIR = os.path.join(BASE_DIR, "extracted_letter_images")

X, y = [], []

for label in sorted(os.listdir(LETTER_DIR)):
    folder = os.path.join(LETTER_DIR, label)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (20, 20))
        X.append(img)
        y.append(label)

X = np.array(X, dtype="float32") / 255.0
X = X[..., np.newaxis]

lb = LabelBinarizer()
y_enc = lb.fit_transform(y)
num_classes = len(lb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=0)

model = models.Sequential([
    layers.Input(shape=(20, 20, 1)),
    layers.Conv2D(20, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(50, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(500, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=32)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"test accuracy: {acc:.4f}")

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

report = classification_report(y_true, y_pred, target_names=lb.classes_, digits=4)
print("\nclassification report:")
print(report)

metrics_path = os.path.join(BASE_DIR, "part1_eval_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"test_accuracy: {acc:.4f}\n\n")
    f.write("classification_report:\n")
    f.write(report)
print(f"saved evaluation metrics to: {metrics_path}")

model.save(os.path.join(BASE_DIR, "captcha_model.keras"))
with open(os.path.join(BASE_DIR, "model_labels.pkl"), "wb") as f:
    pickle.dump(lb, f)
print("saved model and labels")

# training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_curves.png"), dpi=150)
print("saved training_curves.png")

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(cm, display_labels=lb.classes_)
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
print("saved confusion_matrix.png")