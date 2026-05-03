import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import kagglehub

DATASET_DIR = os.path.join(
    kagglehub.dataset_download("mikhailma/test-dataset"),
    "Google_Recaptcha_V2_Images_Dataset",
    "images",
)
IMG_SIZE = 96

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

# handle class imbalance
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = dict(zip(range(len(classes)), weights))
print("class weights computed")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=0, stratify=y
)

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

print("\nphase 1: training top layers...")
history1 = model.fit(X_train, y_train, validation_split=0.1, epochs=5,
                     batch_size=64, class_weight=class_weight_dict)

print("\nphase 2: fine-tuning...")
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(X_train, y_train, validation_split=0.1, epochs=5,
                     batch_size=64, class_weight=class_weight_dict)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\ntest accuracy: {acc:.4f}")

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

report = classification_report(y_true, y_pred, target_names=lb.classes_, digits=4)
print("\nclassification report:")
print(report)

metrics_path = os.path.join(os.path.dirname(__file__), "part2_eval_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"test_accuracy: {acc:.4f}\n\n")
    f.write("classification_report:\n")
    f.write(report)
print(f"saved evaluation metrics to: {metrics_path}")

model.save(os.path.join(os.path.dirname(__file__), "part2_model.keras"))
with open(os.path.join(os.path.dirname(__file__), "part2_labels.pkl"), "wb") as f:
    pickle.dump(lb, f)
print("saved model and labels")

# training curves — combine both phases
acc_all = history1.history["accuracy"] + history2.history["accuracy"]
val_acc_all = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss_all = history1.history["loss"] + history2.history["loss"]
val_loss_all = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc_all, label="train")
plt.plot(val_acc_all, label="val")
plt.axvline(x=4, color="gray", linestyle="--", label="phase 2 start")
plt.title("accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss_all, label="train")
plt.plot(val_loss_all, label="val")
plt.axvline(x=4, color="gray", linestyle="--", label="phase 2 start")
plt.title("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "training_curves.png"), dpi=150)
print("saved training_curves.png")

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(cm, display_labels=lb.classes_)
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"), dpi=150)
print("saved confusion_matrix.png")