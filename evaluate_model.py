import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# ======== Settings ========
img_size = (224, 224)
test_dir = "test"  # Path to the test directory

# ======== Load model ========
model = load_model("efficientnet_b0_trained.keras")

# ======== Labels and images ========
X_test = []
y_test = []
class_names = sorted(os.listdir(test_dir))

label_to_int = {label: idx for idx, label in enumerate(class_names)}
int_to_label = {idx: label for label, idx in label_to_int.items()}

for label in class_names:
    class_path = os.path.join(test_dir, label)
    for fname in os.listdir(class_path):
        img_path = os.path.join(class_path, fname)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        X_test.append(img_array)
        y_test.append(label_to_int[label])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ======== Prediction ========
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ======== Save evaluation report ========
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_path = f"evaluation_report_{now}.txt"
conf_matrix_path = f"confusion_matrix_{now}.png"

with open(report_path, "w", encoding="utf-8") as f:
    print(" Accuracy:", accuracy_score(y_test, y_pred), file=f)
    print(" F1 Score (macro):", f1_score(y_test, y_pred, average="macro"), file=f)
    print("\n Classification Report:", file=f)
    print(classification_report(
        y_test, y_pred, target_names=[int_to_label[i] for i in range(len(int_to_label))]
    ), file=f)

# ======== Confusion Matrix ========
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, square=True, cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.show()

print(f"\n Text report saved to: {report_path}")
print(f" Confusion matrix image saved to: {conf_matrix_path}")
