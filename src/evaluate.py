
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.config import MODEL_PATH, OUTPUT_DIR, CLASS_NAMES
from src.data import load_data


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    _, _, (x_test, y_test) = load_data()
    model = keras.models.load_model(MODEL_PATH)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print("Saved outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
