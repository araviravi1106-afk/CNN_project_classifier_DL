
import sys
import numpy as np
from PIL import Image
from tensorflow import keras

from src.config import MODEL_PATH, CLASS_NAMES, IMG_SIZE

_model = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def predict_image(image: Image.Image, top_k: int = 3):
    """Returns a list of (class_name, probability) sorted high -> low."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)  # (1, 32, 32, 3)
    probs = get_model().predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.predict <image_path>")
    for name, p in predict_image(Image.open(sys.argv[1])):
        print(f"{name:12s} {p * 100:.1f}%")
