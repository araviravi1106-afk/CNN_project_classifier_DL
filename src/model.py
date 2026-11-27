
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, NUM_CLASSES


def conv_block(x, filters, dropout):
    
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Preprocessing + augmentation (augmentation is only active during training)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    # Feature extractor
    x = conv_block(x, 32, 0.2)
    x = conv_block(x, 64, 0.3)
    x = conv_block(x, 128, 0.4)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar10_cnn")


if __name__ == "__main__":
    build_model().summary()
