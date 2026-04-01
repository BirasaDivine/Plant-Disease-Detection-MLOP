import tensorflow as tf
import numpy as np
import os
import cv2

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'mobilenetv2_weights.weights.h5')

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(3, activation='softmax')(x)
    return tf.keras.Model(inputs, out)

model = build_model()
model.load_weights(WEIGHTS_PATH, skip_mismatch=True)

def predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img, verbose=0)
    highest_prob = np.argmax(predictions[0])
    class_name = CLASS_NAMES[highest_prob]
    confidence = predictions[0][highest_prob] * 100
    return {
        "class": class_name,
        "confidence": round(float(confidence), 2)
    }

def reload_model():
    global model
    model = build_model()
    model.load_weights(WEIGHTS_PATH, skip_mismatch=True)