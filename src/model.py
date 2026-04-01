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

def retrain(data_dir, epochs=5):
    images = []
    labels = []

    for label_idx, class_folder in enumerate(CLASS_NAMES):
        folder_path = os.path.join(data_dir, class_folder)
        if not os.path.exists(folder_path):
            continue
        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label_idx)

    if len(images) == 0:
        return "No images found for retraining"

    images = np.array(images)
    labels = np.array(labels)
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=3)

    model = build_model()
    model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(images, labels_onehot, epochs=epochs, batch_size=32)
    model.save_weights(WEIGHTS_PATH)
    return 'Retraining complete'