import numpy as np
import cv2


IMG_SIZE=128
def preprocess_image(image_path):
    img=cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path : {image_path}")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IMG_SIZE , IMG_SIZE))
    img=img.astype('float32')
    img=img/255.0
    img=np.expand_dims(img , axis=0)
    return img