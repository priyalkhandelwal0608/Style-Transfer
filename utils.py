import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing import image as kp_image
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# ----------------------------
# Load Image from URL
# ----------------------------
def load_img_from_url(img_url, max_dim=512):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    long_dim = max(img.size)
    scale = max_dim / long_dim

    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)),
        Image.Resampling.LANCZOS
    )

    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img


# ----------------------------
# Deprocess Image
# ----------------------------
def deprocess_img(processed_img):
    x = processed_img.copy()

    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


# ----------------------------
# Gram Matrix
# ----------------------------
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]

    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)