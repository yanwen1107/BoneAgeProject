from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, morphology, measure
from skimage.filters import threshold_otsu
from scipy import ndimage
import cv2


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index_2')
def index_2():
    return render_template('index_2.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # 獲取靜態文件的本地路徑
    static_path = os.path.join(app.root_path, 'static', 'First_Model.h5')

    # 載入模型
    model = load_model(static_path)
    data_url = request.form['imageData']

    # 將base64轉換為OpenCV圖片
    _, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 進行灰階轉換
    
    processed_image = preprocess_image_all(image)
    test = []
    test.append(processed_image)
    test = np.array(test)
    test = test.astype('float32') / 255.0
    prediction = model.predict(test)

    # 將處理後的圖片轉換成base64
    retval, buffer = cv2.imencode('.png', processed_image.squeeze())
    gray_image_encoded = base64.b64encode(buffer).decode('utf-8')

    # return jsonify({'processedImage': gray_image_encoded})
    return jsonify({'processedImage': gray_image_encoded, 'prediction': float(prediction[0])})


#-----------------------------------------------------------------------
def resize(image):
    size =224
    width, height = image.size
    scale = max(width, height)/size
    width_size = int(width/scale)
    image = image.resize((width_size,size))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = np.zeros((224, 224), dtype=np.uint8)
    resized_image[:, :] = 0
    x_offset = (image.shape[1] - image.shape[1]) // 2
    y_offset = (image.shape[0] - image.shape[0]) // 2
    resized_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    return resized_image


def standardize_grayscale(image):
    current_mean_intensity = np.mean(image)

    adjusted_image = image + (75 - current_mean_intensity)

    return adjusted_image


def clahe_equalize(image, clip_limit=2.0, grid_size=(8, 8)):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized_image = clahe.apply(image)

    return equalized_image

def denoise_image(image):
    # use median filter
    denoised_image = ndimage.median_filter(image, size=3)

    return denoised_image

def sharpen_image(image):

    image_pil = Image.fromarray(image)
    image_pil = image_pil.convert('L')
    sharpened_image = image_pil.filter(ImageFilter.SHARPEN)

    return np.array(sharpened_image)


def preprocess_image_all(image):
    img = Image.fromarray(image)

    #Resize
    img_resized = resize(img)
    # Standardize Grayscale
    img_standardized = standardize_grayscale(img_resized)

    # Histogram(CLACHE) Equalization
    hand_equalized = clahe_equalize(img_standardized)

    # Denoise
    hand_denoised = denoise_image(hand_equalized)

    # Sharpen Filter
    hand_sharpened = sharpen_image(hand_denoised)

    return hand_sharpened


if __name__ == '__main__':
    # 使用相對路徑設定 templates 資料夾的位置
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    app.template_folder = template_dir

    app.run(debug=True)
