from PIL import Image, ImageFilter
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
from scipy import ndimage
import base64
import os

def resize(image):
    size =224
    width, height = image.size
    scale = max(width, height)/size
    width_size = int(width/scale)
    image = image.resize((width_size,size))
    image = np.array(image)
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

def preprocess_image_all(img_path):
    img = Image.open(img_path)
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

app = Flask(__name__)
model = load_model(r'C:\Users\User\Documents\test\First_Model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data_url = request.form['imageData']

    # 將base64轉換為OpenCV圖片
    _, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    processed_image = preprocess_image_all(image)
    
    # 將處理後的圖片轉換成base64
    retval, buffer = cv2.imencode('.png', processed_image)
    
    # 預處理
    processed_image_encoded = base64.b64encode(buffer).decode('utf-8')

    # 使用模型進行預測
    predictions = model.predict(processed_image)  
    
    result = {
        
        #'classification': {
         #   'class_index': predictions.argmax(),
          #  'class_label': 'Your_Class_Label',  
        #},
        'processedImage': processed_image_encoded,
    }

    return jsonify(result)


if __name__ == '__main__':
    # 使用相對路徑設定 templates 資料夾的位置
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    app.template_folder = template_dir

    app.run(debug=True)
