from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

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

    # 進行灰階轉換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 將處理後的圖片轉換成base64
    retval, buffer = cv2.imencode('.png', gray_image)
    gray_image_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processedImage': gray_image_encoded})

if __name__ == '__main__':
    # 使用相對路徑設定 templates 資料夾的位置
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    app.template_folder = template_dir

    app.run(debug=True)
