from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# 加载模型
model_path = r'D:\file\jupyter\dogs_cats\model\dogs_cats_model.keras'
model = load_model(model_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 获取上传的文件
    file = request.files['file']
    if file:
        # 保存文件到临时位置
        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        # 预处理图像
        img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # 预测
        predictions = model.predict(img_array)
        result = "Dog" if predictions[0][0] < 0.5 else "Cat"

        # 删除临时文件
        os.remove(temp_path)

        return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)