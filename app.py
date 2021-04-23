import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import keras
diseases={'Actinic keratosis': 0,
          'Basal cell carcinoma': 1,
          'Blister': 2,
          'Chickenpox': 3,
          'Cold sore': 4,
          'Impetigo': 5,
          'Melanoma': 6,
          'Psoriasis': 7,
          'Ringworm': 8,
          'Rosacea': 9,
          'Wart': 10,
          'acne': 11,
          'rashes from bacterial or fungal infection': 12}
app = Flask(__name__)
model = keras.models.load_model("skin_dieases_detector_model_best.h5")

@app.route('/')
def home():
    return render_template('xxx.html')

@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    img = request.files['image']
    img.save('static/{}.jpg')    
    img_arr = cv2.imread('static/{}.jpg')

    img_arr = cv2.resize(img_arr, (200,200))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 200,200,3)
    val=model.predict(img_arr)
    va=list(val[0].round(2)).index(max(val))
    output=list(diseases.keys())[va]
 
    return render_template('xxx.html', prediction_text='SKIN PROBLEM DETECTED : '+output)


if __name__ == "__main__":
    app.run(debug=True)