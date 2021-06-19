from flask import Flask,render_template,request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("pnm.html")

@app.route('/vgg')
def vgg():
    return render_template("vgg.html")

@app.route("/Dnet")
def dnet():
    return render_template("dnet.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("./pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('result.html', pred = pred)
        except:
            message = "Please upload an Image"
            return render_template('pnm.html', message = message)
    return render_template('pnm.html',message ="some error occured")


if __name__ == '__main__':
    app.run(debug=True)