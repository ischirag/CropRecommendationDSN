from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = mx.transform(single_pred)
    prediction = model.predict(scaled_features)

    crop_dict = {1: "wheat", 2: "soybean", 3: "gram", 4: "paddy", 5: "maize", 6: "mustard", 7: "lentil",
                 8: "urad", 9: "groundnut", 10: "tomato"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)