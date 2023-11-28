from flask import Flask, request, render_template, jsonify, redirect, url_for, json
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
import numpy as np
import pickle
import jsonpickle
import requests
import datetime

app = Flask(__name__)


sensor_data = []


@app.route('/sensors/model', methods=['POST'])
def train_model():
    # generate sample data
    X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=100, n_features=3,
                                                     contamination=0.01, random_state=42)
    # classify data using KNN
    clf = KNN()
    clf.fit(X_train)
    y_pred = clf.predict(X_test)
    print(y_pred)

    # save the model to a file
    pickle.dump(clf, open("model.pkl", 'wb'))

    # return to main page saying the model is trained
    return redirect(url_for('home', trained=True), code=302)


@app.route('/sensors', methods=['POST'])
def create_sensors():
    # get json from request body
    current = request.get_json()

    # load model from file
    model = pickle.load(open("model.pkl", 'rb'))

    # transform model values into numpy array to pass to predict function
    float_features = [float(current["temperature"]), float(current["humidity"]), float(current["sound-volume"])]
    features = [np.array(float_features)]

    # predict according to model
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    # save values in memory
    current["result"] = int(output)
    current['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sensor_data.append(current)

    # return result as json
    return jsonify('{ "result": ' + str(output) + ' }')


@app.route('/sensors', methods=['GET'])
def get_sensors():
    # using jsonpickle to fix data type error for serialization
    return jsonpickle.encode(sensor_data)


# UI routes
@app.route('/', methods=['GET'])
def home():
    train_text = "Model has been trained" if request.args.get('trained') else ""
    return render_template('index.html', train_text=train_text)


@app.route('/', methods=['POST'])
def ui():
    api_url = 'http://127.0.0.1:5000/sensors'
    # create api body
    event_data = {
        "temperature": float(request.form.get("temperature")),
        "humidity": float(request.form.get("humidity")),
        "sound-volume": float(request.form.get("sound-volume")),
    }

    # execute request
    response = requests.post(api_url, json=event_data)

    # read response as json
    response_json = json.loads(response.json())

    # generate view message
    text = 'Prediction:' + ('ANOMALOUS' if str(response_json["result"]) == '0' else 'NORMAL')
    return render_template('index.html', prediction_text=text)


if __name__ == "__main__":
    app.run()
