from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)[0]
    prediction = round(prediction, 2)
    return render_template('index.html', prediction_text=f'Chất lượng dự đoán: {prediction} / 10')

if __name__ == "__main__":
    app.run(debug=True)