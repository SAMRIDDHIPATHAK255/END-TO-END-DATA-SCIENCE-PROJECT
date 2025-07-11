

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    sex = request.form['sex']
    embarked = request.form['embarked']

    
    input_df = pd.DataFrame([{
        'age': age,
        'fare': fare,
        'sex': sex,
        'embarked': embarked
    }])

    prediction = model.predict(input_df)[0]
    result = "Survived" if prediction == 1 else "Did not survive"

    return render_template("form.html", prediction=result)


    
    input_df = pd.DataFrame([data])

    
    prediction = model.predict(input_df)[0]
    return jsonify({"survived": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
