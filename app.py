from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


app = Flask(__name__)
model = xgb.XGBClassifier()
model.load_model("model.txt")

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        age = int(request.form['age'])
        anaemia = int(request.form['anaemia'])
        creatinine = float(request.form['creatinine'])
        diabetes = int(request.form['diabetes'])
        ejection_fraction = float(request.form['ejection_fraction'])
        platelets = int(request.form['platelets'])
        serum_creatinine = float(request.form['serum_creatinine'])
        serum_sodium = float(request.form['serum_sodium'])
        sex = int(request.form['sex'])
        smoking = int(request.form['smoking'])
        time = float(request.form['time'])
        death_event = float(request.form['death_event'])
        temperature = float(request.form['temperature'])

        input_data = [[age, anaemia, creatinine, diabetes, ejection_fraction, platelets, serum_creatinine, serum_sodium, sex, smoking, time, death_event, temperature]]

        prediction=model.predict(np.array(input_data))
        output=round(prediction[0],2)
        if output == 0:
            return render_template('index.html', prediction_text="You don't have High BP")
        else:
            return render_template('index.html', prediction_text="You have high BP")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

