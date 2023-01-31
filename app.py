from flask import Flask,render_template,request
import pickle
app = Flask(__name__)
#load the model
model = pickle.load(open('savedmodel.sav','rb'))

@app.route('/')
def home():
    result = ''
    return render_template('home.html', **locals())

@app.route('/predict', methods=['POST','GET'])
def predict():
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])
    result = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])[0]
    return render_template('home.html', **locals())


if __name__ == '__main__':
        app.run(debug=True)
