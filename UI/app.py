from flask import Flask, render_template, request
import pickle
import numpy as np

model1 = pickle.load(open('heart_disease.pkl', 'rb'))
model2 = pickle.load(open('stroke1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    x1 = request.form.get('gender')
    d2 = int(request.form['age'])
    x3 = request.form.get('hyp')
    x4 = request.form.get('r_type')
    d5 = float(request.form['agl'])
    d6 = float(request.form['bmi'])
    x7 = request.form.get('s_status')
    if(x1=="Male"):
        d1=0
    elif(x1=="Female"):
        d1=1
    elif(x1=="Other"):
        d1=2
    if(x3=="No"):
        d3=0
    elif(x3=="Yes"):
        d3=1
    if(x4=="Urban"):
        d4=0
    elif(x4=="Rural"):
        d4=1
    if(x7=="Never_Smoked"):
        d7=0
    elif(x7=="Formerly_Smoked"):
        d7=2
    elif(x7=="Smokes"):
        d7=3
    if 'heart_disease' in request.form:
        arr = np.array([[d1, d2, d3, d4, d5, d6, d7]])
        pred = model1.predict(arr)
        return render_template('after1.html', data=pred)
    if 'stroke' in request.form:
        arr = np.array([[d1, d2, d3, d4, d5, d6, d7]])
        pred = model2.predict(arr)
        return render_template('after2.html', data=pred)


if __name__ == "__main__":
    app.run(debug=False)
