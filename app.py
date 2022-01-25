from flask import Flask, request, jsonify
import numpy as np
import pickle
model = pickle.load(open('C:/Users/satvi/Documents/modell.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST','GET'])
def predict():
    age = request.args.get('age')
    gender = request.args.get('gender')
    height = request.args.get('height')
    weight = request.args.get('weight')
    bmi = (float(weight)/(float(height)/100)**2)
    ap_hi = request.args.get('ap_hi')
    ap_lo = request.args.get('ap_lo')
    cholestrol = request.args.get('cholestrol')
    if cholestrol=="Normal":
        cholestrol_normal = 1
        cholestrol_aboveNormal = 0
        cholestrol_wellAboveNormal = 0
    elif cholestrol =="Above Normal":
        cholestrol_normal = 0
        cholestrol_aboveNormal = 1
        cholestrol_wellAboveNormal = 0
    else:
        cholestrol_normal = 0
        cholestrol_aboveNormal = 0
        cholestrol_wellAboveNormal = 1
    gluc = request.args.get('gluc')
    if gluc=="Normal":
        gluc_normal = 1
        gluc_aboveNormal = 0
        gluc_wellAboveNormal = 0
    elif cholestrol =="Above Normal":
        gluc_normal = 0
        gluc_aboveNormal = 1
        gluc_wellAboveNormal = 0
    else:
        gluc_normal = 0
        gluc_aboveNormal = 0
        gluc_wellAboveNormal = 1
    smoke = request.args.get('smoke')
    alco = request.args.get('alco')
    active = request.args.get('active')
    input_query = np.array([[age,gender,ap_hi, ap_lo, weight, height, smoke, alco, active, cholestrol, gluc]])
    input_query1 = np.array([[age,gender,ap_hi, ap_lo, smoke, alco, active, bmi, cholestrol_aboveNormal, cholestrol_normal, cholestrol_wellAboveNormal, gluc_aboveNormal, gluc_normal, gluc_wellAboveNormal]])
    result = model.predict(input_query1)[0]
    return jsonify({'Cardio':str(result)})
if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)