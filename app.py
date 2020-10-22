import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import pickle
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('done.pkl', 'rb'))
model1=pickle.load(open('noWaist.pkl','rb'))
model2=pickle.load(open('noTrigs983.pkl','rb'))
model3=pickle.load(open('both.pkl','rb'))
int_features1=[]



@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if("text" == "M"):
        text = 0
    else:
        text =1
    int_features1=[]

    int_features = [float(x) for x in request.form.values()]
    for i in int_features:
        #for male, it is zero
        if i != 100:
            int_features1.append(i)
        
    final_features = [np.array(int_features1)]
    
    if int_features[3]==100 and int_features[9]==100:
        prediction = model3.predict(final_features)
    elif int_features[9]==100:
        prediction = model2.predict(final_features)
    elif int_features[3]==100:
        prediction = model1.predict(final_features)
    else:
        prediction = model.predict(final_features)


    # for i in final_features:
    #     if ct==3 and i==0:    
    #         prediction = model1.predict(final_features)
    #     elif ct==9 and i==0:
    #         prediction = model2.predict(final_features)
        


            
            
        
    #     if i==0:
    #         if ct==3:    
    #             prediction = model.predict(final_features)

                
                
    #             #waist
    #         elif ct==9:  
    #             prediction = model.predict(final_features)
    #     elif 

    #             #trigs
    #     ct=ct
            

    output = prediction[0]
    if output==0:
        output1='Normal'
    elif output==1:
        output1='Diabetic'
    elif output==2:
        output1='Pre-Diabetic'

    return render_template('index1.html', prediction_text='The Patient is {}'.format(output1))
    
if __name__ == "__main__":
    app.run(debug=True) 