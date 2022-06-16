from flask import Flask,render_template,request
import pickle
import numpy as np
import json


model = pickle.load(open("artifacts/model.pkl","rb"))

with open("artifacts/columns_name.json","r") as json_file:
    col_name = json.load(json_file)
col_name_list =col_name['col_name']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def user_info():
    data = request.form
    print(data)

    user_data= np.zeros(len(col_name_list))
    user_data[0] = data['credit_score']
    user_data[1] = data['geography']
    user_data[2] = data['gender']
    user_data[3] = data['age']
    user_data[4] = data['tenure']
    user_data[5] = data['balance']
    user_data[6] = data['numofproducts']
    user_data[7] = data['hascrcard']
    user_data[8] = data['isactivemember']
    user_data[9] = data['estimatedsalary']

    print(user_data)
    result = model.predict([user_data])

    if result[0] == 0:
        modelling_result = "Not Exist"
    else: 
        modelling_result = "Exist"
    
    print(modelling_result)


    return render_template("result.html",prediction =modelling_result)


if __name__=='__main__':
    app.run(host='0.0.0.0',port='8585',debug=True)