from flask import Flask
#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/make_preds',methods = ['GET','POST'])
def make_preds(to_predict_list):
  
  sq_mt_built=float(to_predict_list[0])
  n_bathrooms=float(to_predict_list[1])
  n_rooms=float(to_predict_list[2])
  has_lift=bool(to_predict_list[3])
  house_type_entrada=to_predict_list[4]
  
  #'HouseType 1: Pisos', 'HouseType 4: Dúplex','HouseType 5: Áticos', 'HouseType 2: Casa o chalet'
  if house_type_entrada=="piso":
      house_type_id= 'HouseType 1: Pisos'  
  if house_type_entrada=="duplex":
      house_type_id= 'HouseType 4: Dúplex'   
  if house_type_entrada=="atico":
      house_type_id= 'HouseType 5: Áticos' 
  if house_type_entrada=="casa":
      house_type_id= 'HouseType 2: Casa o chalet'  

  import pickle
  import pandas as pd

  # Load Files  
  encoder_fit =pickle.load(open("app/encoder.pickle","rb"))
  rf_reg_fit = pickle.load(open("app/model.pickle","rb"))
    
 
  # Create df
  x_pred = pd.DataFrame(
    [[sq_mt_built, n_bathrooms, n_rooms, has_lift, house_type_id]],
    columns = ['sq_mt_built', 'n_bathrooms', 'n_rooms', 'has_lift', 'house_type_id'])

  # One hot encoding
  encoded_data_pred = pd.DataFrame( encoder_fit.transform(x_pred['house_type_id']),columns = encoder_fit.classes_.tolist()) 

  # Build final df
  x_pred_transf = pd.concat([x_pred.reset_index(), encoded_data_pred], axis = 1).drop(['house_type_id', 'index'], axis = 1)

  preds= rf_reg_fit.predict(x_pred_transf)

  
  return round(preds[0])


@app.route('/result',methods = ['Get','POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        try:
          to_predict_list = list(map(str,to_predict_list))
          prediction = make_preds(to_predict_list)
        except ValueError:
           prediction='para no deprimirme: 123435'

        return render_template("result.html", prediction=prediction)




if __name__=="__main__":
    app.debug = True
    app.run(port=5001)
