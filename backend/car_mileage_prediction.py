# -*- coding: utf-8 -*-
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

def data_clean(data):
  ds = pd.read_csv(data)
  ds["horsepower"] = pd.to_numeric(ds["horsepower"], errors='coerce')
  ds = ds.fillna(ds.mean())
  ds_x = ds.drop('car name',axis=1)
  X = ds_x.drop('mpg', axis=1)
  Y = ds['mpg']
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  return x_train, x_test, y_train, y_test

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils, metrics

def data_scaler(x_train,x_test):
  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  X_test = sc.transform(x_test)
  return x_train , X_test

def RF_prediction(x_train,y_train,x_test, y_test):
  rfc = RandomForestClassifier(n_estimators=5,criterion='entropy')
  x_train, x_test = data_scaler(x_train, x_test)
  if 'continuous' in utils.multiclass.type_of_target(y_train):
    y_train = y_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_train):
    x_train = x_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_test):
    x_test = x_test.astype('int')
  
  rfc.fit(x_train,y_train)
  rfc_pred = rfc.predict(x_test)
  
  return rfc_pred, rfc.score(x_train, y_train), metrics.mean_squared_error(y_test,rfc_pred)


def KNN(x_train,y_train,x_test, y_test):
  knn = KNeighborsClassifier(n_neighbors=25)
  x_train, x_test = data_scaler(x_train, x_test)
  if 'continuous' in utils.multiclass.type_of_target(y_train):
    y_train = y_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_train):
    x_train = x_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_test):
    x_test = x_test.astype('int')

  knn.fit(x_train,y_train)
  knn_pred = knn.predict(x_test)

  return knn_pred, knn.score(x_train, y_train), metrics.mean_squared_error(y_test,knn_pred)

def NB(x_train,y_train,x_test,y_test):
  classifier = GaussianNB()
  x_train, x_test = data_scaler(x_train, x_test)
  if 'continuous' in utils.multiclass.type_of_target(y_train):
    y_train = y_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_train):
    x_train = x_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_test):
    x_test = x_test.astype('int')
  
  classifier.fit(x_train, y_train)
  classifier_pred = classifier.predict(x_test)

  return classifier_pred, classifier.score(x_train,y_train), metrics.mean_squared_error(y_test,classifier_pred)

def LR(x_train,y_train,x_test,y_test):
  LR = LinearRegression()
  x_train, x_test = data_scaler(x_train, x_test)
  if 'continuous' in utils.multiclass.type_of_target(y_train):
    y_train = y_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_train):
    x_train = x_train.astype('int')
  elif 'continuous' in utils.multiclass.type_of_target(x_test):
    x_test = x_test.astype('int')
  
  LR.fit(x_train,y_train)
  lr_pred = LR.predict(x_test)

  return lr_pred, LR.score(x_train,y_train), metrics.mean_squared_error(y_test,lr_pred)

def cov_key(D,y):
  d = {"RF":{"values":[], "score": "", "MSE" : 0.0}, 
        "LR":{"values":[], "score": "", "MSE" : 0.0},
        "NB":{"values":[], "score": "", "MSE" : 0.0},
        "KNN":{"values":[], "score": "", "MSE" : 0.0},
        "actual":{"values": y}}
  old=[]
  new=[]
  for x in D.keys():
    old.append(x)
  for x in d.keys():
    new.append(x)
  for i in range(len(old)):
	  d[new[i]] = D.pop(old[i]) 
  return d



app = Flask('Car_Mileage_Detection')
CORS(app)

@app.route('/preds', methods=['GET', 'POST'])
def pred_val():
    data = "/auto-mpg.csv"#request.args.get("ds-path")
    funcs = [RF_prediction,LR,NB,KNN]
    x_train, x_test, y_train, y_test = data_clean(data)
    y=[]
    for i in y_test:
        y.append(int(i))
    D = {RF_prediction:{"values":[], "score": "", "MSE" : 0.0}, 
        LR:{"values":[], "score": "", "MSE" : 0.0},
        NB:{"values":[], "score": "", "MSE" : 0.0},
        KNN:{"values":[], "score": "", "MSE" : 0.0},
        "actual":{"values": y}}
#'/Users/kartik_rama_arora/Documents/Python_ws/Helm_task-24/auto-mpg.csv'  
    for fc in D.keys():
      if fc == "actual":
        pass
      else:
        x,y,z= fc(x_train,y_train,x_test,y_test)
        D[fc]["values"], D[fc]["score"], D[fc]["MSE"] = list(x), f"{round(y*100)}%", z
    d = cov_key(D,y)
    return f'{d}'

app.run(host="0.0.0.0", port=5960, debug=True)
