#MacOS 兼容性操作
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#加载模块
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import nbimporter
import joblib



def get_today_price():

    #加载模型  
    model = load_model('BTC_predict_model.keras')
    # 加载scaler
    scaler = joblib.load('scaler.pkl')
    #加载数据
    data_30d = pd.read_csv('BTC_30D.csv')
    
    #数据归一化
    price_30d = data_30d.drop(['timeOpen'], axis=1)
    scaled_30d_data = scaler.transform(price_30d)
    #将数据转换为可以输入模型的shape
    scaled_30d_data = scaled_30d_data.reshape(1, 30, 5)

    #模型预测
    today_price_norm = model.predict(scaled_30d_data)

    #反归一化
    today_price = scaler.inverse_transform(today_price_norm)
    #获取各个属性数值
    today_price_close = today_price[:, 0]
    today_price_open = today_price[:, 1]
    today_price_high = today_price[:, 2]
    today_price_low = today_price[:, 3]
    print('预测完成')

    return today_price_close, today_price_open, today_price_high, today_price_low

