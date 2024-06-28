#MacOS 兼容性操作
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from fetch_btc_data import get_crypto_data
from BTC_predict import get_today_price

API_KEY = ''  # 替换为您的API密钥

get_crypto_data(API_KEY)

today_price_close, today_price_open, today_price_high, today_price_low = get_today_price()

print('预测今天的收盘价位{}和{}之间'.format(today_price_low, today_price_high))
