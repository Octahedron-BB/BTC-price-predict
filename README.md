训练模型以进行BTC日线级别价格预测

数据采用了2014/01/01-2024/06/17的BTC日线级别的Close、Open、High、Low、volume 

ver1使用经典RNN模型进行预测

选取2014/01/01-2023/12/31作为训练集

只使用了Close数据作为输入输出

参数:

```python
time_step = 30

model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(time_step, 1), activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, batch_size=30, epochs=200)
```

训练集预测结果与原结果比较:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver1/train.png)

测试集预测结果与原结果比较:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver1/test.png)

选取近30天数据(2024/05/19-2024/06/17)通过迭代预测进行未来365天的数据预测:

发现效果很差,考虑过拟合?RNN模型问题?调整模型参数?预测时间太长?

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver1/predict.png)


ver2使用LSTM模型进行预测

选取2014/01/01-2022/05/15作为训练集

使用Close、Open、High、Low、volume五个维度的数据进行输入输出

参数:

```python
time_step = 30

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 5), activation='relu'))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(units=25, activation='linear'))
model.add(Dense(5))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, batch_size=32, epochs=50)
```

训练集预测结果与原结果比较:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver2/train.png)

测试集预测结果与原结果比较:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver2/test.png)

选取近30天数据(2024/05/19-2024/06/17)通过迭代预测进行未来30天的数据预测,效果一般,考虑再次调整参数?结合其他模型?

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_train/ver2/predict.png)

LSTM模型已导出至[这里](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/BTC_predict_test3.keras)

归一化方法已导出至[这里](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/scaler.pkl)

可以使用[预测文件](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/BTC_predict_result.ipynb)进行预测

需要给予至少30天的Close、Open、High、Low、volume数据(BTC_30D.csv)



-----------------------------2024/06/28更新-----------------------------

1、采用CNN模型对真实价格与预测价格再次进行建模

文件在[这里](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/BTC_retrain.ipynb)

因为效果很差所以不做过多介绍

2、受好朋友的启发,将预测方向从预测趋势转为尝试预测每日close价格的可能范围

仍使用ver2所建立的模型,在输出预测的close价格之外还输出了预测的high,low,open价格并进行可视化

结果如下:

整个测试集数据:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/all_price.png)

截取部分数据:

![image](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/all_price_csv.png)

对测试集的预测结果进行统计分析:真实close价格位于预测的high,low价格之间的概率为0.789617486

算是可以接受,但目前仍存在的问题是预测得到的high,low价格间相差较大,可能导致实际应用价值不高

预测方法放在了[这里](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/BTC_price_all.ipynb)

结果数据输出为csv放在了[这里](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price_predict/BTC_predict_allprice.csv)


-------------------------------------------------------------------

使用cryptocompare进行了每日BTC价格的获取[BTC predict/BTC_price/fetch_btc_data.py](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price/fetch_btc_data.py)
(连续30天,截至昨日(包含昨日))

获取到的数据存储为BTC_30D.csv

之后载入模型进行预测[BTC predict/BTC_price/BTC_predict.py](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price/BTC_predict.py)

模型预测后返回预测的close, open, high, low价格数据

可以直接使用[BTC predict/BTC_price/get_predict.py](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price/get_predict.py)进行预测

需要将[BTC predict/BTC_price](https://github.com/Octahedron-BB/BTC-price-predict/tree/main/BTC%20predict/BTC_price)下的文件全部下载,并将[BTC predict/BTC_price/get_predict.py](https://github.com/Octahedron-BB/BTC-price-predict/blob/main/BTC%20predict/BTC_price/get_predict.py)中的API_KEY修改为cryptocompare的API key

最后得到的预测数据为今日收盘价预测会处于的价格范围


-----------------------------TODO LIST-----------------------------

- [ ] TODO1:更改模型参数寻找更优模型:调整LSTM单元数量,增加层数,加入Dropout层,尝试不同的批量大小和训练次数,尝试不同的学习率
- [ ] TODO2:加入更多的特征，如技术指标（MA, RSI, MACD等）、交易量、市场情绪（例如推文情绪得分）等
- [ ] TODO3:使用交叉验证来评估模型性能,使用均方根误差（RMSE）、平均绝对误差（MAE）等指标来评估模型
- [ ] TODO4:结合其他模型(LSTM、GRU、ARIMA等)进行模型优化
- [x] TODO5:引入api或使用爬虫获取每日BTC价格,每日更新BTC_30D.csv以进行预测
- [ ] TODO6:滞后性问题?



