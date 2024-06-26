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

- [ ] TODO1:更改模型参数寻找更优模型
- [ ] TODO2:结合其他模型(CNN?)进行模型优化
- [ ] TODO3:引入api或使用爬虫获取每日BTC价格,每日更新BTC_30D.csv以进行预测

