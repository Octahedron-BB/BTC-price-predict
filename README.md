训练一个模型以进行BTC日线级别价格预测

数据采用了2014/01/01-2024/06/17的BTC日线级别的Close、Open、High、Low、volume 

其中,ver1尝试使用经典RNN模型进行预测
选取2014/01/01-2023/12/31作为训练集
只使用了Close数据作为输入输出

参数:
model.add(SimpleRNN(units=50, input_shape=(time_step, 1), activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
time_step = 30
model.fit(X, y, batch_size=30, epochs=200)





ver2使用LSTM模型进行预测
参数:
