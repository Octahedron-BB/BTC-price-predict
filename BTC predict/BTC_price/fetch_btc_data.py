import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


def get_crypto_data(api_key):
    # 定义时间范围
    end_date = datetime.now(timezone.utc) - timedelta(days=1)  # 昨天
    end_date_timestamp = int(end_date.timestamp())

    # 定义URL
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=29&toTs={end_date_timestamp}&api_key={api_key}'
    
    # 发送请求
    response = requests.get(url)
    data = response.json()
    
    # 解析数据
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 重命名列并选择需要的列
    df.rename(columns={'time': 'timeOpen', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volumefrom': 'volume'}, inplace=True)
    df = df[['timeOpen', 'Close', 'Open', 'High', 'Low', 'volume']]
    
    # 保存为CSV文件
    df.to_csv('BTC_30D.csv', index=False)
    print(f"数据已保存到 BTC_30D.csv")
