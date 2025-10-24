import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


def predict_next_months_df(df, months_ahead=5, seq_length=3, plot=True):
    """
    使用 LSTM 模型预测未来 months_ahead 个月的营业额。
    
    参数：
        df : pd.DataFrame，包含 ['Transaction Date', 'Total Spent']
        months_ahead : 预测的月份数
        seq_length : LSTM 时间窗口长度（默认 3）
        plot : 是否绘制预测图
    
    返回：
        pd.DataFrame，包含历史数据 + 预测数据
    """

    # 🧹 Step 1: 清洗与聚合
    df = df.copy()
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    monthly = (
        df.groupby(df['Transaction Date'].dt.to_period('M'))
          .agg({'Total Spent': 'sum'})
          .reset_index()
    )
    monthly['Transaction Date'] = monthly['Transaction Date'].dt.to_timestamp()
    monthly = monthly.sort_values('Transaction Date')

    # 🧮 Step 2: 数据归一化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(monthly[['Total Spent']])

    # 🧩 Step 3: 构造训练集
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i + seq_length])
        y.append(scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    # 🧠 Step 4: 构建 LSTM 模型
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 🚀 Step 5: 训练模型
    model.fit(X, y, epochs=80, batch_size=4, verbose=0)

    # 🔮 Step 6: 预测未来 months_ahead
    last_seq = scaled[-seq_length:].reshape(1, seq_length, 1)
    preds_scaled = []

    for _ in range(months_ahead):
        next_pred = model.predict(last_seq, verbose=0)
        preds_scaled.append(next_pred[0, 0])
        # 滚动窗口更新
        last_seq = np.concatenate([last_seq[:, 1:, :], next_pred.reshape(1, 1, 1)], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # 📅 Step 7: 生成未来日期
    last_date = monthly['Transaction Date'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(months_ahead)]

    # 📊 Step 8: 输出结果
    pred_df = pd.DataFrame({
        'Transaction Date': list(monthly['Transaction Date']) + future_dates,
        'Predicted Revenue': list(monthly['Total Spent']) + list(preds)
    })

    # 🎨 Step 9: 可视化
    if plot:
        plt.figure(figsize=(9, 4))
        plt.plot(monthly['Transaction Date'], monthly['Total Spent'], label='Actual', marker='o')
        plt.plot(future_dates, preds, label='Forecast', marker='x', linestyle='--')
        plt.title('LSTM Monthly Revenue Forecast')
        plt.xlabel('Month')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pred_df


