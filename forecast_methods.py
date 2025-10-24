import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet
from prophet import Prophet


def forecast_revenue(df, months_ahead=5, method='LSTM', seq_length=3, plot=True):
    """
    统一接口：根据 clean_df 预测未来营业额。
    
    参数：
        df : pd.DataFrame，包含 ['Transaction Date', 'Total Spent']
        months_ahead : int，要预测的月份数
        method : str, 'LSTM', 'Prophet', 'RandomForest', 'HoltWinters'
        seq_length : int, LSTM 时间窗口长度
        plot : 是否绘制预测图
        
    返回：
        pd.DataFrame，包含历史 + 预测营业额
    """
    
    # 数据处理：按月聚合
    df = df.copy()
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    monthly = df.groupby(df['Transaction Date'].dt.to_period('M')).agg({'Total Spent': 'sum'}).reset_index()
    monthly['Transaction Date'] = monthly['Transaction Date'].dt.to_timestamp()
    monthly = monthly.sort_values('Transaction Date')
    
    if method == 'LSTM':
        # 🔹 LSTM 预测
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(monthly[['Total Spent']])
        
        X, y = [], []
        for i in range(len(scaled) - seq_length):
            X.append(scaled[i:i + seq_length])
            y.append(scaled[i + seq_length])
        X, y = np.array(X), np.array(y)
        
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(seq_length,1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=80, batch_size=4, verbose=0)
        
        last_seq = scaled[-seq_length:].reshape(1, seq_length, 1)
        preds_scaled = []
        for _ in range(months_ahead):
            next_pred = model.predict(last_seq, verbose=0)
            preds_scaled.append(next_pred[0,0])
            last_seq = np.concatenate([last_seq[:,1:,:], next_pred.reshape(1,1,1)], axis=1)
        
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    
    elif method == 'Prophet':
        # 🔹 Prophet 预测
        prophet_df = monthly[['Transaction Date','Total Spent']].rename(columns={'Transaction Date':'ds','Total Spent':'y'})
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=months_ahead, freq='M')
        forecast = model.predict(future)
        preds = forecast['yhat'].iloc[-months_ahead:].values
    
    elif method == 'RandomForest':
        # 🔹 随机森林预测
        monthly['month_num'] = range(len(monthly))
        X = monthly[['month_num']]
        y = monthly['Total Spent']
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        future_months = pd.DataFrame({'month_num': range(len(monthly), len(monthly)+months_ahead)})
        preds = rf.predict(future_months)
    
    elif method == 'HoltWinters':
        # 🔹 Holt-Winters 预测
        series = monthly['Total Spent']
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
        fit_model = model.fit()
        preds = fit_model.forecast(months_ahead)
    
    else:
        raise ValueError("method must be one of ['LSTM','Prophet','RandomForest','HoltWinters']")
    
    # 生成未来日期
    last_date = monthly['Transaction Date'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_ahead)]
    
    # 输出 DataFrame
    pred_df = pd.DataFrame({
        'Transaction Date': list(monthly['Transaction Date']) + future_dates,
        'Predicted Revenue': list(monthly['Total Spent']) + list(preds)
    })
    
    if plot:
        plt.figure(figsize=(9,4))
        plt.plot(monthly['Transaction Date'], monthly['Total Spent'], label='Actual', marker='o')
        plt.plot(future_dates, preds, label='Forecast', marker='x', linestyle='--')
        plt.title(f'{method} Monthly Revenue Forecast')
        plt.xlabel('Month')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return pred_df


# 测试示例
if __name__ == "__main__":
    data = {
        'Transaction Date': pd.date_range('2023-01-01','2023-12-31',freq='D'),
        'Total Spent': np.random.randint(100,400, size=365)
    }
    df = pd.DataFrame(data)
    df_pred = forecast_revenue(df, months_ahead=5, method='Prophet')
    print(df_pred.tail(10))
