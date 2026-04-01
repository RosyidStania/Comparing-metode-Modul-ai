# ==========================================
# 1. IMPORT LIBRARY
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 2. BACA & PREPROCESSING DATA (COFFEE SALES)
# ==========================================
file_path = 'Coffe_sales.xlsx - index_1.csv'  # Nama file sesuai dataset Anda

print(f"Membaca data dari: {file_path} ...")
df = pd.read_csv(file_path)

# Preprocessing Tanggal
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Mengelompokkan total MONEY (Pendapatan) per hari
df_daily = df.groupby('date')['money'].sum().reset_index()
df_daily.set_index('date', inplace=True)

# Resample menjadi data Mingguan (W) agar tren lebih stabil
y = df_daily['money'].resample('W').sum()

# Membagi data (80% Train, 20% Test)
train_size = int(len(y) * 0.8)
train, test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Total Data Mingguan: {len(y)} minggu")
print(f"Data Training      : {len(train)} minggu")
print(f"Data Testing       : {len(test)} minggu\n")

# ==========================================
# 3. FUNGSI METRIK EVALUASI
# ==========================================
def smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

def mase(y_true, y_pred, y_train):
    naive_error = np.mean(np.abs(np.diff(y_train)))
    return mean_absolute_error(y_true, y_pred) / naive_error if naive_error != 0 else np.nan

def evaluate_model(name, y_true, y_pred, y_train, aic='N/A', bic='N/A'):
    return {
        'Model': name,
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'MAPE (%)': round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        'SMAPE (%)': round(smape(y_true, y_pred), 2),
        'MASE': round(mase(y_true, y_pred, y_train), 2),
        'R²': round(r2_score(y_true, y_pred), 2),
        'AIC': round(aic, 2) if isinstance(aic, (int, float)) else aic,
        'BIC': round(bic, 2) if isinstance(bic, (int, float)) else bic
    }

metrics_list = []
predictions = pd.DataFrame(index=test.index)
predictions['Aktual'] = test

# ==========================================
# 4. TRAINING & PREDIKSI MODEL
# ==========================================

# A. Holt-Winters
print("Melatih model Holt-Winters...")
hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4).fit()
predictions['Holt-Winters'] = hw_model.forecast(len(test))
metrics_list.append(evaluate_model('Holt-Winters', test, predictions['Holt-Winters'], train, hw_model.aic, hw_model.bic))

# B. SARIMA
print("Melatih model SARIMA...")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)).fit(disp=False)
predictions['SARIMA'] = sarima_model.forecast(len(test))
metrics_list.append(evaluate_model('SARIMA', test, predictions['SARIMA'], train, sarima_model.aic, sarima_model.bic))

# C. Random Forest (dengan fitur lag)
print("Melatih model Random Forest...")
def create_lag_features(series, lag=4):
    df_rf = series.to_frame(name='money')
    for i in range(1, lag + 1):
        df_rf[f'lag_{i}'] = df_rf['money'].shift(i)
    df_rf['month'] = df_rf.index.month
    df_rf['week'] = df_rf.index.isocalendar().week
    return df_rf.dropna()

full_rf = create_lag_features(pd.concat([train, test]), lag=4)
X_train_rf = full_rf.loc[train.index.intersection(full_rf.index)].drop('money', axis=1)
y_train_rf = full_rf.loc[train.index.intersection(full_rf.index)]['money']
X_test_rf = full_rf.loc[test.index.intersection(full_rf.index)].drop('money', axis=1)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
predictions['Random Forest'] = rf_model.predict(X_test_rf)
metrics_list.append(evaluate_model('Random Forest', test[4:] if len(test)>4 else test, predictions['Random Forest'], train))

# D. LSTM
print("Melatih model LSTM...")
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
look_back = 4

X_train_lstm = np.array([train_scaled[i-look_back:i, 0] for i in range(look_back, len(train_scaled))])
y_train_lstm = np.array([train_scaled[i, 0] for i in range(look_back, len(train_scaled))])
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

lstm_model = Sequential([LSTM(50, input_shape=(look_back, 1)), Dense(1)])
lstm_model.compile(loss='mse', optimizer='adam')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=4, verbose=0)

inputs = y[len(y) - len(test) - look_back:].values.reshape(-1, 1)
inputs_scaled = scaler.transform(inputs)
X_test_lstm = np.array([inputs_scaled[i-look_back:i, 0] for i in range(look_back, len(inputs_scaled))])
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
predictions['LSTM'] = scaler.inverse_transform(lstm_pred).flatten()
metrics_list.append(evaluate_model('LSTM', test, predictions['LSTM'], train))

# E. Prophet
print("Melatih model Prophet...")
train_p = train.reset_index().rename(columns={'date': 'ds', 'money': 'y'})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(train_p)
future = prophet_model.make_future_dataframe(periods=len(test), freq='W')
forecast = prophet_model.predict(future)
predictions['Prophet'] = forecast.set_index('ds').loc[test.index, 'yhat'].values
metrics_list.append(evaluate_model('Prophet', test, predictions['Prophet'], train))

# ==========================================
# 5. HASIL & VISUALISASI
# ==========================================
df_metrics = pd.DataFrame(metrics_list)
print("\n=== HASIL EVALUASI MODEL (COFFEE SALES) ===")
print(df_metrics.to_string(index=False))

# Plot Perbandingan
plt.figure(figsize=(15, 7))
plt.plot(train, label='Train', color='gray', alpha=0.5)
plt.plot(test, label='Aktual', color='black', marker='o')
for col in predictions.columns[1:]:
    plt.plot(predictions[col], label=col, linestyle='--')
plt.title('Prediksi Total Penjualan Kopi Mingguan (Money)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Confusion Matrix Tren
print("\n=== CONFUSION MATRIX TREN (NAIK/TURUN) ===")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
actual_trend = (test.values > np.append([train.iloc[-1]], test.values[:-1])).astype(int)

for i, model in enumerate(predictions.columns[1:]):
    pred_trend = (predictions[model].values > np.append([train.iloc[-1]], predictions[model].values[:-1])).astype(int)
    cm = confusion_matrix(actual_trend, pred_trend)
    ConfusionMatrixDisplay(cm, display_labels=['Turun', 'Naik']).plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(model)
plt.tight_layout()
plt.show()