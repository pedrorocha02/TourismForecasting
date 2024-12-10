import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from plot import plot
import time

start_time=time.time()

# Carregar o dataset
dataset = pd.read_excel('./Dataset_Final.xlsx')

# Converter a coluna 'Date' para datetime e remover valores ausentes
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
dataset = dataset.dropna(subset=['Date', 'TOL'])

# Coluna alvo 'TOL'
y = dataset['TOL']

# Dividir os dados em treino e teste usando a data especificada
train_end_date = pd.Timestamp("2021-12-31")
train_mask = dataset['Date'] <= train_end_date

# Dividir a série temporal em treino e teste com base na data especificada
y_train, y_test = y[train_mask], y[~train_mask]

# Definir os hiperparametros do modelo SARIMAX (Apenas para a Série Temporal sem variáveis exógenas)
p, d, q = 1, 1, 1  # Parâmetros ARIMA
P, D, Q, s = 1, 1, 1, 12  # Parâmetros sazonais (s=12 para anualidade)

model = SARIMAX(
    y_train,
    order=(p, d, q),
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Treinar o modelo
sarimax_model = model.fit(disp=False)

# Fazer previsões no conjunto de teste
y_pred = sarimax_model.predict(start=len(y_train), end=len(y)-1)

# Avaliar o modelo usando métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

end_time=time.time()

metrics = {
    'Metric': ['MSE', 'RMSE', 'R²', 'MAE', 'MAPE','ExecTime'],
    'Value': [mse, rmse, r2, mae, mape,end_time-start_time]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel('./Results/01_SARIMAX_Metrics.xlsx', index=False)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred.values
})
results_df.to_excel('./Results/01_SARIMAX_Predictions.xlsx', index=False)

# Plot dos valores reais e previstos
plot(dataset,y,y_train,y_pred, 'SARIMAX V1')