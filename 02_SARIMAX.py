import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
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
y_train, y_test = y[train_mask], y[~train_mask]

# Definir o grid de hiperparâmetros para o SARIMAX
param_grid = {
    'p': [0, 1, 2],
    'd': [1],
    'q': [0, 1, 2],
    'P': [0, 1],
    'D': [1],
    'Q': [0, 1],
    's': [12]  # Periodicidade sazonal (por exemplo, 12 para anualidade)
}

# Variáveis para armazenar o melhor modelo e o menor erro
best_score = float("inf")
best_params = None
best_model = None

# Loop pelos parâmetros
for params in ParameterGrid(param_grid):
    try:
        # Definir o modelo SARIMAX com os parâmetros atuais
        model = SARIMAX(
            y_train,
            order=(params['p'], params['d'], params['q']),
            seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Treinar o modelo
        sarimax_model = model.fit(disp=False)
        
        # Fazer previsões no conjunto de teste
        y_pred = sarimax_model.predict(start=len(y_train), end=len(y)-1)
        
        # Avaliar o modelo usando a métrica RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Atualizar o melhor modelo se o RMSE for menor
        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = sarimax_model
            
    except Exception as e:
        print(f"Erro com parâmetros {params}: {e}")

# Exibir os melhores parâmetros e o menor RMSE
print(f"Melhores parâmetros: {best_params}")
print(f"Melhor RMSE: {best_score:.2f}")

# Fazer previsões com o melhor modelo
y_pred_best = best_model.predict(start=len(y_train), end=len(y)-1)

# Avaliar o melhor modelo com métricas adicionais
mse = mean_squared_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_best)
mape = np.mean(np.abs((y_test - y_pred_best) / y_test)) * 100

end_time=time.time()

metrics = {
    'Metric': ['MSE', 'RMSE', 'R²', 'MAE', 'MAPE','ExecTime'],
    'Value': [mse, rmse, r2, mae, mape,end_time-start_time]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel('./Results/02_SARIMAX_Metrics.xlsx', index=False)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best.values
})
results_df.to_excel('./Results/02_SARIMAX_Predictions.xlsx', index=False)

plt.figure(figsize=(12, 6))
plt.plot(dataset['Date'][len(y_train):], y[len(y_train):], label='Valores Reais')
plt.plot(dataset['Date'][len(y_train):], y_pred_best, label='Valores Previstos', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Taxa Líquida de Ocupação-Cama (%)')
plt.title('Comparação entre Valores Reais e Previstos: SARIMAX V2')
plt.legend()
plt.show()
