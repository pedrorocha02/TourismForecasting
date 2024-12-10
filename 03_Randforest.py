import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from plot import plot
import time

start_time=time.time()

dataset = pd.read_excel('./Dataset_Final.xlsx')
print(dataset.dtypes)

# Separar as features (X) e o alvo (y)
X = dataset.drop('TOL', axis=1)  # Todas as colunas menos 'TOL'
y = dataset['TOL']  # Coluna alvo 'TOL'

# Identificar colunas de tipo 'object' que dão problema com o RandomForest
object_cols = ['Month Name', 'Season']
X = pd.get_dummies(X, columns=object_cols, drop_first=True)
X = X.drop(columns=['Date'])

# Dividir o dataset em conjunto de treino e teste (90% treino, 10% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Definir o modelo base
base_model = RandomForestRegressor()

# Definir os hiperparâmetros
param_dist = {
    'n_estimators': np.arange(100, 1000, step=100),  # Número de árvores na floresta
    'max_depth': [10, 20, 30, None],  # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],  # Número mínimo de amostras em cada folha
    'max_features': ['sqrt', 'log2'], 
    'bootstrap': [True, False], 
    'criterion': ['squared_error'] #Criterio que verifica se um modelo é melhor ou pior que um anterior
}

# Aplicar RandomizedSearchCV
model = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=100,  # Número de combinações a testar
    cv=5,  # Validação cruzada com 5 folds (k=5)
    verbose=2,
    n_jobs=-1, 
    random_state=42,
    error_score='raise'
)

# Treinar o modelo usando RandomizedSearchCV
model.fit(X_train, y_train)

# Mostrar os melhores parâmetros
print("Melhores parâmetros:", model.best_params_)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo usando métricas de avaliação:
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

end_time=time.time()
print("Tempo de Execução ", end_time-start_time)

metrics = {
    'Metric': ['MSE', 'RMSE', 'R²', 'MAE', 'MAPE','ExecTime'],
    'Value': [mse, rmse, r2, mae, mape,end_time-start_time]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel('./Results/03_Randforest_Metrics.xlsx', index=False)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results_df.to_excel('./Results/03_Randforest_Predictions.xlsx', index=False)

# Plot dos valores reais e previstos
plot(dataset,y,y_train,y_pred, 'Random Forest V3')