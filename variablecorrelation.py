import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './Dataset_Final.xlsx'
df = pd.read_excel(file_path)

df_numeric = df.select_dtypes(exclude=['object'])

# Calcular a correlação entre as variáveis numéricas
correlation_matrix = df_numeric.corr()

# Focar na correlação com a variável alvo 'TOL' e remover 'TOL' do gráfico
target_correlation = correlation_matrix['TOL'].drop('TOL').sort_values(ascending=False)

# Criar um gráfico de barras para a correlação com a variável alvo 'TOL'
plt.figure(figsize=(10, 8))
sns.barplot(x=target_correlation.values, y=target_correlation.index, palette='coolwarm')
plt.title('Correlação das Variáveis com TOL')
plt.ylabel('Variáveis')
plt.xlabel('Correlação')
plt.xlim(-0.6, 0.6)
plt.show()

print(target_correlation)
