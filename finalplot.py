import pandas as pd
import matplotlib.pyplot as plt

file_path = './Results/FinalPlot.xlsx'
data = pd.read_excel(file_path, sheet_name='Folha1')

data['Date'] = pd.to_datetime(data['Date'])

plt.figure(figsize=(14, 8))

# Plot the 'Real' column with emphasis
plt.plot(data['Date'], data['Real'], label='Real', color='red', linewidth=3, marker='o')

# Plot the other columns with less emphasis
for column in data.columns[2:]:  # Exclude 'Date' and 'Real' columns
    plt.plot(data['Date'], data[column], label=column, linewidth=1.25, linestyle='--', alpha=1)

plt.xlabel('Data', fontsize=12)
plt.ylabel('Taxa Líquida de Ocupação-Cama (%)', fontsize=12)
plt.title('Comparação entre Valores Reais e Previstos pelos diferentes modelos', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()