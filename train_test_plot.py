import matplotlib.pyplot as plt
import pandas as pd

def plotDatasetPartition(df, dataSplitDate):
    df['Date'] = pd.to_datetime(df['Date'])
    train = df[df['Date'] <= dataSplitDate]
    test = df[df['Date'] > dataSplitDate]

    plt.figure(figsize=(10, 6))
    plt.plot(train['Date'], train['TOL'], label='Treino')
    plt.plot(test['Date'], test['TOL'], label='Teste')
    plt.xlabel('Data')
    plt.ylabel('Taxa Líquida de Ocupação-Cama (%)')
    plt.title('Distribuição dos Conjuntos de Treino e Teste')
    plt.legend()
    plt.show()

dataset = pd.read_excel('./Dataset_Final.xlsx')
dataSplitDate='2021-12-31'
plotDatasetPartition(dataset,dataSplitDate)