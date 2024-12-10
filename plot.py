import matplotlib.pyplot as plt

def plot(dataset, y, y_train, y_pred,model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset['Date'][len(y_train):], y[len(y_train):], label='Valores Reais')
    plt.plot(dataset['Date'][len(y_train):], y_pred, label='Valores Previstos', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Taxa Líquida de Ocupação-Cama (%)')
    plt.title('Comparação entre Valores Reais e Previstos: ' + model_name)
    plt.legend()
    plt.show()