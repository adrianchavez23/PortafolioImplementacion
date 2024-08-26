import pandas as pd
import numpy as np

#CARGAR DATASETS
df = pd.read_csv('informacion_nutrimental.csv')

df.columns = df.columns.str.strip()

features = ['Carbohidratos','Lipidos','Proteina']
label = ['Calorias']

X = df[features]
y = df[label]

def divide_training_test(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)  
    
    total_size = len(X)
    test_size = int(total_size * test_size)
    
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices].values.ravel()  
    y_test = y.iloc[test_indices].values.ravel() 
    
    return X_train, X_test, y_train, y_test

#ENTRENAMIENTO
def update_w_and_b(X, y, w, b, alpha):
    
    N = len(X)
    
    predictions = np.dot(X, w) + b
    
    dl_dw = -2 * np.dot(X.T, (y - predictions)) / N
    dl_db = -2 * np.sum(y - predictions) / N

    w = w - alpha * dl_dw
    b = b - alpha * dl_db
    
    return w, b


def avg_loss(X, y, w, b):
    
    N = len(X)
    predictions = np.dot(X, w) + b
    total_error = np.sum((y - predictions) ** 2)
    return total_error / N

def train(X, y, w, b, alpha, epochs):

  print('ENTRENAMIENTO::')
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)
  # log the progress
    if e % 400 == 0:
      avg_loss_ = avg_loss(X, y, w, b)
      print("Epoch {} | Loss: {} | w:{}, b:{}".format(e, avg_loss_, w, b))
  return w, b


def predict(X, w, b):
    return np.dot(X, w) + b


def calcula_mse(y_true, y_pred):
    N = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / N
    return mse


#DIVISIÓN DE DATASETS
X_train, X_test, y_train, y_test = divide_training_test(X, y, seed=42)


#PARÁMETROS INICIALES
w = np.ones(X_train.shape[1]) 
b = 0.0
alpha = 0.00001
epochs = 12000

w, b = train(X_train.values, y_train, w, b, alpha, epochs)  

predictions = predict(X_test.values, w, b)

mse_test = calcula_mse(y_test, predictions)

print("\nPREDICCIONES:")

for i, (pred, real) in enumerate(zip(predictions, y_test)):
    print(f"X_test[{i}] = {X_test.iloc[i].values},  y_pred = {round(pred, 4)}, y_real = {real}")

print(f"\nMSE: {mse_test}")