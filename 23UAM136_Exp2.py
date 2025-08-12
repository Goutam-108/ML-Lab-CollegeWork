import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

df  = pd.read_csv("california_housing_test.csv")
df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("california_housing_test.csv")
print("Columns:", df.columns)  
df.rename(columns={'median_house_value': 'price'}, inplace=True)

features = df.columns[df.columns != 'price']
df[features] = (df[features] - df[features].mean()) / df[features].std()

X = df.drop('price', axis=1).values
y = df['price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

w = np.random.randn(1, X_train.shape[1])
b = 0
learning_rate = 0.01
def for_prop(w, b, X):
    z = np.dot(w, X.T) + b
    return z.T
def cost(z, y):
    m = y.shape[0]
    return (1 / (2 * m)) * np.sum((z - y) ** 2)
def back_prop(z, y, X):
    m = y.shape[0]
    dz = z - y
    dw = (1 / m) * np.dot(dz.T, X)
    db = (1 / m) * np.sum(dz)
    return dw, db
def gradient_descent(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
def linear_model(X_train, y_train, X_test, y_test, epochs):
    global w, b
    losses = []
    for i in range(epochs):
        z = for_prop(w, b, X_train)
        c = cost(z, y_train)
        dw, db = back_prop(z, y_train, X_train)
        w, b = gradient_descent(w, b, dw, db, learning_rate)
        losses.append(c)
        if i % 100 == 0:
            print(f"Epoch {i} - Cost: {c:.4f}")
    return w, b, losses

w, b, losses = linear_model(X_train, y_train, X_test, y_test, epochs=1000)

y_pred = for_prop(w, b, X_test)

mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Loss during Training")
plt.show()

feature_name = 'median_income' 
if feature_name in df.columns:
    feature_index = df.columns.get_loc(feature_name)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, feature_index], y_test, label='Actual')
    plt.scatter(X_test[:, feature_index], y_pred, color='red', alpha=0.5, label='Predicted')
    plt.xlabel(f"{feature_name} (Standardized)")
    plt.ylabel("Price")
    plt.title(f"Regression: {feature_name} vs Price")
    plt.legend()
    plt.show()
else:
    print(f"Feature '{feature_name}' not found.")

residuals = y_pred - y_test
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()
