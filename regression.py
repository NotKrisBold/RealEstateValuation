from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
real_estate_valuation = fetch_ucirepo(id=477)

# Inizializzazione del database
X = pd.DataFrame(real_estate_valuation.data.features, columns=real_estate_valuation.feature_names)
y = pd.Series(real_estate_valuation.data.targets.values.flatten(), name='house price of unit area')

# Calculate average latitude and longitude within the dataset
average_latitude = X['X5 latitude'].mean()
average_longitude = X['X6 longitude'].mean()

# Calculate distances from each data point to the average coordinates
X['distance_to_avg_coords'] = np.sqrt((X['X5 latitude'] - average_latitude)**2 + (X['X6 longitude'] - average_longitude)**2)

# Training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello di regressione lineare
model = LinearRegression()

# Addestramento del modello sul training set
model.fit(X_train, y_train)

# Valutazione del modello sul test set
y_pred = model.predict(X_test)

# Metriche di valutazione
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

# Coefficienti modello
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)

# Grafico residui
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values (10,000 New Taiwan Dollar/Ping)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
