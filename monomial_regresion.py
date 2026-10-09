import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error


# 1. Fixed seed
np.random.seed(17)


# 2. Read dataset
data = pd.read_csv("data2_200x30.csv")

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()


# 3. Split into training and validation
X_train = X[:100]
X_val = X[100:]

y_train = y[:100]
y_val = y[100:]


# 4. Define monomial polynomial features
def monomials_poly_features(X, degree):
    X = np.asarray(X)

    X_poly = X.copy()

    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))

    return X_poly


# 5. Test different degrees
degrees = [1, 2, 3, 4,5]

train_rmse = []
val_rmse = []

for degree in degrees:

    # Generate monomial features
    X_train_poly = monomials_poly_features(X_train, degree)
    X_val_poly = monomials_poly_features(X_val, degree)

    # Scale features using training data only
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)

    # Train model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Calculate RMSE
    train_error = root_mean_squared_error(y_train, y_train_pred)
    val_error = root_mean_squared_error(y_val, y_val_pred)

    train_rmse.append(train_error)
    val_rmse.append(val_error)

    print(f"Degree: {degree}")
    print(f"Number of features: {X_train_poly.shape[1]}")
    print(f"Train RMSE: {train_error:.4f}")
    print(f"Validation RMSE: {val_error:.4f}")
    print("-" * 35)


# 6. Plot results
plt.figure(figsize=(9, 5))

plt.plot(degrees, train_rmse, marker="o", label="Training RMSE")
plt.plot(degrees, val_rmse, marker="o", label="Validation RMSE")

plt.xticks(degrees)
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Polynomial Regression Without Cross Features")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()