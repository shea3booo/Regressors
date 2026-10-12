import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import root_mean_squared_error


# Data reading
np.random.seed(17)

data = pd.read_csv("data2_200x30.csv")


# Select one feature
selected_feature = int(
    input("Enter a feature index from 0 to 29: ")
)

if not 0 <= selected_feature < 30:
    raise ValueError("Feature index must be between 0 and 29.")


X = data.iloc[:, [selected_feature]]
y = data.iloc[:, -1]


# Data splitting
X_train = X.iloc[:100]
X_val = X.iloc[100:]

y_train = y.iloc[:100]
y_val = y.iloc[100:]


degrees = [1, 2, 3]

rmse_train = []
rmse_val = []


# Train a separate model for each degree
for degree in degrees:

    # Polynomial transformation
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Scaling
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)

    # Model training
    model = LinearRegression(fit_intercept=True)

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Evaluation
    train_error = root_mean_squared_error(
        y_train, y_train_pred
    )

    val_error = root_mean_squared_error(
        y_val, y_val_pred
    )

    rmse_train.append(train_error)
    rmse_val.append(val_error)

    print(f"Feature: {selected_feature}, Degree: {degree}")
    print(f"Train RMSE: {train_error:.4f}")
    print(f"Validation RMSE: {val_error:.4f}")
    print("-" * 30)


# Compare training and validation errors
results = pd.DataFrame({
    "Degree": degrees,
    "Train RMSE": rmse_train,
    "Validation RMSE": rmse_val
})

results.plot(
    x="Degree",
    y=["Train RMSE", "Validation RMSE"],
    kind="bar",
    figsize=(9, 5)
)

plt.title(f"Feature {selected_feature}: Train vs Validation RMSE")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()