import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

# =====================================
# 1. Read data
# =====================================

np.random.seed(17)

data = pd.read_csv("data2_200x30.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# =====================================
# 2. Split data
# =====================================

X_train = X.iloc[:100]
X_val = X.iloc[100:]

y_train = y.iloc[:100]
y_val = y.iloc[100:]

# =====================================
# 3. Create Pipeline
# =====================================

pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', MinMaxScaler()),
    ('ridge', Ridge())
])

# =====================================
# 4. Define hyperparameters
# =====================================

param_grid = {
    'poly__degree': [1, 2, 3, 4],
    'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000]
}

# =====================================
# 5. Grid Search + Cross Validation
# =====================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

# =====================================
# 6. Best parameters
# =====================================

print("Best Parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# =====================================
# 7. Evaluate the best model
# =====================================

y_pred_train = grid_search.predict(X_train)
y_pred_val = grid_search.predict(X_val)

train_rmse = root_mean_squared_error(y_train, y_pred_train)
val_rmse = root_mean_squared_error(y_val, y_pred_val)

print("Training RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)


#====================================
# visualization  the performance of all hyperparameters
#====================================
import matplotlib.pyplot as plt
import pandas as pd

# Extract all Grid Search results
results = pd.DataFrame(grid_search.cv_results_)

# Convert negative RMSE to positive RMSE
results["CV_RMSE"] = -results["mean_test_score"]

# Plot one curve for each polynomial degree
plt.figure(figsize=(10, 6))

for degree in sorted(results["param_poly__degree"].unique()):
    subset = results[
        results["param_poly__degree"] == degree
    ].sort_values("param_ridge__alpha")

    plt.plot(
        subset["param_ridge__alpha"].astype(float),
        subset["CV_RMSE"],
        marker="o",
        label=f"Degree {degree}"
    )

plt.xscale("log")
plt.xlabel("Ridge Alpha (log scale)")
plt.ylabel("Cross-Validation RMSE")
plt.title("Polynomial Degree and Ridge Alpha vs CV RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()