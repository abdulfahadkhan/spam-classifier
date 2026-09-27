import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # so it saves to file without needing a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Load & preprocess (same as before) ----------
df = pd.read_csv("housing.csv")
imputer = SimpleImputer(strategy="median")
df[["bathrooms", "age_years"]] = imputer.fit_transform(df[["bathrooms", "age_years"]])

X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 1. Compare multiple models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

print("=== Model Comparison ===")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2, model, y_pred)
    print(f"{name:20s} MSE: {mse:,.2f}   R2: {r2:.4f}")

# ---------- 2. Cross-validation (on Linear Regression) ----------
print("\n=== 5-Fold Cross-Validation (Linear Regression) ===")
X_all_scaled = scaler.fit_transform(X)  # refit scaler on full data for CV
cv_scores = cross_val_score(LinearRegression(), X_all_scaled, y, cv=5, scoring="r2")
print(f"R2 per fold: {np.round(cv_scores, 4)}")
print(f"Mean R2: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

# ---------- 3. Feature importance (Linear Regression coefficients) ----------
lr_model = results["Linear Regression"][2]
print("\n=== Linear Regression Coefficients (standardized) ===")
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lr_model.coef_
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))

# ---------- 4. Feature importance (Random Forest) ----------
rf_model = results["Random Forest"][2]
print("\n=== Random Forest Feature Importances ===")
imp_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
print(imp_df.to_string(index=False))

# ---------- 5. Visualization: Actual vs Predicted (best model) ----------
best_name = min(results, key=lambda k: results[k][0])  # lowest MSE
best_pred = results[best_name][3]
print(f"\nBest model by MSE: {best_name}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.5)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, 'r--', label='Perfect prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Price ({best_name})')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=100)
print("Saved actual_vs_predicted.png")

# ---------- 6. Visualization: Feature importance bar chart ----------
plt.figure(figsize=(8, 5))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
print("Saved feature_importance.png")
