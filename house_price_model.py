import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Step 1: Load / generate data ----------
np.random.seed(42)
n = 1000

size_sqft = np.random.normal(1800, 650, n).clip(400, 5000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.2, 0.35, 0.25, 0.1, 0.05])
bathrooms = (bedrooms * np.random.uniform(0.5, 1.0, n)).round().clip(1, 5)
age_years = np.random.exponential(15, n).clip(0, 100)
location_score = np.random.uniform(1, 10, n)
distance_to_city_km = np.random.exponential(8, n).clip(0.5, 60)
garage = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

price = (
    50_000 + size_sqft * 120 + bedrooms * 8_000 + bathrooms * 5_000
    - age_years * 600 + location_score * 15_000 - distance_to_city_km * 1_200
    + garage * 10_000 + np.random.normal(0, 25_000, n)
).clip(30_000, None)

df = pd.DataFrame({
    "size_sqft": size_sqft.round(0), "bedrooms": bedrooms, "bathrooms": bathrooms,
    "age_years": age_years.round(1), "location_score": location_score.round(2),
    "distance_to_city_km": distance_to_city_km.round(2), "garage": garage,
    "price": price.round(0),
})

# inject some missing values to simulate real-world messiness
missing_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_idx[:20], "age_years"] = np.nan
df.loc[missing_idx[20:], "bathrooms"] = np.nan
df.to_csv("housing.csv", index=False)

# ---------- Step 2: Explore ----------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nCorrelation with price:\n", df.corr(numeric_only=True)["price"].sort_values(ascending=False))

# ---------- Step 3: Handle missing data ----------
imputer = SimpleImputer(strategy="median")
df[["bathrooms", "age_years"]] = imputer.fit_transform(df[["bathrooms", "age_years"]])

# ---------- Step 4: Train/test split + normalize ----------
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Step 5: Train ----------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------- Step 6: Evaluate ----------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R2:   {r2:.4f}")
