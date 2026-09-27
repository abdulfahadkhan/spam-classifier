# House Price Prediction

Predicts house prices from features like size, bedrooms, location score, and age using Linear Regression (with Ridge/Lasso/Random Forest comparison).

## Steps
1. Load/generate data
2. Explore distributions & correlations
3. Handle missing data (median imputation)
4. Normalize features (StandardScaler)
5. Train/test split (80/20)
6. Train models (Linear Regression, Ridge, Lasso, Random Forest)
7. Evaluate with MSE, RMSE, MAE, R²
8. Cross-validation + feature importance

## Results
- Best model: Ridge (MSE ≈ 619M, R² ≈ 0.92)
- Top features: size_sqft, location_score

## Usage
```bash
pip install pandas numpy scikit-learn matplotlib joblib
python3 house_price_model.py
python3 house_price_extended.py
```
