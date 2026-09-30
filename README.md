# Car Price Prediction with Machine Learning

## Objective
Build regression models to predict used-car selling prices from car-related features. The workflow covers preprocessing, feature engineering, exploratory analysis, model training, and evaluation.

## Files
- `car_price_prediction.py` — complete ML project
- `car data.csv` — place the dataset here
- `requirements.txt` — Python dependencies
- `car_price_prediction.ipynb` — notebook version
- `outputs/` — generated charts and model metrics

## Expected Dataset Columns
The common CodeAlpha used-car dataset contains:
- `Year`
- `Present_Price`
- `Driven_kms`
- `Fuel_Type`
- `Selling_type`
- `Transmission`
- `Owner`
- `Selling_Price`
- `Car_Name` (optional)

## Machine Learning Workflow
1. Load and inspect the dataset
2. Remove duplicates
3. Handle missing values
4. Create `Car_Age`
5. Encode categorical features
6. Split data into training and testing sets
7. Train Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor
8. Evaluate using MAE, RMSE, and R²
9. Compare models
10. Plot actual vs predicted prices

## How to Run

```bash
pip install -r requirements.txt
python car_price_prediction.py
```

For Jupyter:

```bash
jupyter notebook car_price_prediction.ipynb
```

## GitHub Repository Name
`CodeAlpha_Car_Price_Prediction`
