"""
Car Price Prediction with Machine Learning

Input:
    car data.csv

The script supports the common used-car dataset containing:
Year, Present_Price, Driven_kms, Fuel_Type, Selling_type,
Transmission, Owner, Selling_Price, and optionally Car_Name.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("car data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Place the CodeAlpha car dataset in this folder."
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

required = ["Year", "Present_Price", "Driven_kms", "Fuel_Type",
            "Selling_type", "Transmission", "Owner", "Selling_Price"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Numeric conversion
for col in ["Year", "Present_Price", "Driven_kms", "Owner", "Selling_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

# Feature engineering
current_year = pd.Timestamp.today().year
df["Car_Age"] = current_year - df["Year"]
df.loc[df["Car_Age"] < 0, "Car_Age"] = 0

print("\nDATASET SHAPE:", df.shape)
print("\nFIRST FIVE ROWS:")
print(df.head())
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nSUMMARY STATISTICS:")
print(df.describe(include="all"))

# EDA plots
plt.figure(figsize=(8, 5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_selling_price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price")
plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_present_vs_selling_price.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price")
plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age (years)")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_age_vs_selling_price.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price by Fuel Type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_fuel_type_price.png", dpi=300)
plt.close()

# Features
target = "Selling_Price"
drop_cols = [target, "Car_Name", "Year"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []
predictions = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

print("\nMODEL RESULTS:")
print(results_df.to_string(index=False))

# Model comparison
plot_df = results_df.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(9, 5))
sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric")
plt.title("Model Error Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_model_comparison.png", dpi=300)
plt.close()

# Actual vs predicted for the model with highest R2
best_model_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_model_name]

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=best_pred)
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.title(f"Actual vs Predicted — {best_model_name}")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_actual_vs_predicted.png", dpi=300)
plt.close()

print(f"\nModel with highest R² on this run: {best_model_name}")
print("Task 3 prediction workflow completed. Check the 'outputs' folder.")
