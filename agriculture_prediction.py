import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Dataset/datafile (1).csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
for col in df.columns:
    print(repr(col))

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================
# TARGET COLUMN
# ==========================

target_col = df.columns[-1]

print("\nTarget Column:", target_col)

# ==========================
# YIELD DISTRIBUTION
# ==========================

plt.figure(figsize=(8, 5))
sns.histplot(df[target_col], kde=True)
plt.title("Yield Distribution")
plt.savefig("yield_distribution.png")
plt.close()

# ==========================
# ENCODE CATEGORICAL COLUMNS
# ==========================

le_crop = LabelEncoder()
le_state = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["State"] = le_state.fit_transform(df["State"])

# ==========================
# CORRELATION HEATMAP
# ==========================

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODEL TRAINING
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=========================")
print("MODEL PERFORMANCE")
print("=========================")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# ==========================
# ACTUAL VS PREDICTED
# ==========================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

# ==========================
# FEATURE IMPORTANCE
# ==========================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# ==========================
# SAMPLE PREDICTION
# ==========================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Prediction:")
print("Predicted Yield:", prediction[0])

print("\nGraphs saved successfully:")
print("1. yield_distribution.png")
print("2. correlation_heatmap.png")
print("3. feature_importance.png")