import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("gld_price_data.csv")

# Drop Date column
df = df.drop("Date", axis=1)

# Features and Target
X = df.drop("GLD", axis=1)
y = df["GLD"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "gold_price_model.pkl")

print("Model saved successfully!")