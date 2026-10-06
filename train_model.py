import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# OrchardGuard Random Forest Training
# Soil ADC 0-4000 -> Soil Percentage 0-100%
# ============================================================


print("=" * 60)
print("              OrchardGuard AI Training")
print("=" * 60)
print()


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "OrchardGuard-AI",
    "dataset",
    "orchard_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "orchardguard_model.pkl"
)


print("Base directory:")
print(BASE_DIR)
print()

print("Dataset:")
print(DATASET_PATH)
print()

print("Model output:")
print(MODEL_PATH)
print()


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("=" * 60)
    print("ERROR: Dataset not found!")
    print(DATASET_PATH)
    print("=" * 60)

    raise SystemExit(1)


# ============================================================
# 3. LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(DATASET_PATH)

except Exception as e:

    print("ERROR: Cannot read dataset")
    print(e)

    raise SystemExit(1)


print("Dataset loaded successfully")
print()

print("Rows:", len(df))
print("Columns:", list(df.columns))
print()


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "temperature",
    "humidity",
    "soil",
    "rain",
    "risk_level"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("ERROR: Missing columns:")
    print(missing_columns)

    raise SystemExit(1)


# ============================================================
# 5. CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()


# Convert numeric columns

numeric_columns = [
    "temperature",
    "humidity",
    "soil",
    "rain"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove invalid rows

df = df.dropna()


print("Rows after cleaning:", len(df))
print()


# ============================================================
# 6. CONVERT SOIL ADC -> SOIL %
# ============================================================
#
# User-defined project mapping:
#
# ADC 0     = 0%
# ADC 1000  = 25%
# ADC 2000  = 50%
# ADC 3000  = 75%
# ADC 4000  = 100%
#
# Formula:
# soil_percent = soil_adc / 4000 * 100
#

print("-" * 60)
print("Converting Soil ADC -> Soil Percentage")
print("-" * 60)

df["soil_adc"] = df["soil"]

df["soil"] = (
    df["soil_adc"] / 4000.0
) * 100.0

df["soil"] = df["soil"].clip(
    lower=0,
    upper=100
)


print("Soil conversion completed")
print()

print(
    df[
        ["soil_adc", "soil"]
    ].head(10)
)

print()


# ============================================================
# 7. FEATURES
# ============================================================

FEATURES = [
    "temperature",
    "humidity",
    "soil",
    "rain"
]

TARGET = "risk_level"


X = df[FEATURES]

y = df[TARGET]


# ============================================================
# 8. SHOW CLASS DISTRIBUTION
# ============================================================

print("=" * 60)
print("Risk Level Distribution")
print("=" * 60)

print(
    y.value_counts()
)

print()


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================
#
# 60% Training
# 40% Testing
#
# stratify=y keeps the class distribution similar
#

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.40,

    random_state=42,

    stratify=y
)


print("=" * 60)
print("Train / Test Split")
print("=" * 60)

print(
    "Training data:",
    len(X_train)
)

print(
    "Testing data :",
    len(X_test)
)

print(
    f"Train/Test ratio: {len(X_train) / len(X):.0%} / "
    f"{len(X_test) / len(X):.0%}"
)

print()


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("=" * 60)
print("Training Random Forest")
print("=" * 60)

model = RandomForestClassifier(

    n_estimators=600,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


print("Training completed")
print()


# ============================================================
# 11. TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("=" * 60)
print("MODEL RESULT")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

classes = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

print()

print(
    pd.DataFrame(
        cm,
        index=[
            f"Actual_{c}"
            for c in classes
        ],
        columns=[
            f"Predicted_{c}"
            for c in classes
        ]
    )
)

print()


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("=" * 60)
print("Feature Importance")
print("=" * 60)

feature_importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)


print(
    feature_importance
)

print()


# ============================================================
# 15. SAVE MODEL
# ============================================================

print("=" * 60)
print("Saving Model")
print("=" * 60)

joblib.dump(
    model,
    MODEL_PATH
)


print(
    "Model saved successfully:"
)

print(
    MODEL_PATH
)

print()


# ============================================================
# 16. VERIFY MODEL
# ============================================================

print("=" * 60)
print("Model Verification")
print("=" * 60)

loaded_model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully"
)

print(
    "Classes:",
    loaded_model.classes_
)

print(
    "Features:",
    FEATURES
)

print()


# ============================================================
# 17. TEST WITH SOIL %
# ============================================================

test_data = np.array([

    [
        30.0,    # temperature
        85.0,    # humidity
        75.0,    # soil %
        2.0      # rain
    ]

])


test_prediction = loaded_model.predict(
    test_data
)[0]


test_probability = loaded_model.predict_proba(
    test_data
)[0]


test_probability_result = {}

for class_name, probability in zip(
    loaded_model.classes_,
    test_probability
):

    test_probability_result[class_name] = round(
        float(probability) * 100,
        2
    )


print(
    "Test Sensor Data:"
)

print(
    "Temperature:",
    test_data[0][0]
)

print(
    "Humidity:",
    test_data[0][1]
)

print(
    "Soil:",
    test_data[0][2],
    "%"
)

print(
    "Rain:",
    test_data[0][3]
)

print()

print(
    "Prediction:",
    test_prediction
)

print(
    "Probability:",
    test_probability_result
)

print()


print("=" * 60)
print("          TRAINING COMPLETED")
print("=" * 60)