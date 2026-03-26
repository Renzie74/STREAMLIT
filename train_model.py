import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

from sklearn.ensemble import AdaBoostClassifier


# =========================
# 1. LOAD DATA
# =========================
# Replace with your actual file path
df = pd.read_csv("venv\data\claims_data.csv")


# =========================
# 2. BASIC CLEANING
# =========================
target_col = "Claims_Status"

drop_if_exists = ["Accident_Cause", "Policy_ID"]
df = df.drop(columns=[c for c in drop_if_exists if c in df.columns], errors="ignore")

# Optional leakage columns if they exist
leakage_cols = [
    "Claims_Frequency",
    "Previous_Claims_Count",
    "Total_Claim_Amount_KES",
    "No_Claim_Bonus_%",
    "Customer_Age_Band"
]

X = df.drop(columns=[target_col] + [c for c in leakage_cols if c in df.columns], errors="ignore")
y = df[target_col]


# =========================
# 3. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. COLUMN TYPES
# =========================
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()


# =========================
# 5. PREPROCESSING
# =========================
numeric_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)


# =========================
# 6. FINAL PIPELINE
# =========================
best_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("sampler", ADASYN(random_state=42, n_neighbors=5)),
    ("model", AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ))
])


# =========================
# 7. TRAIN
# =========================
best_pipeline.fit(X_train, y_train)


# =========================
# 8. EVALUATE
# =========================
y_pred = best_pipeline.predict(X_test)

print("Deployable pipeline test performance:")
print(classification_report(y_test, y_pred))


# =========================
# 9. SAVE MODEL
# =========================
model_path = "claim_model.pkl"
joblib.dump(best_pipeline, model_path)

print(f"Model saved successfully to: {os.path.abspath(model_path)}")