# ✅ train_model.py — Final Enhanced Version
import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier

# -----------------------------------------
# 1️⃣ Paths setup
# -----------------------------------------
project_path = r"C:\Users\ASUS\Documents\smart_diabetes_predict"
models_dir = os.path.join(project_path, "models")
os.makedirs(models_dir, exist_ok=True)

print(f"📁 Models folder path: {models_dir}")

# -----------------------------------------
# 2️⃣ Load dataset
# -----------------------------------------
df = pd.read_csv(os.path.join(project_path, "dataset", "Healthcare-Diabetes.csv"))
print(f"✅ Dataset Loaded Successfully! Shape: {df.shape}")

if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)
    print("⚙️ Removed 'Id' column to fix feature mismatch issue.")

# -----------------------------------------
# 3️⃣ Split and Scale
# -----------------------------------------
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 4️⃣ Define Models + Params
# -----------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }),
    "SVM": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }),
    "Decision Tree": (DecisionTreeClassifier(), {
        "max_depth": [3, 5, 10, None],
        "criterion": ["gini", "entropy"]
    }),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None]
    }),
    "AdaBoost": (AdaBoostClassifier(), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5]
    }),
    "LightGBM": (LGBMClassifier(), {
        "num_leaves": [20, 31, 40],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200]
    })
}

# -----------------------------------------
# 5️⃣ Train + Tune Models
# -----------------------------------------
results = []
best_model = None
best_model_name = ""
best_accuracy = 0
best_params = {}

for name, (model, params) in models.items():
    print(f"\n🔍 Tuning {name}...")
    try:
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        preds = grid.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))

        print(f"✅ {name}: Accuracy = {acc:.4f}, Best Params = {grid.best_params_}")

        if acc > best_accuracy:
            best_model_name = name
            best_accuracy = acc
            best_model = grid.best_estimator_
            best_params = grid.best_params_
    except Exception as e:
        print(f"❌ {name} failed: {e}")

# -----------------------------------------
# 6️⃣ Save Results
# -----------------------------------------
comparison_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
comparison_df.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)

# Save model + scaler + tuning info
joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

tuning_info = {
    "best_model_name": best_model_name,
    "best_accuracy": float(best_accuracy),
    "best_params": best_params
}
with open(os.path.join(models_dir, "tuning_info.json"), "w") as f:
    json.dump(tuning_info, f, indent=4)

print(f"🏆 Best Model: {best_model_name} | Accuracy: {best_accuracy:.4f}")
print("🧠 Tuning Info saved to tuning_info.json")

# -----------------------------------------
# 7️⃣ Plot Comparison
# -----------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x='Accuracy', y='Model', data=comparison_df, palette='viridis')
plt.title('Model Comparison (Accuracy)')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "model_comparison.png"))
plt.show()
