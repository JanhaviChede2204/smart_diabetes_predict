# ✅ train_model.py — Final Fixed Version
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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

print(f"🔍 Current Working Directory: {os.getcwd()}")
print(f"📁 Models folder path: {models_dir}")

# -----------------------------------------
# 2️⃣ Load dataset
# -----------------------------------------
df = pd.read_csv(os.path.join(project_path, "dataset", "Healthcare-Diabetes.csv"))
print(f"✅ Dataset Loaded Successfully! Shape: {df.shape}")

# Remove 'Id' column if it exists
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)
    print("⚙️ Removed 'Id' column to fix feature mismatch issue.")

# -----------------------------------------
# 3️⃣ Split and Scale
# -----------------------------------------
X = df.drop(columns=['Outcome'])
y = df['Outcome']
print(f"✅ Training features: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 4️⃣ Train multiple models
# -----------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "LightGBM": LGBMClassifier()
}

results = []

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        print(f"{name}: ✅ Accuracy = {acc:.4f}")
    except Exception as e:
        print(f"❌ {name} failed: {e}")

# -----------------------------------------
# 5️⃣ Compare and save best model
# -----------------------------------------
comparison_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
comparison_df.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)
print("📊 Model comparison CSV saved successfully!")

best_model_name, best_accuracy = max(results, key=lambda x: x[1])
best_model = models[best_model_name]

# Save model and scaler (force overwrite)
best_model_path = os.path.join(models_dir, "best_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

joblib.dump(best_model, best_model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ Best model '{best_model_name}' saved at: {best_model_path}")
print(f"✅ Scaler saved at: {scaler_path}")
print(f"🏆 Best Accuracy: {best_accuracy:.4f}")

# -----------------------------------------
# 6️⃣ Optional: Visualization
# -----------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(x='Accuracy', y='Model', data=comparison_df, palette='viridis')
plt.title('Model Comparison (Accuracy)')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "model_comparison.png"))
plt.show()
