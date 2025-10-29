<h1 align="center">🩺 Smart Diabetes Predictor</h1>

<p align="center">
  <b>An intelligent web app that predicts diabetes risk using Machine Learning and Streamlit 🚀</b><br>
  <i>Designed & developed by <a href="https://github.com/JanhaviChede2204">Janhavi Chede</a></i>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/ML-ScikitLearn%20%7C%20LightGBM-orange" alt="ML">
  <img src="https://img.shields.io/badge/Database-SQLite-blueviolet" alt="SQLite">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

---

## 🌟 Overview
**Smart Diabetes Predictor** is a data-driven web app that predicts the likelihood of diabetes based on medical parameters such as glucose, BMI, insulin levels, and more.  
It uses multiple machine learning models to ensure high accuracy and provides an easy-to-use Streamlit interface.

---

## ✨ Features
✅ Predicts diabetes probability using multiple ML models  
✅ Stores patient records in a SQLite database  
✅ View or delete past prediction records  
✅ Built-in Exploratory Data Analysis (EDA) notebook  
✅ Clean and interactive Streamlit GUI  

---

## ⚙️ Tech Stack

| Layer | Technologies |
|--------|--------------|
| **Frontend / GUI** | Streamlit |
| **Backend** | Python |
| **Database** | SQLite |
| **Machine Learning** | Scikit-learn, LightGBM |
| **Visualization** | Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |

---

## 🧠 ML Models Used
| Model | Description |
|--------|--------------|
| Logistic Regression | Baseline classifier |
| Naive Bayes | Probabilistic model |
| K-Nearest Neighbors (KNN) | Distance-based classifier |
| Support Vector Machine (SVM) | Margin-based model |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble of decision trees |
| AdaBoost | Boosted weak learners |
| LightGBM | Gradient boosting algorithm |

---

## 📂 Project Structure
smart_diabetes_predict/
│
├── dataset/ # Dataset files
├── models/ # Trained ML models
├── notebooks/ # EDA notebook
│ └── eda.ipynb
├── app.py # Main Streamlit app
├── train_model.py # ML model training and comparison
├── create_tables.sql # SQLite database creation
├── requirements.txt # Dependencies list
└── README.md # Documentation



---

## ⚡ How It Works
1️⃣ User enters health information (Glucose, BMI, Age, etc.)  
2️⃣ Data is scaled & fed into the trained ML model  
3️⃣ The model predicts **Positive** / **Negative** diabetes risk  
4️⃣ Result is displayed and stored in SQLite database  
5️⃣ User can view or delete past records anytime  

---

## 🧩 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/JanhaviChede2204/smart-diabetes-predict.git

# Go inside the folder
cd smart-diabetes-predict

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # for Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


📊 Dataset Information

The app uses the PIMA Indians Diabetes Database, which contains diagnostic health measurements and diabetes outcomes for female patients above age 21.

Columns include:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0 = Negative, 1 = Positive)

📈 EDA Highlights

The included notebook eda.ipynb explores:

Data distribution and correlations

Outlier detection and handling

Model accuracy comparison across algorithms

Feature importance visualization

💡 Future Scope

✨ Add user authentication (login/signup)
✨ Deploy model using Flask/FastAPI backend
✨ Integrate Plotly for dynamic data visualization
✨ Build REST API for mobile app integration

👩‍💻 Author

Janhavi Chede
🎓 B.Tech Artificial Intelligence & Machine Learning
📧 janhavichede220@gmail.com

🌐 GitHub Profile

<p align="center"> Made with ❤️ using <b>Python</b> & <b>Streamlit</b> </p>