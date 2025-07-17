# Employee_Salary_Prediction
This project develops a machine learning and deep learning model to **classify employee income** (i.e., whether they earn <=50K or >50K per year) based on various demographic and employment features. The best-performing model is then deployed as an interactive **Streamlit web application**, made publicly accessible via **Ngrok**.

The project follows a typical machine learning workflow, from data preprocessing and exploratory data analysis to model training, evaluation, selection, and deployment.

## 🌟 Features

* **Comprehensive Data Preprocessing:** Handles missing values, outliers, and converts categorical features into numerical formats.
* **Exploratory Data Analysis (EDA):** Visualizes data distributions, relationships between features, and insights into income classes.
* **Multiple Model Comparison:** Trains and evaluates various Machine Learning algorithms (e.g., K-Nearest Neighbors, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Support Vector Machine) and a Deep Learning model (Multi-Layer Perceptron).
* **Best Model Selection:** Identifies the most accurate and robust model based on key classification metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
* **Model Persistence:** Saves the trained model and preprocessing objects (scaler, encoders) for seamless deployment.
* **Interactive Streamlit Web App:**
    * **Single Prediction:** Allows users to input individual employee details and get an instant income classification.
    * **Batch Prediction:** Supports uploading a CSV file for predictions on multiple employee records, with results downloadable.
    * **Data Visualization:** Includes charts within the web app to visualize prediction distributions and insights.
* **Public Deployment:** Uses Ngrok to create a public URL for the Streamlit app, enabling easy sharing and access.

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (for various ML models, preprocessing, model selection)
* **Deep Learning:** `tensorflow`, `keras` (for Multi-Layer Perceptron)
* **Model Persistence:** `joblib`
* **Web Application Framework:** `streamlit`
* **Public Tunneling:** `pyngrok` (for Ngrok integration)
* **Development Environment:** Google Colab

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone [https://github.com/SaiPraneeth-E/Employee-Salary-Prediction-App.git](https://github.com/SaiPraneeth-E/Employee-Salary-Prediction-App.git) # Update with your actual repo name
cd Employee-Salary-Prediction-App
