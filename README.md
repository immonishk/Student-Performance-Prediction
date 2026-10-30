# Student Performance Prediction

This project predicts a student's final score using academic and study-related factors. It was developed to demonstrate data preprocessing, machine learning, model evaluation, and an interactive prediction application.

## Run the Project

### Requirements

- Python 3.10 or later
- The project files and dataset included in this folder

### Installation

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

### Start the Application

**Option 1: Using the launcher**

Double-click `RUN_PROJECT.bat` on Windows.

**Option 2: Using the terminal**

```bash
python -m streamlit run RUN.py
```

The application will open in your web browser. On the first run, the system trains the models and saves evaluation results, trained models, and graphs in the `outputs/` folder. Training may take a little time.

Use the dashboard to enter student details, select a prediction model, estimate a final score, and compare model performance.

## Project Structure

```text
Student_Performance_Prediction/
├── data/
│   └── student_performance.csv
├── outputs/
├── RUN.py
├── RUN_PROJECT.bat
├── Sem3_MiniProject.ipynb
├── requirements.txt
└── README.md
```

- `RUN.py` — Main application, model training, evaluation, and dashboard.
- `data/student_performance.csv` — Dataset used to train and test the models.
- `outputs/` — Generated evaluation metrics, graphs, saved models, and run summary.
- `RUN_PROJECT.bat` — Windows launcher for starting the application.
- `Sem3_MiniProject.ipynb` — Jupyter Notebook for reviewing and running the analysis.
- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.

## Machine Learning Models

The project implements four regression models:

1. **Linear Regression** — Baseline regression model.
2. **Decision Tree Regressor** — Predicts scores using decision rules.
3. **Random Forest Regressor** — Combines multiple decision trees.
4. **Artificial Neural Network (ANN)** — Implemented using Scikit-learn's `MLPRegressor`.

The models are evaluated using:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of determination.

The evaluation results and comparison charts are saved in the `outputs/` folder.

## Dataset Description

The dataset contains 1,200 synthetic student records with six input features:

- Study hours per day
- Attendance percentage
- Previous examination score
- Assignment average
- Sleep hours per night
- Extracurricular hours per week

The target variable is `final_score`. The dataset is divided into training and testing subsets using an 80:20 split.

## Important Note

The dataset was generated for educational and experimental purposes. It does not contain actual student records. The predictions demonstrate machine learning functionality and should not be treated as reliable assessments of real students or used as the sole basis for academic decisions.

Author - Monish Kumar (25MDA10035)