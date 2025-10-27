import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# 1. Set Up MLflow

mlflow.set_experiment("Telco Churn Prediction - Tuning")
print("MLflow diatur untuk logging LOKAL.")

# 2. Load Data

DATA_FILE = 'telco_churn_preprocessing.csv'
print(f"Memuat data dari {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# 3. Preprocessing (Data Splitting)
print("Memisahkan data (X dan y)...")
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model with Hyperparameter Tuning

with mlflow.start_run():
    
    print("Memulai training dengan GridSearchCV...")
    
    lr = LogisticRegression(max_iter=1000)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    # 5. Manual Logging
    
    print("Logging parameter dan metrik secara manual...")
    
    mlflow.log_param("best_C", best_model.C)
    mlflow.log_param("best_solver", best_model.solver)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall", recall) 
    
    mlflow.sklearn.log_model(best_model, "model")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    print("Training dan logging selesai.")
    print(f"Akurasi Model: {accuracy:.4f}")

print("\nSelesai")
print("Cek MLflow UI LOKAL di: http://127.0.0.1:5000")