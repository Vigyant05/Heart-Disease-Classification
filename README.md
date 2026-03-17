# HeartGuard AI - Multi-Model Heart Disease Classification System

HeartGuard AI is a comprehensive, full-stack diagnostic application designed to predict and classify five distinct cardiovascular conditions using advanced machine learning models. Built with a high-performance **FastAPI** backend and an interactive **Next.js** frontend, this system leverages ensemble learning and gradient boosting techniques to achieve clinical-grade accuracy.

## 🚀 Project Overview

The system analyzes patient clinical data (such as age, chest pain type, cholesterol, max heart rate, etc.) to provide real-time risk assessments. It features a modern web interface with **interactive chat diagnostics** and **batch file processing** capabilities.

The project demonstrates:
- **Full-Stack Development:** Integration of a React/Next.js frontend with a Python/FastAPI backend.
- **Advanced ML Pipelines:** Usage of SMOTE/ADASYN for class balancing, RandomizedSearchCV for hyperparameter tuning, and ensemble methods.
- **High-Performance Models:** Achieving >90% accuracy on critical health indicators.

## 🩺 Diagnostic Models & Performance

Our models have been rigorously trained and evaluated on specific medical datasets. We use **mAP (Mean Average Precision)** as a primary metric to ensure we correctly rank positive cases, alongside **accuracy** and **AUC**.

### 💓 **Normal Heart** – *Initial Cardiac Screening*
*   **Goal:** Distinguish healthy patients from those with potential cardiac issues.
*   **Algorithm:** AdaBoost + Random Forest
*   **Performance:** ~88.33% Accuracy

### ⚠️ **Heart Attack** – *Myocardial Infarction Detection*
*   **Goal:** Detection of immediate heart attack detection.
*   **Algorithm:** LightGBM / XGBoost
*   **Performance:**
    *   **Accuracy:** 96.70%
    *   **mAP:** 99.69%
    *   **AUC:** 0.9933

### 📉 **Heart Failure** – *Heart Failure Prediction*
*   **Goal:** Prediction of heart failure incidents.
*   **Algorithm:** CatBoost
*   **Performance:**
    *   **Accuracy:** 91.67%
    *   **mAP:** 94.45%
    *   **AUC:** 0.9588

### 🩸 **Hypertension** – *Hypertension Risk Assessment*
*   **Goal:** Assessment of high blood pressure risks.
*   **Algorithm:** Balanced Random Forest
*   **Performance:**
    *   **Accuracy:** 92.22%
    *   **mAP:** 97.74%
    *   **AUC:** 0.9717

### 🫀 **Coronary** – *Coronary Artery Disease Detection*
*   **Goal:** Identification of coronary artery disease indicators.
*   **Algorithm:** Optimized XGBoost
*   **Performance:** High Precision Optimized

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Frontend:** Next.js (React), TypeScript, Tailwind CSS
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn
- **Data Processing:** Pandas, NumPy, Joblib
- **AI Tools:** GitHub Copilot (Code generation, debugging, and optimization)
- **Deployment:** Vercel (Frontend), Railway/Render (Backend compatible)

## ⚡ Quick Start Guide

### 1. Backend Setup

```bash
# Navigate to project root
cd /Users/vigyantnayak/heart_disease_classification

# Activate virtual environment
source .venv/bin/activate

# Run the API server
uvicorn main:app --reload
```
*The backend API will be available at [http://localhost:8000](http://localhost:8000)*

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The web interface will be running at [http://localhost:3000](http://localhost:3000)*

## 📂 Project Structure

- `main.py`: Main entry point for the FastAPI application.
- `final_models/`: Directory containing the best-performing pre-trained models (.pkl).
- `Code_rahul/` & `Code_vigyant/`: Analysis notebooks and training scripts.
- `frontend/`: Next.js web application source code.
- `Dataset/`: Source CSV data for training and testing.
