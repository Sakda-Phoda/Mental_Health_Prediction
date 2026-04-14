# 🧠 Mental Health & Behavioral Risk Analysis

---

## 📌 Project Overview
This project focuses on identifying the key lifestyle and behavioral factors that contribute to mental health conditions, specifically **Anxiety, Depression, and Burnout**. By leveraging **LightGBM** and **Optuna** for hyperparameter optimization, we built a robust predictive pipeline to analyze which factors—such as stress levels, screen time, and sleep—are the most significant predictors of an individual's mental well-being.

---

## 📊 Summary of Key Findings

**1. The Impact of Stress & Screens** 
- **Stress Level** is the top predictor for all three conditions, especially for **Burnout and Anxiety**.
- **Daily Screen Time** and **Social Media Usage** are the top factors that lead to **Anxiety**.

**2. Work-Life Balance & Burnout** 
- For **Burnout**, the most important factors are **Sleep Hours** and **Work/Study Hours**.
- This shows a direct link between a lack of rest and feeling exhausted at work or school.

**3. Predictive Quality** 
- The models are effective at finding "Yes" cases. For example, the **Anxiety model** correctly identified 163 out of 253 individuals at risk.
- **ROC Curves** prove that the models are successfully learning patterns to tell the difference between different mental states.

---

## 🛠 Tech Stack & Methodology

- **Language:** Python (Pandas, NumPy)
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** LightGBM, Scikit-learn, Optuna
- **Data Pipeline:** - `OneHotEncoder`: For categorical features like Gender and Occupation.
    - `OrdinalEncoder`: For ranked physical activity data.
    - `PowerTransformer (Yeo-Johnson)` & `StandardScaler`: Used to handle data skewness and feature scaling.
    - `Optuna`: Utilized for automated hyperparameter tuning to maximize the **weighted F1-score**.

---

## 📚 Data Source
**Mental Health and Behavioral Risk Dataset** (2000 records)  
Available on Kaggle: [abrerjawod/mental-health-and-behavioral-risk-dataset](https://www.kaggle.com/datasets/abrerjawod/mental-health-and-behavioral-risk-dataset)