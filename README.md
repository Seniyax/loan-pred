# Fraud Detection with Pyspark
This Project implements a scalable machine learning pipeline to detect fraudulent financial transactions using PySpark and Google Colab.This addresses the class imbalance common in financial data by employing Undersampling techniques to improve model sensitivity.
## 📊 Dataset
This project uses the PaySim Synthetic Financial Dataset, which stimulates mobile money transactions.
- **Size**- ~6.3 million rows
- **Target**: isFraud (Binary classification)
- **Key Challenges**: Extreme class imbalance (Fraudulent transactions represent < 0.2% of the total data).
## Tech Stack
- **Languages** : Python
- **Framework** : PySpark (Spark SQL & MLlib)
- **Visualization** : Matplotlib,Seaborn
## Pipeline Architecture
### 1. Exploratory Data Analysis (EDA)
Used Spark’s distributed processing to aggregate data before visualizing. Key insights included identifying that fraud occurs exclusively in TRANSFER and CASH_OUT transaction types.
### 2. Feature Engineering
- **Categorical Encoding**: Used StringIndexer and OneHotEncoder to transform transaction types.
- **Vectorization**: Used VectorAssembler to merge features into a single dense vector for Spark MLlib compatibility.
### 3. Handling Imbalance (Undersampling)
Implemented a random undersampling strategy on the majority class (Legitimate transactions) to create a balanced 50/50 training set. This forces the model to learn the specific characteristics of fraud.
### 4. Model Training & Evaluation
- **Algorithms**: Random Forest Classifier
- **Metrics**: Evaluated using Area Under ROC and Precision-Recall Curves
## 📈 Results
By applying undersampling, the model significantly improved its Recall, successfully identifying the majority of fraudulent cases that a baseline model (trained on imbalanced data) would have missed.
 
