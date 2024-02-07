# Diabetes Prediction Using Machine Learning

## Project Overview
This project focuses on predicting diabetes using a dataset from Kaggle. The analysis includes data preprocessing, exploratory data analysis (EDA), and building predictive models with AdaBoost and XGBoost classifiers. The objective is to accurately predict diabetes outcomes based on various health metrics.

## Dataset
The dataset used is "Diabetes Dataset" from Kaggle, which contains data on several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

- **Source**: [Predict Diabetes on Kaggle](https://www.kaggle.com/datasets/whenamancodes/predict-diabities)

## Requirements
To run this project, ensure you have Python installed and the following libraries:

- `pandas`
- `seaborn`
- `matplotlib`
- `sklearn`
- `xgboost`

## How to Use
1. **Clone the Repository**: Clone this GitHub repository to your local machine.
   
3. **Download the Dataset**: Download the dataset from the provided Kaggle link and save it to a folder named `data` in the project directory.

4. **Install Required Libraries**: Install the required Python libraries listed in the requirements section.

5. **Run the Notebook**: Navigate to the project directory and launch Jupyter Notebook or JupyterLab. Open the project notebook to run the analyses.

6. **Exploratory Data Analysis (EDA)**: The notebook includes detailed EDA, such as distribution of variables and counts of outcomes.

7. **Model Training**: Follow the steps in the notebook to train AdaBoost and XGBoost classifiers. The process includes data preprocessing, model training, prediction, and evaluation.

8. **Hyperparameter Tuning**: The project demonstrates how to perform hyperparameter tuning for both classifiers to improve model performance.

## Project Structure
- `data/`: Directory containing the diabetes dataset.
- `Diabetes_Prediction.ipynb`: Jupyter notebook with all the analyses and model training steps.

## Findings

### Exploratory Data Analysis (EDA)
- **Variable Distributions**: Our EDA revealed significant differences in distributions of key variables such as Glucose, BloodPressure, and BMI when comparing outcomes (diabetes presence vs. absence). Particularly, higher glucose levels were closely associated with diabetes outcomes.
- **Outcome Distribution**: The dataset contains 500 instances without diabetes and 268 with diabetes, indicating a reasonable balance but slight imbalance towards negative outcomes.

### Hyperparameter Tuning
- **AdaBoost Tuning**: After tuning, the best AdaBoost model achieved a 75% accuracy with 500 estimators and a learning rate of 0.1. This indicated a slight decrease in overall accuracy but helped in understanding the trade-offs between model complexity and performance.
- **XGBoost Tuning**: The tuned XGBoost model achieved an 81% accuracy, with optimal parameters including a learning rate of 0.01, 500 estimators, and a max depth of 4. This model represented my best performing model, balancing complexity with predictive power.

### Key Insights
- **Feature Importance**: Across models, Glucose level emerged as a highly predictive feature, underscoring its critical role in diabetes onset prediction. Other important features included BMI and Age, highlighting the multifactorial nature of diabetes risk.
- **Model Selection**: The XGBoost model, with its high accuracy and balanced precision-recall metrics, was identified as the most effective model for predicting diabetes outcomes in our dataset.
- **Data Quality**: The impact of zero values in Insulin and Skin Thickness variables was mitigated through predictive imputation, significantly improving model accuracy and reliability.

### Conclusions
My analysis underscores the potential of machine learning models in predicting diabetes outcomes from clinical measurements. The XGBoost classifier, with its fine-tuned parameters, offers a robust tool for early diabetes risk assessment. Future work could explore more complex feature engineering, the integration of additional clinical factors, and the deployment of these models in clinical decision support systems.


This section provides a comprehensive overview of your project's key findings, offering readers insight into the data's story, model performance, and the implications of your work. Tailor the content to reflect your project's specific results and conclusions.
