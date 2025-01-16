# consumer-risk-models-python-machine-learning

![alt text](https://github.com/gaptab/consumer-risk-models-python-machine-learning/blob/main/default_class_distribution_visual.png)
![alt text](https://github.com/gaptab/consumer-risk-models-python-machine-learning/blob/main/feature_importances_visual.png)
![alt text](https://github.com/gaptab/consumer-risk-models-python-machine-learning/blob/main/roc_curve_visual.png)

The goal of this project is to validate consumer risk models, standardize validation workflows, set Model Risk Management (MRM) guidelines, identify validation gaps, and manage workflows efficiently. Additionally, data insights are visualized, and results are saved for further analysis.

Steps
1. Data Creation: 
Generated  data simulating consumer financial attributes like age, income, loan amount, and credit score, with a target variable Default.
This enables testing and validation of risk models in the absence of real-world data.
2. Model Training and Validation: 
A Random Forest Classifier was used to predict whether a consumer defaults on a loan.
Split data into training and test sets to evaluate the model.
Validated the model using key metrics like AUC-ROC and a classification report (precision, recall, F1-score).
To assesses the performance of the consumer risk model and identifies its strengths and weaknesses.
3. Standardization of Validation Workflows: 
Encapsulated the validation logic in reusable functions.
Functions compute metrics such as AUC-ROC and classification reports for any given model.
This Ensures consistency and reusability across multiple models during validation.
4. Model Risk Management (MRM) Guidelines: 
Created predefined guidelines for model development, validation, and monitoring.
Automated the generation of MRM guidelines in a white paper format.This establishes a standardized approach to manage risks associated with machine learning models.

458
