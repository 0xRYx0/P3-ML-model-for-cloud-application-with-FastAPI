# Model Card

## Overview

This model is a classifier trained to predict whether an employer's income exceeds $50K/year. It was developed as part of the third project for Udacity's Machine Learning DevOps nanodegree in June 2023. The model utilizes the Random Forest algorithm.

## Intended Use

The model is intended to be used to determine what features impact the income of a person and identify underprivileged employers. It can provide insights into factors such as gender, race, and other features that may contribute to income disparities. However, it is important to note that the model may not be suitable for modern data analysis since the training data is relatively old.

## Training Data

The model was trained on the Census Income Dataset from the UCI Machine Learning Repository. The dataset consists of both categorical and numerical features. To preprocess the data, missing values in categorical features were imputed using the most frequent value, and the categories were encoded using `OrdinalEncoder`. Unknown categories were assigned a value of 1111. Numerical data was normalized using `StandardScaler`. The education column was dropped as it was already encoded in the education-num column.

## Evaluation Data

The evaluation data was obtained by splitting the training data using `train_test_split` from the `sklearn` library into `Training dataset [70%]` and `Testing dataset [30%]` . The split was stratified based on the salary label, with a fixed random_state=42.

|        | Train | Test  |
|--------|-------|-------|
| Percentage | 70% | 30% |
| X Shape  | (22775, 14) | (9762, 14)|
| Y Sahepe | (22775,) | (9762,)|


## Metrics

The model's performance was evaluated using `precision`, `recall`, and `F1` score. These metrics are suitable for imbalanced problems like this binary classification task. Precision represents the ratio of correct predictions to the total predictions, recall represents the ratio of correct predictions to the total number of correct items in the dataset, and F1 score is the harmonic mean between precision and recall, providing a balanced measure of both.

## Ethical Considerations

The data used in this model is open-sourced and available on the UCI Machine Learning Repository for educational purposes. However, it is important to note that the dataset was collected in 1996 and may not reflect insights from the modern world. When interpreting and using the model's results, it is essential to consider the limitations of the dataset's temporal relevance.

## Caveats and Recommendations

It is recommended to focus more on collecting additional data for features with minor categories to enhance the model's performance and coverage. As the training data is relatively old, it may not capture the dynamics of the modern world accurately. Therefore, caution should be exercised when applying the model to current scenarios.

## Quantitative Analyses

The following table shows the performance metrics calculated for class 1 (>50K) using sklearn metrics:

|        | Train | Test  |
|--------|-------|-------|
| Precision | 0.6817 | 0.6210 |
| Recall    | 0.9382 | 0.8359 |
| F1        | 0.7896 | 0.7126 |

These metrics indicate the model's precision, recall, and F1 score on both the training and test datasets.