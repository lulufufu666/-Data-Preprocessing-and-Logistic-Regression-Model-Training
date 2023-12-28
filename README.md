# Data Preprocessing and Logistic Regression Model Training

This project demonstrates how to preprocess data and train a logistic regression model using the pandas, matplotlib, and scikit-learn libraries.

## Installation and Dependencies

Please ensure that the following libraries are installed:

- pandas
- matplotlib
- scikit-learn

You can install the dependencies using the following command:

```
pip install pandas matplotlib scikit-learn
```

## Usage

1. Prepare the dataset:
    - Place the dataset file (train.csv) in the same directory as the code or specify the correct file path (filename).
2. Execute the code:
    - Run the code file.
3. View the results:
    - The output will include the following:
        - Accuracy scores for the training and test sets.
        - AUC score for the test set.
        - Visualization of the feature coefficients.

## Parameter Description

You can customize the following parameters in the code:

- test_size: The proportion of the test set.
- random_state: The random seed used to generate a stable split of the training and test sets.
- strategy: The filling strategy for SimpleImputer.
- C: The regularization parameter for the logistic regression model.
