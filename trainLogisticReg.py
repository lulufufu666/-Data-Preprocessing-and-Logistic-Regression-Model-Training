import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the data
filename = 'D:/train.csv'
data = pd.read_csv(filename)

# Extract features and labels
header = data.columns.tolist()
header = header[1:len(header)-1:1]  # Get column names of features

X = data.drop('target', axis=1)
y = data['target']

# Data preprocessing
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the logistic regression model
logreg = LogisticRegression(C=100)
logreg.fit(X_train, y_train)

# Model evaluation
print(f"C value: {logreg.C}")
accuracy = round(logreg.score(X_train, y_train), 4)
print("Training set accuracy:", accuracy)
accuracy = round(logreg.score(X_test, y_test), 4)
print("Test set accuracy:", accuracy)
print("Test set AUC value: {:.3f}".format(roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])))
print("Classification report:\n", classification_report(y_test, logreg.predict(X_test), target_names=["Non-5G", "5G"]))

# Visualize feature coefficients
coefs = logreg.coef_
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(coefs[0]))), coefs[0])
plt.xticks(list(range(len(coefs[0]))), header, rotation=90)
plt.xlabel("Feature Index")
plt.ylabel("Feature Coefficient Value")
plt.title(f"Feature Coefficients when 'C' is {logreg.C}")
plt.show()
