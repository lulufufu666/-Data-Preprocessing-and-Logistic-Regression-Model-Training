import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


filename = 'D:\\train.csv'
datas = pd.read_csv(filename)
header = datas.columns.tolist()  # 获取列名，即头信息
header = header[1:len(header)-1:1]

X = datas.drop('target', axis=1)
y = datas['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(C=100)

logreg.fit(X_train, y_train)

print(f"C in value: {logreg.C} >>>")
accuracy = round(logreg.score(X_train, y_train), 4)
print("Training Setting Accuracy: ", accuracy)
accuracy = round(logreg.score(X_test, y_test), 4)
print("Testing Setting Accuracy: ", accuracy)
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, logreg.predict(X_test))))
print("classification report:\n", classification_report(y_test, logreg.predict(X_test),
                                                        target_names=["非5g", "5g"]))

# 获取模型学习到的系数
coefs = logreg.coef_

# 可视化系数
plt.figure(figsize=(10, 6))  # 设置图表宽度为12，高度为6
plt.bar(list(range(len(coefs[0]))), coefs[0])
plt.xticks(list(range(58)), header, rotation=90)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient value")
plt.title(f"Coefficients of features when 'C' is {logreg.C}")
plt.show()

