import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


kuch  = pd.read_csv("Salary_dataset.csv")


kuch.head()

kuch = kuch[["YearsExperience", "Salary"]]

kuch.head()

x = kuch["YearsExperience"]
y = kuch["Salary"]

reg = LinearRegression()

plt.scatter(x,y)
plt.title("scatter plot")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape
y_train.shape


x_train = np.array(x_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

x_train.shape
y_train.shape


reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

plt.scatter(x,y)
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title("Linear Regression prediction")
plt.xlabel('Year Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)