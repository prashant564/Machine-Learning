import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape((10,1)))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf', verbose=True,epsilon=0.1)
regressor.fit(X,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('SVR')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(y)-1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('SVR')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()
