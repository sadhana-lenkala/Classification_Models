# KERNEL SVM
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
Dataset = pd.read_csv('Social_Network_Ads.csv')
X = Dataset.iloc[:,2:-1].values
Y = Dataset.iloc[:,-1].values

#split into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)

#perform feature scaling
from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)

#perform SVM on train set
from sklearn.svm import SVC
SVM = SVC(kernel = 'rbf', random_state = 0) #gaussian kernel
SVM.fit(X_train,Y_train)

#predict values using SVM for test set
y_pred = SVM.predict(X_test)

#confusion Matrix
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(Y_test,y_pred)

#plot train values along with SVM predicted values
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVM.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#plot test values along with predicted test values
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVM.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

