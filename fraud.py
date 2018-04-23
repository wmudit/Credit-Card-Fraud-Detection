#Importing Libraries
import numpy as np
import pandas as pd

#Importin the dataset
dataset = pd.read_csv('creditcard.csv')
dataset.head()
dataset.isnull().sum()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset['Amount'] = sc.fit_transform(dataset['Amount'].reshape(-1, 1))
dataset = dataset.drop(['Time'],axis=1)

#Splitting into X and Y
X = dataset.iloc[:, 0:29].values
y = dataset.iloc[:, 29:].values

# Number of data points in the minority class
number_records_fraud = len(dataset[dataset.Class == 1])
fraud_indices = np.array(dataset[dataset.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = dataset[dataset.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

#Undersampled dataset
undersampled_data = dataset.iloc[under_sample_indices,:]

#Splitting into Undersampled X and Y
X_undersampled = undersampled_data.iloc[:, 0:29].values
y_undersampled = undersampled_data.iloc[:, 29:].values

#Splitting data into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_undersampled, y_undersampled, test_size = 0.2, random_state = 0)

#Using Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Predict Output
gauss_pred = gaussian.predict(X_test)

#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, Y_train)
#Predict output
regression_pred = reg.predict(X_test)

#Using K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
k_near = KNeighborsClassifier()
k_near.fit(X_train, Y_train)
#Predict output
k_near_pred = k_near.predict(X_test)

#Using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, Y_train)
#Predict output
dec_tree_pred = dec_tree.predict(X_test)

# Fitting SVC to the dataset
from sklearn.svm import SVC
regressor = SVC()
regressor.fit(X_train, Y_train)
# Predict output
svc_pred = regressor.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, dec_tree_pred)
