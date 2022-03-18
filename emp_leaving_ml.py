import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import imblearn
from imblearn.over_sampling import SMOTE
#from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import confusion_matrix,mean_absolute_error,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout


employee_df=pd.read_csv('/home/nimish/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

employee_df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis = 1, inplace = True)

#binary categorical data into numerical
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes'else 0)

#Encode categorical variables i.e. get_dummies() 
bt = pd.get_dummies(employee_df['BusinessTravel'])
dpt = pd.get_dummies(employee_df['Department'])
edu = pd.get_dummies(employee_df['EducationField'])
gender = pd.get_dummies(employee_df['Gender'])
jrole = pd.get_dummies(employee_df['JobRole'])
mstatus = pd.get_dummies(employee_df['MaritalStatus'])

frames = [employee_df,bt,dpt,edu,gender,jrole,mstatus]
X_all = pd.concat(frames,axis=1)

X_all.drop(['BusinessTravel', 'Department', 'Gender', 'EducationField','JobRole','MaritalStatus'], axis = 1, inplace = True)

y = employee_df['Attrition']
#X_all.info()

X_all.drop(['Attrition'], axis = 1, inplace = True)


scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)


print(employee_df.shape)

print(employee_df.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Check imbalance of dataset
unique, count = np.unique(y_train, return_counts=True)
y_train_count = { k:v for (k,v) in zip(unique,count)}
print(y_train_count)

#SMOTE method to oversample the minority classes
oversampler = SMOTE(random_state=0)
X_smote_train, y_smote_train = oversampler.fit_resample(X_train,y_train)


#Logistic Regression
lr = LogisticRegression()
scoring = 'accuracy'
lr.fit(X_smote_train, y_smote_train)
y_pred = lr.predict(X_test)
result = model_selection.cross_val_score(lr,X_smote_train,y_smote_train, scoring=scoring)
print('LR Accuracy:%.3f' % result.mean())
cm_lr = confusion_matrix(y_pred, y_test)
sns.heatmap(cm_lr, annot= True)
mae = mean_absolute_error(y_test, y_pred)
print("Absolute Mean Error LR: %.2f" % mae)
print(cm_lr)


RF = RandomForestClassifier()
RF.fit(X_smote_train, y_smote_train)
rf_y_pred = RF.predict(X_test)
cm_rf = confusion_matrix(rf_y_pred, y_test)
sns.heatmap(cm_rf, annot= True)
print(cm_rf)
mae_rf = mean_absolute_error(y_test, y_pred)
print("Absolute Mean Error RF: %.2f" % mae_rf)
rf_result = accuracy_score(y_pred,y_test)
print('RF Accuracy:%.3f' % rf_result.mean())




model = Sequential()
model.add(Dense(units = 50, activation = 'relu', input_shape = (50, )))
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
epochs_hist = model.fit(X_smote_train, y_smote_train, epochs = 50, batch_size = 50)



y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot= True)
print(cm)
mae = mean_absolute_error(y_test, y_pred)
print("Absolute Mean Error ANN: %.2f" % mae)

ann_result = accuracy_score(y_pred,y_test)
print('ANN Accuracy:%.3f' % ann_result.mean())

plt.show()
