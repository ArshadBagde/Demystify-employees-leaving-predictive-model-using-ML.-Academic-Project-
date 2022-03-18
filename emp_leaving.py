import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

employee_df=pd.read_csv('/home/nimish/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

#binary categorical data into numerical
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes'else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
employee_df.hist(bins = 30, figsize= (15, 15),color='r')

#print(employee_df.describe())
employee_df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis = 1, inplace = True)
#print(employee_df.describe())

#Correlated features don’t improve model performance. It is wise to remove correlated features
correlations = employee_df.corr()
plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)

#Age Variable Correlation Analysisi
plt.figure(figsize = (15, 10), dpi = 300)
age = sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df )
age.set_xticklabels(age.get_xticklabels(),fontsize=7, rotation=40, ha="right")
#plt.tight_layout()

#Job Role 
sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)


#distance from home
#sns.kdeplot(left_df['DistanceFromHome'], label = 'Employee left', shade = True, color = 'r')
#sns.kdeplot(stay_df['DistanceFromHome'], label = 'Employee stay', shade = True, color = 'b')



plt.show()
