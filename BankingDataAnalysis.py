import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import seaborn as sns

data_train = pd.read_csv("bank-additional-full1.csv", na_values =['NA'])
#columns = data_train.columns.values[0].split(';')
#columns = [column.replace('"', '') for column in columns]
#data_train = data_train.values
#data_train = [items[0].split(',') for items in data_train]
#data_train = pd.DataFrame(data_train,columns = columns)

data_train['job'] = data_train['job'].str.replace('"', '')
data_train['marital'] = data_train['marital'].str.replace('"', '')
data_train['education'] = data_train['education'].str.replace('"', '')
data_train['default'] = data_train['default'].str.replace('"', '')
data_train['housing'] = data_train['housing'].str.replace('"', '')
data_train['loan'] = data_train['loan'].str.replace('"', '')
data_train['contact'] = data_train['contact'].str.replace('"', '')
data_train['month'] = data_train['month'].str.replace('"', '')
data_train['day_of_week'] = data_train['day_of_week'].str.replace('"', '')
data_train['poutcome'] = data_train['poutcome'].str.replace('"', '')
data_train['y'] = data_train['y'].str.replace('"', '')

print("Data train Heads :",data_train.head())

data_test = pd.read_csv("bank-additional.csv", na_values =['NA'])
#data_test = data_test.values
#data_test = [items[0].split(';') for items in data_test]
#data_test = pd.DataFrame(data_test,columns = columns)

data_test['job'] = data_test['job'].str.replace('"', '')
data_test['marital'] = data_test['marital'].str.replace('"', '')
data_test['education'] = data_test['education'].str.replace('"', '')
data_test['default'] = data_test['default'].str.replace('"', '')
data_test['housing'] = data_test['housing'].str.replace('"', '')
data_test['loan'] = data_test['loan'].str.replace('"', '')
data_test['contact'] = data_test['contact'].str.replace('"', '')
data_test['month'] = data_test['month'].str.replace('"', '')
data_test['day_of_week'] = data_test['day_of_week'].str.replace('"', '')
data_test['poutcome'] = data_test['poutcome'].str.replace('"', '')
data_test['y'] = data_test['y'].str.replace('"', '')

print("Data test Heads :",data_test.head())


def categorize(df):
    new_df = df.copy()
    le = preprocessing.LabelEncoder()

    new_df['job'] = le.fit_transform(new_df['job'])
    new_df['marital'] = le.fit_transform(new_df['marital'])
    new_df['education'] = le.fit_transform(new_df['education'])
    new_df['default'] = le.fit_transform(new_df['default'])
    new_df['housing'] = le.fit_transform(new_df['housing'])
    new_df['month'] = le.fit_transform(new_df['month'])
    new_df['loan'] = le.fit_transform(new_df['loan'])
    new_df['contact'] = le.fit_transform(new_df['contact'])
    new_df['day_of_week'] = le.fit_transform(new_df['day_of_week'])
    new_df['poutcome'] = le.fit_transform(new_df['poutcome'])
    new_df['y'] = le.fit_transform(new_df['y'])
    return new_df

data = pd.concat([data_train, data_test])
data.replace(['basic.6y', 'basic.4y', 'basic.9y'], 'basic', inplace=True)
print ("Missing Values:" ,data.isnull().sum())

sns.set(style="ticks", color_codes=True)
sns.countplot(y='job', data=data)
plt.show()

data = data[data.job != 'unknown']
sns.countplot(y='marital', data=data)
plt.show()

print(" Marital Data description:",data.marital.value_counts())

data = data[data.marital != 'unknown']
data = data[data.loan != 'unknown']
sns.countplot(y='education', data=data)
plt.show()

data = data[data.education != 'illiterate']

print("Data descrption :",data.describe())


sns.countplot(y='y', data=data)
plt.show()

data = categorize(data)
#data = data.convert_objects(convert_numeric=True)
for i in data:
    data[i] = pd.to_numeric(data[i] , errors='ignore')
sns.boxplot(x='y', y='duration', data=data)
plt.show()
sns.boxplot(x='y', y='education', data=data)
plt.show()

sns.boxplot(x='y', y='housing', data=data)
plt.show()
sns.boxplot(data['y'],data['age'])
plt.show()

sns.boxplot(data['y'],data['job'])
plt.show()
sns.boxplot(data['y'],data['campaign'])
plt.show()

def remove_outliers(df, column , minimum, maximum):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values < minimum, col_values>maximum), col_values.mean(), col_values)
    return df

min_val = data["duration"].min()
max_val = 1500
data = remove_outliers(df=data, column='duration' , minimum=min_val, maximum=max_val)

min_val = data["age"].min()
max_val = 80
data = remove_outliers(df=data, column='age' , minimum=min_val, maximum=max_val)

min_val = data["campaign"].min()
max_val = 6
data = remove_outliers(df=data, column='campaign' , minimum=min_val, maximum=max_val)

sns.countplot(x='education',hue='y',data=data)
plt.show()
sns.countplot(x='default',hue='y',data=data)
plt.show()
data = data.drop('default',axis=1)

sns.countplot(x='poutcome',hue='y',data=data)
plt.show()

sns.countplot(x='contact',hue='y',data=data)
plt.show()
# The Correltion matrix
corr = data.corr()
print("\n\t\tCorreltion matrix:\n",corr,"\n\n")

# Heatmap
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')
plt.show()

data = data.drop('contact',axis=1)

data = data.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1)

print("Data information :",data.info())

print("Data Head",data.head())


X = data.drop('y',axis = 1).values
y = data['y'].values

