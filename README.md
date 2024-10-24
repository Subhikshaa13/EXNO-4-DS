# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/720408be-d1c6-41cb-80d4-c40eea578d19)

```
df.dropna()
```

![image](https://github.com/user-attachments/assets/b6c9eebe-cf97-4caf-8f8d-28a86d5d7609)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

```
![image](https://github.com/user-attachments/assets/5377bef6-496e-47d2-b072-079f1cb450e1)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/9ade912f-907d-47f9-8046-495b99357bad)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/248f1a29-192e-4d5a-8c4f-efa2c94d1f45)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/06c13dd9-7210-4c8d-894f-9c5104dcfe31)

```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1

```
![image](https://github.com/user-attachments/assets/f26b3a1d-26c8-4590-89b3-adc706737f17)

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```

![image](https://github.com/user-attachments/assets/4ec7b9b3-3e28-439f-b7b0-e95d08be38dd)

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```

![image](https://github.com/user-attachments/assets/6c99a1f5-995b-4e88-9ec5-de2ad527b33d)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/1f30d4ce-e912-40ee-95e8-ef433d815dc2)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/9c8cae9a-4886-4096-860f-fbbc675dfbef)

```
data2 = data.dropna(axis=0)
data2

```
![image](https://github.com/user-attachments/assets/ab91c906-7a7a-442b-afa9-0a5c1af5fe81)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
![image](https://github.com/user-attachments/assets/3fe9954b-1ffc-41b0-95ed-5f436a11dd66)

```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/037dc215-da12-4cb8-b132-d87df67d7fd5)

```
data2
```

![image](https://github.com/user-attachments/assets/6795ae3c-e371-4f1b-b2da-ce06ae337f65)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/6560e469-40bd-4466-8da4-045b800b1838)

```
columns_list=list(new_data.columns)
print(columns_list)

```
![image](https://github.com/user-attachments/assets/4caf37d9-cc34-405f-8c74-814b8e849b86)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)

```
![image](https://github.com/user-attachments/assets/963570a3-09b6-4395-9f81-2ace1e275f70)

 ```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/ab2aec5f-221c-4efa-b657-a1eed156d1e1)

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/a9bc56d8-7a1e-4e56-a503-dfaad3ea1719)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

```
![image](https://github.com/user-attachments/assets/324ddeea-6d29-46a4-bd83-23c9c9d1b488)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

```
0.8392087523483258
```

print('Misclassified samples: %d' % (test_y != prediction).sum())
```

Misclassified samples: 1455

```
data.shape

(31978, 13)
```

## FEATURE SELECTION TECHNIQUES

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
![image](https://github.com/user-attachments/assets/593b1748-fbed-4a04-8c4e-9b1c25df6a12)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```
![image](https://github.com/user-attachments/assets/0020945d-dc82-49ca-a036-5fd2cc8ae138)

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

```
![image](https://github.com/user-attachments/assets/3e871862-24c0-441b-92d5-e6e9212eab60)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```
![image](https://github.com/user-attachments/assets/62737b28-34c3-4df1-a251-3e55403ccc61)



## RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.
