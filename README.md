<H3>ENTER YOUR NAME : SABEEHA SHAIK</H3>
<H3>ENTER YOUR REGISTER NO: 212223230176</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
### IMPORT LIBRARIES
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### READ DATASET
```
df=pd.read_csv("Churn_Modelling.csv")
```

### CHECK DATA
```
df.head()
df.tail()
df.columns
```

### CHECK MISSING DATA
```
df.isnull().sum()
```

### ASSIGNING X
```
X = df.iloc[:,:-1].values
X
```

### ASSIGNING Y
```
Y = df.iloc[:,-1].values
Y
```

### CHECK FOR OUTLIERS
```
df.describe()
```

### DROP STRING VALUES FROM DATASET
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

### CHECKING DATASET AFTER DROPPING STRING VALUES
```
data.head()
```

### NORMALISE DATASET 
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### SPLIT THE DATASET
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```

### TRAINING AND TESTING MODEL
```
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```



## OUTPUT:
### DATA CHECKING
<img width="747" height="102" alt="image" src="https://github.com/user-attachments/assets/16478408-8465-4eb1-9621-220d6ee0ff0a" />


### MISSING DATA
<img width="292" height="302" alt="image" src="https://github.com/user-attachments/assets/f8c6fd8b-7f67-4115-9a51-c0ed86425100" />

### DUPLICATE VALUES
<img width="753" height="162" alt="image" src="https://github.com/user-attachments/assets/52c0c6a6-2ead-47e2-8539-b98ea62a77af" />

### Y VALUES
<img width="480" height="38" alt="image" src="https://github.com/user-attachments/assets/0e8255b6-422c-4b58-8716-1c9cdc430c1e" />

### OUTLIERS
<img width="812" height="281" alt="image" src="https://github.com/user-attachments/assets/bc04a29f-be53-4dff-98bf-e740ad0cc1c5" />

### CHECKING DATASET AFTER DROPPING STRING VALUES
<img width="821" height="227" alt="image" src="https://github.com/user-attachments/assets/37ec9e0e-40b9-4ca3-b577-4ad792c61a94" />

### NORMALISING DATASET
<img width="728" height="526" alt="image" src="https://github.com/user-attachments/assets/afba4a9a-816b-4a0d-9e26-7ebe19e96335" />

### SPLIT THE DATASET
<img width="446" height="168" alt="image" src="https://github.com/user-attachments/assets/527df098-d16c-4536-b16a-0a912981fbd1" />

### TRAIN AND TEST MODEL
<img width="540" height="451" alt="image" src="https://github.com/user-attachments/assets/615f3514-817c-4096-bc07-825d61f8e78f" />





## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


