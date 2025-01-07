"""IMPORT LIBRARIES"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

"""IMPORT DATASETS"""

dataset = pd.read_csv('Data.csv')
'Feature dataset'
'-1 = last value'
'[ROW,COLUMN]'
X = dataset.iloc[:, :-1].values
'Dependable dataset'
Y = dataset.iloc[:,-1].values


"""TAKING CARE OF MISSING DATA"""
'GIVE AN AVERAGE IN THE MISSING DATA'

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
'accessing all missing numerical cells'
imputer.fit(X[:,1:3])
'transform missing cell'
X[:,1:3] = imputer.transform(X[:,1:3])

"""ENCODING CATEGORICAL DATA"""
'CHANGE ALPHABET VARIABLES TO NEUMARICAL VARIABLES'
'FOR SEVERAL COLUMN MORE THAN TWO'

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])] , remainder='passthrough')
X = np.array(ct.fit_transform(X))

'FOR TWO COLUMNS ONLY'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


"""SPLITTING THE DATASET WITH TEST AND TRAINING"""

'80% training 20% test'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=1)


"""FEATURE SCALING"""
'NOT HAVE TO USE ALL MACHINE LEARNING PROJECTS'
'ONLY FOR NEUMARICAL VALUES'

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

