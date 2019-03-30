import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

df = pd.read_csv('train.csv')

labelEncoder = preprocessing.LabelEncoder()
df['Sex'] = labelEncoder.fit_transform(df['Sex'])
df['Cabin'] = labelEncoder.fit_transform(df['Cabin'].fillna('0'))
df['Embarked'] = labelEncoder.fit_transform(df['Embarked'].fillna('0'))

x_np = np.array(df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].fillna(0))
d = df[['Survived']].to_dict('record')
vectorizer = DictVectorizer(sparse=False)
y_np = vectorizer.fit_transform(d)

print(y_np)
