# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:52:11 2020

@author: Vinicius
"""
# Import necessary libraries #
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import  train and test data #
train = pd.read_csv("D:/Unknown folder/Machine Enginner/titanic/train.csv")
test = pd.read_csv("D:/Unknown folder/Machine Enginner/titanic/test.csv")

# Verifying types of variables #
print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))

# Percentage of missing data #
display(train.dtypes)
display(test.dtypes)

display(train.head())
display(test.head())


train.describe()
test.describe()

# Percentage of missing data #
(train.isnull().sum() / train.shape[0]).sort_values(ascending=False)

# Histogram of each variable #
train.hist(figsize=(10,8))

# Probability of surviving by sex
train[['Sex','Survived']].groupby(['Sex']).mean()

# Graphic of quantity #
sns.countplot(x='Sex', data=train)
sns.countplot(x='Pclass', data=train)
sns.countplot(x='Embarked', data=train)

# Probability of surviving by age #
age_survived = sns.FacetGrid(train, col='Survived')
age_survived.map(sns.distplot, 'Age')


train.describe(include=['O'])

# Recovery of data frame #
train_idx = train.shape[0]
test_idx = test.shape[0]

# Extract column 'survived' and delete it of dataset train #
target = train.Survived.copy()
train.drop(['Survived'], axis=1, inplace=True)

# Concatenate train and test data #
df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

# Deleting not necessary variables #
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Completing missing data #
df_merged.isnull().sum()

# For Age #
age_mean = df_merged['Age'].mean()
df_merged['Age'].fillna(age_mean, inplace=True)

# For Fare #
fare_top = df_merged['Fare'].value_counts()[0]
df_merged['Fare'].fillna(fare_top, inplace=True)

# For Embarked #
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)

# Converting variable 'Sex' #
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})

# Converting variable 'Embarked' #
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)

display(df_merged.head())

# Recovering train and test data sets #
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]

# Importing machine learning library - Logistic regression #
from sklearn.linear_model import LogisticRegression

# Creating model #
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train, target)

# Printing model accuracy #
acc_logReg = round(lr_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(acc_logReg))

# Importing machine learning library - Decision tree #
from sklearn.tree import DecisionTreeClassifier

# Creating model #
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(train, target)

# Printing model accuracy #
acc_tree = round(tree_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acc_tree))


print("Would my girlfriend and I survive the titanic? ")

# Mine and my girlfriend's variables values #
vinicius_oliveira = np.array([3,0,22,1,0,15.5,0,0,0,1]).reshape((1,-1))
giovanna_lyssa = np.array([3,1,20,1,0,15.5,0,0,0,1]).reshape((1,-1))

# Printing results #
print("Vinícius:\t{}".format(tree_model.predict(vinicius_oliveira)[0]))
print("Giovanna:\t{}".format(tree_model.predict(giovanna_lyssa)[0]))
# By the tree decision model, we can conclude that my girlfriend would survive titanic, instead me
#SADDY! :/

