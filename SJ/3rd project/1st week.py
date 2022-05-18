# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:11:09 2022

@author: sjnam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/FIFA_train.csv')
test = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/FIFA_test.csv')
submission = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/submission.csv')

train.info()
train.isnull().sum()
train.describe()
train.describe(include='object')

nums = ['age', 'stat_overall', 'stat_potential'] # 연속형 (continuous)
noms = ['continent', 'contract_until' ,'position', 'prefer_foot', 'reputation', 'stat_skill_moves'] # 이산형 (nominal)
y = 'value'


for col in ['continent', 'position', 'prefer_foot', 'contract_until','reputation', 'stat_skill_moves']:
  print(col)
  print(train[col].value_counts())

f, axes = plt.subplots(3,4, figsize=(30,18))
axes = axes.flatten()
for col, ax in zip(train.columns, axes):
  sns.histplot(data = train, x=col, ax=ax)
plt.show()

#연속형 자료 시각화

f, axes = plt.subplots(1,len(nums), figsize=(24,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.histplot(data = train, x=col, ax=ax) # 연속형 자료는 histplot 사용
plt.show()

# 이산형 자료 시각화

f, axes = plt.subplots(1,len(noms), figsize=(28,5))
axes = axes.flatten()                         
for col, ax in zip(noms, axes):
  sns.countplot(data = train, x=col, ax=ax) # 이산형은 countplot 사용
plt.show()

# 연속형 자료 시각화 (boxplot)

f, axes = plt.subplots(1,len(nums), figsize=(24,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.boxplot(data = train, x=col, ax=ax)
plt.show()

sns.barplot(data = train, x='position', y=y) # 포지션과 몸값 사이의 관계
sns.barplot(data = train, x='prefer_foot', y=y) # 주로쓰는 발과 몸값 사이의 관계
sns.heatmap(data= train.corr(), cmap='coolwarm', annot=True, vmax=1, vmin=-1)


# 범주형 변수랑 예측값과의 상관관계

f, axes = plt.subplots(1,len(noms), figsize=(30,5))
axes = axes.flatten()                         
for col, ax in zip(noms, axes):
  sns.boxplot(data = train, x=col, y=y,ax=ax) # 박스플롯을 사용
plt.show()

# 연속형 변수랑 예측값과의 상관관계
f, axes = plt.subplots(1,len(nums), figsize=(20,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.scatterplot(data = train, x=col, y=y, ax=ax, hue = 'continent') # 대륙별로 색을 지정
plt.show()

# 대륙 따로따로 나타내기
for col, ax in zip(nums, axes):
  sns.lmplot(data = train, x=col, y=y, hue='continent', col='continent') # 대륙별로 위 그래프를 분리
  plt.show()
  
# 선호하는 발마다 나타내기
for col, ax in zip(nums, axes):
  sns.lmplot(data = train, x=col, y=y, hue='prefer_foot', col='prefer_foot') # 선호하는 발로 그래프를 분리
  plt.show()




