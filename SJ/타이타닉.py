# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:09:41 2022

@author: sjnam
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression #로지스틱 회귀 모델 불러오기
from sklearn.tree import DecisionTreeClassifier # 의사결정 나무 모델 불러오기

train = pd.read_csv('/Users/sjnam/Desktop/데이콘 데이터/타이타닉/train.csv') #데이터 불러오기 (모델 학습 파일)
test = pd.read_csv('/Users/sjnam/Desktop/데이콘 데이터/타이타닉/test.csv') #모델 시험지 파일
submission = pd.read_csv('/Users/sjnam/Desktop/데이콘 데이터/타이타닉/submission.csv') #제출을 위한 답지

# 자료 분석 과정 = EDA
type(train)

head = train.head()
head1 = test.head()

shape = train.shape # train 데이터의 행과 열의 수

info = train.info() # 결측치 확인할떄 사용

describe = train.describe() # 데이터의 기술 통계량을 알아볼때 사용

train = train.drop(['Name', 'Ticket'], axis = 1)
test = test.drop(['Name', 'Ticket'], axis = 1)

train.pivot_table(values = 'Age', index = 'Pclass', aggfunc = 'mean' )
# 나이 결측치를 처리할때 pclass 기준으로 해도 될거 같다는 결론 도출

train_age_null = train[train.Age.isnull()]
# 결측치만 따로 빼서 저장


train_firstclass = train_age_null[train_age_null['Pclass'] == 1]
train_secondclass = train_age_null[train_age_null['Pclass'] == 2]
train_thirdclass = train_age_null[train_age_null['Pclass'] == 3]

train_firstclass = train_firstclass.fillna(value = '38')
train_secondclass = train_secondclass.fillna(value ='30')
train_thirdclass = train_thirdclass.fillna(value ='25')

train_drop_na = train.dropna(subset = ['Age'])
# 나이가 결측치인거 제거하고 저장

train_concat = pd.concat([train_drop_na, train_firstclass, train_secondclass, train_thirdclass])
# 4가지 데이터 프레임 병합

train = train_concat
train.info()
# 나이가 아직 문자형이기 때문에 정수형으로 바꿔줘야함
train = train.astype({'Age':'int'})
# 이제 테스트 데이터도 똑같이 해준다.
test_age_null = test[test.Age.isnull()]

test_firstclass = test_age_null[test_age_null['Pclass'] == 1]
test_secondclass = test_age_null[test_age_null['Pclass'] == 2]
test_thirdclass = test_age_null[test_age_null['Pclass'] == 3]

test_firstclass = test_firstclass.fillna(value = '38')
test_secondclass = test_secondclass.fillna(value ='30')
test_thirdclass = test_thirdclass.fillna(value ='25')
# test셋은 평균을 구해보면 train셋이랑 다르게 나오는데 이거는 그냥 무시하는건지?

test_drop_na = test.dropna(subset = ['Age'])
test_concat = pd.concat([test_drop_na, test_firstclass, test_secondclass, test_thirdclass])
test = test_concat
test = test.astype({'Age':'int'})

train.info()

pclass_train_dummies = pd.get_dummies(train['Pclass'])
pclass_test_dummies = pd.get_dummies(test['Pclass'])

train.drop(['Pclass'], axis = 1, inplace = True)
test.drop(['Pclass'], axis = 1, inplace = True)

pclass_train_dummies.columns = ['First', 'Second', 'Third']
pclass_test_dummies.columns = ['First', 'Second', 'Third']




value_counts = train['Embarked'].value_counts() #각 시리즈마다 고유의 값 개수를 세줌

groupby = train.groupby('Sex').mean() #성별로 묶어서 각각의 평균값 산출

groupby1 = train.groupby('Pclass').mean() #좌석 등급별로 묶어서 각각의 평균값 산출

#전처리 과정
groupby1 = train.groupby('Pclass').mean()['Survived'].plot(kind = 'bar', rot =0) 
# 좌석 등급별로 묶어서 각각의 평균값 산출 후 생존 변수 대상으로 막대 그래프 그린다

train['Age'].plot(kind = 'hist', bins = 30, grid = True) # 나이 구간별로 히스토그램을 그린다

train.plot(x = 'Age', y = 'Fare', kind = 'scatter') # 나이와 요금을 산점도로 나타냄

train.isna().sum() # 결측치 확인하기

train['Age'].median()
train['Age'] = train['Age'].fillna(28) # 중앙값인 28로 대체

train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S') # 최빈값인 S로 대체

train['Sex'] = train['Sex'].map({'male' :0, 'female' :1 }) # 문자형 변수인 성별을 수치형으로 변환

# 카빈은 왜 결측치 대체를 안해줄까?

# 모델링
X_train = train[['Sex', 'Pclass']]
y_train = train['Survived']

test['Sex'] = test['Sex'].map({'male' :0, 'female' :1 })
X_test = test[['Sex', 'Pclass']]

lr = LogisticRegression()   # 로지스틱과 의사결정나무 사용
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train) # fit을 이용해서 모델을 학습시킨다.
dt.fit(X_train, y_train)

lr.predict(X_test) # 학습한 모델을 test데이터를 이용해 모델 예측
lr.predict_proba(X_test)  # 예측값을 0,1이 아닌 확률로 표현 => 1열이 사망확률 2열이 생존 확률 
lr_pred = lr.predict_proba(X_test)[:,1]

dt_pred = dt.predict_proba(X_test)[:,1]

submission['Survived'] = lr_pred

submission.to_csv('logistic_regression_pred.csv', index = False)

submission['Survived'] = dt_pred
submission.to_csv('decision_tree_pred.csv', index = False)












