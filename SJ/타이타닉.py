# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:09:41 2022

@author: sjnam
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression #로지스틱 회귀 모델 불러오기
from sklearn.tree import DecisionTreeClassifier # 의사결정 나무 모델 불러오기
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

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

groupby = train.groupby('Sex').mean() #성별로 묶어서 각각의 평균값 산출

groupby1 = train.groupby('Pclass').mean() #좌석 등급별로 묶어서 각각의 평균값 산출

groupby1 = train.groupby('Pclass').mean()['Survived'].plot(kind = 'bar', rot =0) 
# 좌석 등급별로 묶어서 각각의 평균값 산출 후 생존 변수 대상으로 막대 그래프 그린다


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

train = train.join(pclass_train_dummies)
test = test.join(pclass_test_dummies)

sex_train_dummies = pd.get_dummies(train['Sex'])
sex_test_dummies = pd.get_dummies(test['Sex'])

sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']

train.drop(['Sex'], axis = 1, inplace = True)
test.drop(['Sex'], axis = 1, inplace = True)

train = train.join(sex_train_dummies)
test = test.join(sex_test_dummies)

embarked_train_dummies = pd.get_dummies(train['Embarked'])
embarked_test_dummies = pd.get_dummies(test['Embarked'])

embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']

train.drop(['Embarked'], axis = 1, inplace = True)
test.drop(['Embarked'], axis = 1, inplace = True)

train = train.join(embarked_train_dummies)
test = test.join(embarked_test_dummies)

train.info()

train = train.drop(['Cabin', 'PassengerId'], axis = 1)
test = test.drop(['Cabin'], axis = 1 )

train.info()
print('-'*80)
test.info()

test['Fare'].fillna(0,inplace = True)

X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']
X_test = test.drop('PassengerId', axis = 1).copy()
X_test.info()

# --------------------------------------------------------------- 
# 모델 적용 

random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
# 랜덤 포레스트 이용해서 했을때 약 0.98 정도 나옴

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, Y_train)
# 서포트 벡터 머신을 이용할때 0.68 정도 나옴

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)
# knn 모델을 이용하면 0.83정도 나옴





















