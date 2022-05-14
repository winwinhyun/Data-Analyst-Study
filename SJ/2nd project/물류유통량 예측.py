# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:56:55 2022

@author: sjnam
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# 한글 폰트 설정
local_font_path = "c:/Windows/Fonts/malgun.ttf"
import matplotlib.font_manager as fm

try : 
    plt.rc('font', family=fm.FontProperties(fname=local_font_path).get_name()) # 윈도우에서 폰트 경로 설정
except : 
    plt.rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False

# 마이너스 부호 깨지는 현상 방지
plt.rc('axes', unicode_minus=False)


train = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/물류 유통량 예측 경진대회/train.csv')
test = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/물류 유통량 예측 경진대회/test.csv')
submission = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/물류 유통량 예측 경진대회/sample_submission.csv')

train.info() #결측치가 없음
test.info() #결측치가 없음

#물품 카테고리별로 운송장 건수가 어떻게 다른지 파악하기

train_df = train.groupby(['물품_카테고리'])['운송장_건수'].agg(['count','mean','median','std']).reset_index()

x = train_df['물품_카테고리'].values
y = train_df['mean'].values

plt.bar(x,y)
plt.show()

#평균을 기준으로 내림차순으로 다시 정리
train_df = train_df.sort_values(by = ['mean'], ascending = False)

x = train_df['물품_카테고리'].values
y = train_df['mean'].values

plt.bar(x,y)
plt.xticks(rotation = 60)
plt.show()

x = x[:20]
y = y[:20]

plt.bar(x,y)
plt.xticks(rotation = 60)
plt.show()






