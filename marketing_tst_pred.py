# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:13:07 2023

@author: asmit
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import warnings
warnings.simplefilter("ignore")

#load dataset
df=pd.read_csv("C:/Users/asmit/Downloads/marketing_tst_data.csv")
df1=pd.read_csv("C:/Users/asmit/Downloads/marketing_clean_df.csv")
                



be5 = MultinomialNB(alpha=0.005) 
clf = AdaBoostClassifier(base_estimator=be5,algorithm='SAMME',n_estimators=100,random_state=20)
pipe=Pipeline([('scaler',None),('feat_sel',SelectKBest(mutual_info_classif,k=4)),('model',clf)])




def basic_info(df):
    print("This dataset has ", df.shape[1], " columns and ", df.shape[0], " rows.")
    print("This dataset has ", df[df.duplicated()].shape[0], " duplicated rows.")
    print(" ")
    print("Descriptive statistics of the numeric features in the dataset: ")
    print(" ")
    print(df.head())
    print(" ")
    print("Information about this dataset: ")
    print(" ")
    print(df.info())
print("\n###  Test data before pre-processing  ###\n")
basic_info(df)

df_copy_tst=df.copy()
df_copy_tst.rename(columns={' Income ':'Income'},inplace=True)

df_copy_tst.Income = df_copy_tst.Income.str.strip('$')
df_copy_tst.Income = df_copy_tst.Income.str.replace(".0", "")
df_copy_tst.Income = df_copy_tst.Income.str.replace(",", "")
df_copy_tst.Income = df_copy_tst.Income.str.replace("00 ", "")
#df_copy_tst.Income = df_copy_tst.Income.astype('float64')

have_income = df_copy_tst[df_copy_tst.Income.isnull()==False]
missing_income = df_copy_tst[df_copy_tst.Income.isnull()==True]
have_income.Income = have_income.Income.apply(int)
missing_income.Income = str(have_income.Income.median())
missing_income.Income = missing_income.Income.str.replace(".0", "")
missing_income.Income = missing_income.Income.apply(int)

df_copy_tst[df_copy_tst.Income.isnull()==False]=have_income
df_copy_tst[df_copy_tst.Income.isnull()==True]=missing_income
df_copy_tst.Income=df_copy_tst.Income.apply(int)

#imputer=KNNImputer()
#df_copy_tst['Income']=imputer.fit_transform(df_copy_tst[['Income']])

df_copy_tst.Dt_Customer = pd.to_datetime(df_copy_tst.Dt_Customer)
df_copy_tst['D_Customer']=df_copy_tst['Dt_Customer'].dt.day
df_copy_tst['M_Customer']=df_copy_tst['Dt_Customer'].dt.month
df_copy_tst['Y_Customer']=df_copy_tst['Dt_Customer'].dt.year
df_copy_tst=pd.get_dummies(df_copy_tst,columns=['Education'])
df_copy_tst['Marital_Status_Absurd']=False
df_copy_tst=pd.get_dummies(df_copy_tst,columns=['Marital_Status'])
df_copy_tst.pop("Dt_Customer")
df_copy_tst['Marital_Status_YOLO']=False

print("\n Information about modified test dataset \n")
basic_info(df_copy_tst)

labels=df1.target
df1.pop("target")
pipe.fit(df1,labels)

tst_labels=pipe.predict(df_copy_tst)
print("\n")
print(tst_labels)
c0=0   
c1=0
with open("C:/Users/asmit/Downloads/tst_labels",'w') as f:
    for i in tst_labels:
        f.write(str(i)+"\n")

for i in tst_labels:
    if i==1:
        c1+=1 
    else:
        c0+=1
print("class 0:",c0,"\nclass 1:",c1)

