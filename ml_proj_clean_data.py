# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:47:23 2023

@author: asmit
"""

import pandas as pd
#from sklearn.impute import KNNImputer

import warnings
warnings.simplefilter("ignore")

#load dataset
df=pd.read_csv("C:/Users/asmit/Downloads/marketing_trn_data (1).csv")
trn_class=pd.read_csv("C:/Users/asmit/Downloads/marketing_trn_class_labels.csv",header=None)
def basic_info(df):
    print("This dataset has ", df.shape[1], " columns and ", df.shape[0], " rows.")
    print("This dataset has ", df[df.duplicated()].shape[0], " duplicated rows.")
    print(" ")
    print("Descriptive Statistics of dataset:")
    print(" ")
    print(df.describe())
    print("Few rows of the dataset: ")
    print(" ")
    print(df.head())
    print(" ")
    print("Information about this dataset: ")
    print(" ")
    print(df.info())
print("\n### Before pre-processing the data ###\n")
basic_info(df)
 
df_copy=df.copy()
df_copy.rename(columns={' Income ':'Income'},inplace=True)

df_copy.Income = df_copy.Income.str.strip('$')
df_copy.Income = df_copy.Income.str.replace(".", "")
df_copy.Income = df_copy.Income.str.replace(",", "")
df_copy.Income = df_copy.Income.str.replace("00 ", "")
#df_copy.Income = df_copy.Income.astype('float64')

#imputer=KNNImputer()
#df_copy['Income']=imputer.fit_transform(df_copy[['Income']])   
have_income = df_copy[df_copy.Income.isnull()==False]
missing_income = df_copy[df_copy.Income.isnull()==True]
have_income.Income = have_income.Income.apply(int)
missing_income.Income = str(have_income.Income.median())
missing_income.Income = missing_income.Income.str.replace(".0", "")
missing_income.Income = missing_income.Income.apply(int)

df_copy[df_copy.Income.isnull()==False]=have_income
df_copy[df_copy.Income.isnull()==True]=missing_income
df_copy.Income=df_copy.Income.apply(int)
df_copy.Dt_Customer = pd.to_datetime(df_copy.Dt_Customer)
df_copy['D_Customer']=df_copy['Dt_Customer'].dt.day
df_copy['M_Customer']=df_copy['Dt_Customer'].dt.month
df_copy['Y_Customer']=df_copy['Dt_Customer'].dt.year
df_copy=pd.get_dummies(df_copy,columns=['Education','Marital_Status'])
df_copy.pop("Dt_Customer")

#df_copy.info()
c_labels=trn_class.iloc[:,1]
df_copy['target']=c_labels
print("\n Information about new dataset \n")
basic_info(df_copy)

df_copy.reset_index(drop=True)
df_copy.to_csv("C:/Users/asmit/Downloads/marketing_clean_df.csv", index=False)

