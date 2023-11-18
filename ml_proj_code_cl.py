#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue November 7, 2023, 17:53:29

@author: asmit
"""


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,VarianceThreshold,SelectFromModel,RFE,SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


class classification():
     def __init__(self,path="C:/Users/asmit/Downloads/marketing_clean_df.csv",clf_opt='lr',no_of_selected_features=None):
        self.path = path
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of classifiers  
     def classification_pipeline(self):    
    # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='poly', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='lbfgs',class_weight='balanced',penalty="l2",max_iter=200) 
            #be3 = DecisionTreeClassifier(max_depth=10)
            be4 = GaussianNB(var_smoothing=0.5)
            be5 = MultinomialNB(alpha=0.005)
            #be6 = RandomForestClassifier(criterion='entropy',n_estimators=80)
            be7 = svm.LinearSVC(class_weight='balanced')
            be8 = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf_parameters = {
            'clf__base_estimator':(be1,be2,be4,be5,be7,be8),
            'clf__random_state':(0,5,10,20,50)
            }      
    # Decision Tree
        elif self.clf_opt=='dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
            'clf__criterion':('gini', 'entropy','log_loss'), 
            'clf__max_features':('auto', 'sqrt', 'log2'),
            'clf__max_depth':(10,60),
            'clf__ccp_alpha':(0.09,0.05,0.01),
            } 
    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(class_weight='balanced',penalty="l2") 
            clf_parameters = {
            'clf__solver':('lbfgs','liblinear'),
            'clf__random_state':(30,50),
            'clf__max_iter':(20,60),
            } 
    # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.05,1,100),
            }         
    #Gaussian Naive Bayes
        elif self.clf_opt=='gnb':
            print('\n\t### Training Gaussian Naive Bayes Classifier ###\n')
            clf=GaussianNB()
            clf_parameters={
                'clf__var_smoothing':(0.5,0.2,2,3,1)
                }
    # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0.005,1),
            }            
    
    # K Neighbours Classifier
        elif self.clf_opt=='kn':
            print('\n\t### Training K Neighbours Classifier ###\n')
            clf = KNeighborsClassifier()
            clf_parameters={
                "clf__n_neighbors":(5,3,10,6),
                "clf__algorithm":('ball_tree','kd_tree','brute'),
                }
    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy','gini'),       
            'clf__n_estimators':(80,100),
            'clf__max_depth':(100,200),
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(0.1,100),
            'clf__kernel':('linear','rbf','poly'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters    
 
# Statistics of individual classes
     def get_class_statistics(self,labels):
        class_statistics=Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))
       
# Load the data 
     def get_data(self,filename):
        reader=pd.read_csv(self.path+filename)  
        data=reader.iloc[:,:]
        labels=reader['target']

        self.get_class_statistics(labels)          

        return data, labels
    
    
     def classification(self):  
   # Get the data
        data,labels=self.get_data('marketing_clean_df.csv')
        data=np.asarray(data)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=5)
        predicted_class_labels=[]; actual_class_labels=[]; 
        count=0;
        for train_index, test_index in skf.split(data,labels):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count+=1                
            print('Training Phase '+str(count))
            clf,clf_parameters=self.classification_pipeline()
            pipeline = Pipeline([
                        #('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),
                        #('feature_selection', VarianceThreshold()),
                        #('feature_selection', SelectFromModel(estimator=clf, max_features=self.no_of_selected_features)),
                        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)), 
                        #('feature_selection',SequentialFeatureSelector(estimator=clf,n_features_to_select=self.no_of_selected_features,direction='backward')),
                        ('clf', clf),])
            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1',cv=10)          
            grid.fit(X_train,y_train)     
            clf= grid.best_estimator_ 
            #sel_fea=grid.best_estimator_.named_steps['feature_selection'].get_support()
            
            print('\n\n The best set of parameters of the pipiline are: ')
            print(clf)
            #print("\n Selected features:",sel_fea)
            predicted=clf.predict(X_test)
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)
            class_names=list(Counter(labels).keys())
            class_names = [str(x) for x in class_names]
            print("\n\t### Test Report on",count,"fold ###\n")
            print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names),'\n') 
 
       
    # Evaluation
        class_names=list(Counter(labels).keys())
        class_names = [str(x) for x in class_names] 
        print('\n\n The classes are: ')
        print(class_names)      
       
        print('\n ##### Classification Report on Training Data ##### \n')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        
        
        ac=accuracy_score(actual_class_labels, predicted_class_labels)
        print('\n Accuracy:\t'+str(ac)) 
        
        pr=precision_score(actual_class_labels, predicted_class_labels) 
        print ('\n Precision:\t'+str(pr)) 
        
        rl=recall_score(actual_class_labels, predicted_class_labels) 
        print ('\n Recall:\t'+str(rl))
        
        fm=f1_score(actual_class_labels, predicted_class_labels) 
        print ('\n F1-Score:\t'+str(fm))
        
        print('\n Confusion Matrix \n')
        print(confusion_matrix(actual_class_labels, predicted_class_labels))
        
        # Experiments on Given Random Validation Data during Test Phase
        print('\n ***** Classifying Validation  Data ***** \n')   
        predicted_cat=[]
        data,tst_cat=self.get_data('marketing_clean_df.csv')
        training_data, validation_data, training_cat, validation_cat = train_test_split(data,actual_class_labels,test_size=0.5, random_state=40,stratify=actual_class_labels)
        tst_data=validation_data
        predicted_cat=clf.predict(tst_data)
        print('\n ##### Classification Report on test data ##### \n')
        print(classification_report(validation_cat, predicted_cat, target_names=class_names)) 
    