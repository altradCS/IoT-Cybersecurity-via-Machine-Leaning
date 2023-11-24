from IoTFunctions import *
#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV, train_test_split,cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Perceptron



####################################
 
# Add a title and intro text
st.title('Malware Explore')
st.text('WebApp for Cybersecurity PenTest')

# Sidebar setup
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Upload Dataset')

# Check if file has been uploaded
if upload_file is not None:
    dataset = pd.read_csv(upload_file)

####################################################

#Sidebar navigation
#st.sidebar.title('Navigation')

option = st.sidebar.radio('**Explore Data**',['Home', 'Data Summary',\
                                              'Sample Count', 'Data Header'])


st.sidebar.header("Dataset Feature Ranker")
if st.sidebar.button("Rank", use_container_width=True):
    target_column = 'Label'
    num_features = 20  # Number of top features to select
    feature_weights = weightRanker(dataset, target_column, num_features)
    feature_weights.sort_values(by="Score",ascending=True)
    st.write(feature_weights)

option1 = st.sidebar.radio('**Plot Data**',['Plot','Plot Sample'])


option2 = st.sidebar.radio('**Train ML Model**',\
                           ['ML Model','Decision Tree','NBs','Linear Reg',\
                            'Perceptron','Quadratic Discriminant Analysis',\
                            'Adaboost','Random Forest'])

if option == 'Home':
    home(upload_file)
elif option == 'Data Summary':
    data_summary()
    
elif option == 'Sample Count':
     st.write("### Number of Each Sample ")
     labels=dataset.Label
     plotData= dataset.value_counts(labels)
     st.write(plotData)
     
     
     
elif option == 'Data Header':
    data_header()


if option1 == 'Plot':
    st.write("Choose & Plot Data Relationship")
    
elif option1 == 'Plot Sample':
    plotFunc(dataset)

    
if option2 == 'ML Model':
    st.write("Choose then build & train ML Model for cybersecurity applications")
    
elif option2 == 'Decision Tree':
    algo_name = "Decision Tree"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, DTCModel)
    
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")
   
elif option2 == 'KNeighbors Classifier':
    algo_name = "KNeighbors Classifier"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, KNC_Model)
    
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"***Accuracy***: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")

elif option2 == 'NBs':
    algo_name = "NaiveBayes"
    accuracy, precision, recall, f1, start_time, end_time= \
              evaluate_algorithm(algo_name, NBModel)
    
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"**Accuracy**: {accuracy:.4f}, '\n' , Precision: {precision:.4f}")
    st.write(f"**Precision**: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")

elif option2 == 'Quadratic Discriminant Analysis':
    algo_name = "Quadratic Discriminant Analysis"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, modelQDN)
    
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")


elif option2 == 'Linear Reg':
    algo_name = "Linear Regression"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, linearRegModel)

    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")

elif option2 == 'Perceptron':
    algo_name = "Perceptron Model"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, percepModel)
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")

elif option2 == 'Adaboost':
    algo_name = "adaboost"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, adaboost)
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")

elif option2 == 'Random Forest':
    algo_name = "Random Forest"
    accuracy, precision, recall, f1, start_time, end_time=\
              evaluate_algorithm(algo_name, randomForest)
    st.write(f"File Name:  Algorithm: {algo_name}")
    st.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
    st.write(f"Time: {end_time - start_time:.4f} seconds")


