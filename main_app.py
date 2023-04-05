
import streamlit as st
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np

st.title("Clustering App Test by iris")

st.write(""" # My First app # """)



#df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
#st.write(df)

#textbox



#load data in in putbox
try:
    url_input = st.text_input('url :',)
    data = pd.read_csv(url_input)
    st.write(data)
    
    y = st.text_input('pred y :',)

    from sklearn.model_selection import train_test_split

    # create X and y
    X = data.drop(y, axis=1)
    y = data[y]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # display train and test sets
    #st.subheader("Train Set")
    #st.write(X_train)
    #st.write(y_train)

    #st.subheader("Test Set")
    #st.write(X_test)
    #st.write(y_test)

# create a KMeans clustering model with n_clusters=3
    kmeans = KMeans(n_clusters=3)

# fit the model on the training data
    kmeans.fit(X_train)

# make predictions on the test data
    y_pred = kmeans.predict(X_test)

# display the predicted clusters for the test data
    st.subheader("Predicted Clusters for Test Data")
    
    result_concat = pd.concat([y_test.reset_index(),pd.Series(y_pred, name='pred')], axis='columns')
    st.write(result_concat)

    from sklearn import model_selection
# knn - k-nearest neighbours
    from sklearn.neighbors import KNeighborsClassifier
    model_knn = KNeighborsClassifier()

    train_model_knn = model_knn.fit(X_train, y_train)

# print to get performance
    st.write("KNN Accuracy: ",model_knn.score(X_test, y_test) * 100)

# decision tree
    from sklearn.tree import DecisionTreeClassifier
    model_tree = DecisionTreeClassifier()
    model_tree.fit(X_train, y_train)
# print to get performance
    st.write("Decision Tree Accuracy: ",model_tree.score(X_test, y_test) * 100)

   
except FileNotFoundError:
    st.write("Please enter your data !")
except NameError:
    st.write("HTTP error Please try again!")
except KeyError:
    st.write("Please enter you y for predict !")

   

