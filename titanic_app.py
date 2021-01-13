import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
import altair as alt
from PIL import Image
import os
import pickle 

st.write("""
# Simple Titanic Survival Prediction App

The Titanic is sinking qustion is will you make it out alive ?
***
""")

image = Image.open('img/titanic2.jpg')

st.image(image, use_column_width=True)

st.write("""
*The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis*

*Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.*  

For more information [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set).

You will now get to see which features get which flower(Versicolor,Setosa,Virginica) with the sliders in the sidebar 
""")

st.sidebar.header('User Input Parameters')
st.sidebar.markdown('Play around with the features and see which flower you get')
def user_input_features():
    passenger_class = st.sidebar.selectbox('Island',('First Class','Second Class','Economy Class'))
    sex = st.sidebar.selectbox('Sex',('Male','Female'))
    age = st.sidebar.slider('Age', 0,80,29)
    SibSp = st.sidebar.slider('Number of Adults WIth You', 0,10,0)
    Parch = st.sidebar.slider('Number of Children', 0,10,0)
    Embarked = st.sidebar.selectbox('Port of Embarkation',('Cherbourg','Queenstown','Southampton'))

    if 'First' in passenger_class:
        passenger_class = 1
    elif 'Second' in passenger_class:
        passenger_class = 2
    else:
        passenger_class = 3

    data = {'Pclass': passenger_class,
                'Sex': sex.lower(),
                'Age': age,
                'SibSp': SibSp,
                'Parch': Parch,
                'Embarked':Embarked[0]}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# train = pd.read_csv("csv/train.csv")
# train.drop('Cabin',inplace=True,axis=1)

# def ageChange(cols):
#     if pd.isnull(cols[0]):
#         qwe = train[train.Pclass == cols[1]]
#         mean = qwe['Age'].mean() 
#         return mean
#     else:
#         return cols[0]

# train['Age'] = train[['Age','Pclass']].apply(ageChange,axis=1)
# embarked = pd.get_dummies(train['Embarked'],drop_first=True)
# sex = pd.get_dummies(train['Sex'],drop_first=True)

# train.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
# train = pd.concat([train,sex,embarked],axis=1)
# st.subheader('EDA(Exploratory Data Analysis)')

# for i in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
#     f, ax = plt.subplots(figsize=(7, 5))
#     # ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     ax = sns.swarmplot(y=i, x="Species", data=data)
#     st.pyplot(f)

# f, ax = plt.subplots(figsize=(7, 5))
# ax =plt.scatter(data['PetalLengthCm'], data['PetalWidthCm'], c=data['Species_Target'])
# plt.xlabel('Sepal Length', fontsize=18)
# plt.ylabel('Sepal Width', fontsize=18)
# plt.legend()
# st.pyplot(f)

# corrmat = data.corr()
# f, ax = plt.subplots(figsize=(7, 5))
# ax = sns.heatmap(corrmat, annot = True, vmax=1, square=True)
# st.pyplot(f)

# X = data.drop(columns=['Id','Species_Target','Species'])
# Y = data.Species_Target
# st.write(X)
# clf = RandomForestClassifier()
# clf.fit(X, Y)
load_clf = pickle.load(open('titanic_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(data.Species)

st.subheader('Prediction')
doa = np.array(['Dead','Alive',])
st.write(doa[prediction])
if doa[prediction] == 'Dead':
    st.write("I guess you will be following Jack at the bottom of the Sea. Say Hi to Spongebob")
else:
    st.write("You probably survived telling school kids about the time you almost died . Guessing you dont want drink with ice in them now ")

# st.write(iris_species[prediction][0])

image = Image.open(f'img/{doa[prediction][0]}.jpg')

st.image(image, use_column_width=True)


st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write('There is  ' + str(int(prediction_proba[0][0]*100)) + "% chance you would have died")
st.write('There are  ' + str(int(prediction_proba[0][1]*100)) + '% chance you would have lived')
# st.write('There are  ' + str(int(prediction_proba[0][2]*100)) + '% chance the flower is Virginica')




