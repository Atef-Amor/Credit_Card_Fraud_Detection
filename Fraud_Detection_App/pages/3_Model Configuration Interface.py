import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection import ShuffleSplit, learning_curve

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report,make_scorer
from sklearn.metrics import roc_curve, roc_auc_score


from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import pickle


import warnings
warnings.filterwarnings('ignore')

from config import set_page_config

set_page_config()

def main():

    # Charger les données

    train = pd.read_csv('fraudTrain.csv')
    test = pd.read_csv('fraudTest.csv')

    def load_data():
        df = pd.concat([train,test],axis = 0)
        return df
    
    df = load_data()

#Data preprocessing

    #Supprimer la colonne Unamed: 0
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # renommer les colonnes
    new_headers = {
    'trans_date_trans_time': 'transaction_time',
    'cc_num': 'account_number',
    'merchant': 'merchant_name',
    'category': 'category',
    'amt': 'transaction_amount',
    'first': 'first_name',
    'last': 'last_name',
    'gender': 'gender',
    'street': 'street',
    'city': 'city',
    'state': 'state',
    'zip': 'zip',
    'lat': 'client_latitude',
    'long': 'client_longitude',
    'city_pop': 'city_population',
    'job': 'job',
    'dob': 'birthday',
    'trans_num': 'transaction_number',
    'unix_time': 'unix_time',
    'merch_lat': 'merchant_latitude',
    'merch_long': 'merchant_longitude',
    'is_fraud': 'is_fraud',
    }
    df.rename(columns=new_headers, inplace=True)

    #transformer le type de transaction_time et birthday au datetime

    df.transaction_time = pd.to_datetime(df.transaction_time)
    df.birthday = pd.to_datetime(df.birthday)

    #Ajouter la colonne age

    def calculate_age(born, trans_date):
        age = trans_date.year - born.year - ((trans_date.month, trans_date.day) < (born.month, born.day))
        return age

    df['age'] = df.apply(lambda row: calculate_age(row['birthday'], row['transaction_time']), axis=1)

    #Ajouter les colonnes dérivés de la colonne transaction_time
    df['transaction_hour'] = df['transaction_time'].dt.hour
    df['transaction_day'] = df['transaction_time'].dt.day
    df['transaction_month'] = df['transaction_time'].dt.month

    # ajouter la colonne transaction distance
    df['transaction_distance'] = np.sqrt((df.merchant_longitude - df.client_longitude)**2 + (df.merchant_latitude - df.client_latitude)**2)

    #supprimer les colonnes unitilisables
    columns_to_drop = ['account_number', 'birthday','merchant_latitude','merchant_longitude','client_latitude','client_longitude','first_name','last_name','transaction_number','unix_time','street','merchant_name']
    df = df.drop(columns=columns_to_drop)

#train test split
    df_train = df.iloc[:train.shape[0], :]
    df_test = df.iloc[train.shape[0]:, :]  

#Mise à l'èchelle
    features_to_scale = [ 'transaction_amount','city_population', 'age', 'transaction_distance']
    scaler = StandardScaler()
    df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])
    df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

#Encodage

    #transaction_time
    df_train['transaction_year'] = df_train['transaction_time'].dt.year
    df_train['transaction_minute'] = df_train['transaction_time'].dt.minute
    df_train['transaction_second'] = df_train['transaction_time'].dt.second
    df_train['transaction_day_of_week'] = df_train['transaction_time'].dt.dayofweek

    df_test['transaction_year'] = df_test['transaction_time'].dt.year
    df_test['transaction_minute'] = df_test['transaction_time'].dt.minute
    df_test['transaction_second'] = df_test['transaction_time'].dt.second
    df_test['transaction_day_of_week'] = df_test['transaction_time'].dt.dayofweek

    
    df_train.drop('transaction_time', axis = 1, inplace =True)
    df_test.drop('transaction_time', axis = 1, inplace =True)

    # Encoder les colonnes catégoriques
    encoder = LabelEncoder()


    df_train['category'] = encoder.fit_transform(df_train['category'])
    df_train['job'] = encoder.fit_transform(df_train['job'])
    df_train['city'] = encoder.fit_transform(df_train['city'])
    df_train['state'] = encoder.fit_transform(df_train['state'])

    df_test['category'] = encoder.fit_transform(df_test['category'])
    df_test['job'] = encoder.fit_transform(df_test['job'])
    df_test['city'] = encoder.fit_transform(df_test['city'])
    df_test['state'] = encoder.fit_transform(df_test['state'])

    #colonne gender 
    gender_mapping = {"F": 0, "M": 1}

    df_train["gender_binary"] = df_train["gender"].map(gender_mapping)
    df_test["gender_binary"] = df_test["gender"].map(gender_mapping)

    df_train.drop('gender', axis = 1, inplace =True)
    df_test.drop('gender', axis = 1, inplace =True)

    # ajouter un header au sidebar
    st.sidebar.header("Barre de contrôle ")

    # checkbox pour afficher les données
    st.sidebar.write("")
    if st.sidebar.checkbox("Afficher les données", False):
        st.write("")
        st.subheader("Les données d'entrainement suite au praitretement,mise à l'échelle et encodage")
        st.write(df_train)
    st.sidebar.write("")

    # séparation des features et de la var cible
    X_train = df_train.drop(['is_fraud'], axis = 1)
    y_train = df_train['is_fraud']
    X_test = df_test.drop(['is_fraud'], axis = 1)
    y_test = df_test['is_fraud']
    st.sidebar.write("")

    
    # sélectionner le type de calibrage
    st.sidebar.subheader("Calibrage : ")
    st.write("")
    sampling_method = st.sidebar.selectbox("Choisir la méthode de rééchantillonnage", ("None", "Undersampling", "SMOTE", "Oversampling"))
    

    #calibrage
    if sampling_method == "Undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    elif sampling_method == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    elif sampling_method == "Oversampling":
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.sidebar.write("")

# Checkbox pour afficher la distribution de la variable cible
    if st.sidebar.checkbox("Afficher la distribution de la variable cible", False):
        st.subheader(f"**Distribution de la variable cible suite au calibrage**")
        st.write("")
        dist_target_resampled = pd.Series(y_train_resampled).value_counts()
        fig, ax = plt.subplots(figsize=(3, 4))
        dist_target_resampled.plot(kind='bar', color=['#abebc6', '#edbb99'], ax=ax)
        plt.xticks(range(2), ['Normal = 0', 'Fraud = 1'], rotation=0)
        for i, v in enumerate(dist_target_resampled):
            ax.text(i - 0.1, v + 0.05, str(v), color='black')
        plt.title(f"{sampling_method}")
        st.pyplot(fig)
    st.sidebar.write("")

    # les modèles disponibles en fonction du type de calibrage
    if sampling_method in ("Undersampling", "None"):
        model_options = ("Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "GradientBoostingClassifier")
    else:
        model_options = ("Logistic Regression", "Decision Tree", "Random Forest", "GradientBoostingClassifier")

    # Sélection du modèle
    st.sidebar.subheader("Modèle : ")
    model_name = st.sidebar.selectbox("Choisir le modèle", model_options)

    # choisir les hyperparamétres des modèles
    st.sidebar.write("")
    st.sidebar.subheader("Les paramétres du modéle : ")
    st.sidebar.write("")


    if model_name == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization strength)", 0.01, 10.0)
        max_iter = st.sidebar.slider("max_iter", 100, 300)
        penalty = st.sidebar.selectbox("penalty", ['l1', 'l2', None])
        solver = st.sidebar.selectbox("solver", ['lbfgs', 'liblinear', 'saga'])
        model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)
    
    elif model_name == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("n_neighbors", 3, 9)
        weights = st.sidebar.selectbox("weights", ['uniform', 'distance'])
        metric = st.sidebar.selectbox("metric", ["euclidean", "manhattan"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    
    elif model_name == "Decision Tree":
        max_depth = st.sidebar.slider("max_depth", 1, 40)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 4)
        criterion = st.sidebar.selectbox("criterion", ['gini', 'entropy'])
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf, criterion=criterion)
    
    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, step=50)
        max_depth = st.sidebar.slider("max_depth", None, 20)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 4)
        criterion = st.sidebar.selectbox("criterion", ['gini', 'entropy'])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)

    elif model_name == "GradientBoostingClassifier":
        n_estimators = st.sidebar.slider("n_estimators", 100, 300, step=50)
        max_depth = st.sidebar.slider("max_depth", None, 5)
        learning_rate = st.sidebar.slider("learning_rate", 0.1, 10.0)
        subsample = st.sidebar.slider("subsample", 0.8, 1.0)

        model = GradientBoostingClassifier(n_estimators=n_estimators, 
         learning_rate=learning_rate,
          subsample=subsample, max_depth=max_depth)    
    st.sidebar.write("")


    # Entraînement et évaluation du modèle

    if st.sidebar.button("Entraîner le modèle"):
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]  
        
        st.subheader(f"Évaluation du modèle : {model_name}")
        st.write("")
        st.write("")



        # Afficher la matrice de confusion 

        st.subheader("Matrice de confusion")
        st.write("")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='PiYG', fmt='d', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.write("")
        

       # Afficher les métriques 
        st.write("")
        st.subheader("Les métriques")
        metrics_dict = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC Score": roc_auc_score(y_test, pred_proba)
        }
        
        st.table(pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Score']))



#css 
page_bg = """"

<style>
[data-testid="stApp"]{
background-color: #ffffff;

}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0)}
[data-testid="stSidebar"]{
background-color: # }
[data-testid="stHeading"]{
text-align: center; }
[data-testid="stButton"]{
display: flex;
justify-content: center; }
</style>
"""
st.markdown(page_bg,unsafe_allow_html=True)

if __name__ == '__main__':
    main()
