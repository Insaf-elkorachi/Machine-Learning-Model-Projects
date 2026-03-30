# 📌 Importation des bibliothèques nécessaires

import pandas as pd
from sklearn.preprocessing import LabelEncoder                     
from sklearn.model_selection import train_test_split                
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
from sklearn.metrics import accuracy_score, mean_squared_error      
from pre_traitement.clean import detect_target_column              

#Encoder les colonnes catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()                     # Initialisation d'un encodeur
        data[column] = le.fit_transform(data[column])  # Transformation de la colonne
    return data

#Fonction pour entraîner et évaluer le modèle Random Forest : - classification : RandomForestClassifier -régression : RandomForestRegressor 
def random_forest_model(X_train, y_train, X_test, y_test, problem_type, n_estimators=100):
    #Cas : Classification
    if problem_type == "classification":
        #Création du classifieur
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42) 
        #Entraînement du modèle
        model.fit(X_train, y_train)  

        #Prédictions sur train et test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        #Évaluation du modèle
        train_accuracy = accuracy_score(y_train, y_pred_train)  # Accuracy sur le train
        test_accuracy = accuracy_score(y_test, y_pred_test)     # Accuracy sur le test

        print(f"Accuracy (Train) : {train_accuracy:.2f}")
        print(f"Accuracy (Test) : {test_accuracy:.2f}")

        return model, train_accuracy, test_accuracy

    #Cas : Régression
    elif problem_type == "regression":
        # Création du régressuer
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        # Entraînement  
        model.fit(X_train, y_train)  

        #Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        #Évaluation du modèle : 
        # Erreur quadratique moyenne sur train
        train_mse = mean_squared_error(y_train, y_pred_train)
        # Erreur quadratique moyenne sur test  
        test_mse = mean_squared_error(y_test, y_pred_test)     

        print(f"MSE (Train) : {train_mse:.2f}")
        print(f"MSE (Test) : {test_mse:.2f}")

        return model, train_mse, test_mse

    #Cas : Type inconnu
    else:
        print("Type de problème inconnu, aucun modèle entraîné.")
        return None, None

#Fonction pour prétraiter les données
def preprocess_data(data):
    # Encodage
    data = encode_data(data)                  
    # Détection de la cible et du type   
    target_column, problem_type = detect_target_column(data)  

    if target_column:
        X = data.drop(columns=[target_column])  # Variables explicatives
        y = data[target_column]                 # Variable cible
    else:
        X = data
        y = None

    return X, y, target_column, problem_type

#Fonction principale : exécution du modèle
def RF(data, n_estimators=100):
    # Prétraitement
    X, y, target_column, problem_type = preprocess_data(data)

    if X is None or y is None or problem_type is None:
        print("Les données ne sont pas adaptées pour le modèle Random Forest.")
        return

    print(f"Colonne cible détectée : {target_column}")
    print(f"Type de problème détecté : {problem_type}")

    #Division train/test (70% / 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Exécution du modèle
    print("Exécution du modèle Random Forest...")
    model, train_metric, test_metric = random_forest_model(X_train, y_train, X_test, y_test, problem_type, n_estimators)

    return model, train_metric, test_metric
