#Importation des bibliothèques 

import pandas as pd
from sklearn.preprocessing import LabelEncoder                 
from sklearn.model_selection import train_test_split           
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  
from sklearn.metrics import accuracy_score, mean_squared_error 
from pre_traitement.clean import detect_target_column           


#Encoder les colonnes catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()                     # Initialisation de l'encodeur
        data[column] = le.fit_transform(data[column])  # Transformation des valeurs
    return data


#Fonction d'entraînement KNN selon le type de problème : - classification → KNeighborsClassifier - régression → KNeighborsRegressor
def knn_model(X_train, y_train, X_test, y_test, problem_type, n_neighbors=5):
    #Cas n°1 : Problème de Classification
    if problem_type == "classification":
        # Création du modèle KNN
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        # Apprentissage du modèle  
        model.fit(X_train, y_train)                            

        # Prédiction sur Train
        y_pred_train = model.predict(X_train)            
        # Prédiction sur Test      
        y_pred_test = model.predict(X_test)                    

        # Calcul de l'accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        print(f"Accuracy (Train) : {train_accuracy:.2f}")
        print(f"Accuracy (Test) : {test_accuracy:.2f}")

        return model, train_accuracy, test_accuracy


    #Cas n°2 : Problème de Régression
    elif problem_type == "regression":
        #Modèle KNN pour valeurs continues
        model = KNeighborsRegressor(n_neighbors=n_neighbors)   
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        #Calcul du MSE
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        print(f"MSE (Train) : {train_mse:.2f}")
        print(f"MSE (Test) : {test_mse:.2f}")

        return model, train_mse, test_mse


    #Cas Inconnu : Pas de Détection
    else:
        print("Type de problème inconnu, aucun modèle entraîné.")
        return None, None


#Prétraitement : encodage et détection cible
def preprocess_data(data):
    #Encodage des colonnes texte
    data = encode_data(data)                                

    #Détection de la variable cible
    target_column, problem_type = detect_target_column(data) 

    # Séparation X et y (cible)
    if target_column:
        #Supprimer colonne cible des features
        X = data.drop(columns=[target_column])              
        y = data[target_column]
    else:
        X = data
        y = None

    return X, y, target_column, problem_type


#Fonction principale, lancement du modèle KNN :-Nettoyage et préparation des données -Séparation train/test - Entraînement du modèle KNN adapté au type détecté -Affichage des performances
def KNN(data, n_neighbors=5):
    X, y, target_column, problem_type = preprocess_data(data)

    #Vérification au cas où la détection échoue
    if X is None or y is None or problem_type is None:
        print("Les données ne sont pas adaptées pour le modèle KNN.")
        return

    print(f"Colonne cible détectée : {target_column}")
    print(f"Type de problème détecté : {problem_type}")

    #Division du dataset (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Exécution du modèle KNN")
    
    #Appel du modèle KNN adapté au type
    model, train_metric, test_metric = knn_model(X_train, y_train, X_test, y_test, problem_type, n_neighbors)

    return model, train_metric, test_metric
