#Importation des bibliothèques
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

#Fonction externe pour détecter la colonne cible et le type de problème
from pre_traitement.clean import detect_target_column


#Fonction pour encoder les colonnes catégorielles
#Encode en valeurs numériques toutes les colonnes de type 'object'
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()  # Création de l'encodeur
        data[column] = le.fit_transform(data[column])  # Transformation des valeurs
    return data


#Fonction d'entraînement des modèles -> Choisit automatiquement selon le type de problème
def regression_models(X_train, y_train, X_test, y_test, problem_type):

    #Cas n°1 : Régression
    if problem_type == "regression":

        #Régression Linéaire Simple
        model_simple = LinearRegression()
        model_simple.fit(X_train, y_train)  # Entraînement du modèle

        #Prédictions
        y_pred_train_simple = model_simple.predict(X_train)
        y_pred_test_simple = model_simple.predict(X_test)

        #Mesure d'erreur : MSE
        mse_train_simple = mean_squared_error(y_train, y_pred_train_simple)
        mse_test_simple = mean_squared_error(y_test, y_pred_test_simple)

        print("\nRégression Linéaire Simple")
        print(f"MSE (Train) : {mse_train_simple:.2f}")
        print(f"MSE (Test)  : {mse_test_simple:.2f}")


        #Régression Linéaire Multiple
        model_multiple = LinearRegression()
        model_multiple.fit(X_train, y_train)

        y_pred_train_multiple = model_multiple.predict(X_train)
        y_pred_test_multiple = model_multiple.predict(X_test)

        mse_train_multiple = mean_squared_error(y_train, y_pred_train_multiple)
        mse_test_multiple = mean_squared_error(y_test, y_pred_test_multiple)

        print("\nRégression Linéaire Multiple")
        print(f"MSE (Train) : {mse_train_multiple:.2f}")
        print(f"MSE (Test)  : {mse_test_multiple:.2f}")


        #Régression Polynomiale
        # Ajout de features : x, x², xy,....
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model_poly = LinearRegression()
        model_poly.fit(X_train_poly, y_train)

        y_pred_train_poly = model_poly.predict(X_train_poly)
        y_pred_test_poly = model_poly.predict(X_test_poly)

        mse_train_poly = mean_squared_error(y_train, y_pred_train_poly)
        mse_test_poly = mean_squared_error(y_test, y_pred_test_poly)

        print("\nRégression Polynomiale (degré 2)")
        print(f"MSE (Train) : {mse_train_poly:.2f}")
        print(f"MSE (Test)  : {mse_test_poly:.2f}")


    #Cas n°2 : Classification
    elif problem_type == "classification":

        #Régression Logistique → modèle pour prédire des classes
        model_logistic = LogisticRegression()
        model_logistic.fit(X_train, y_train)

        #Prédictions
        y_pred_train_logistic = model_logistic.predict(X_train)
        y_pred_test_logistic = model_logistic.predict(X_test)

        #Metric : Accuracy = pourcentage de bonnes prédictions
        acc_train_logistic = accuracy_score(y_train, y_pred_train_logistic)
        acc_test_logistic = accuracy_score(y_test, y_pred_test_logistic)

        print("\nRégression Logistique (Classification)")
        print(f"Accuracy (Train) : {acc_train_logistic:.2f}")
        print(f"Accuracy (Test)  : {acc_test_logistic:.2f}")


    else:
        print("Type de problème inconnu, aucun modèle entraîné.")


#Fonction de prétraitement global
def preprocess_data(data):
    #Encodage des données
    data = encode_data(data)

    #Détection automatique de la cible
    target_column, problem_type = detect_target_column(data)

    #Séparation X (features) et y (target)
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X, y = data, None

    return X, y, target_column, problem_type


#Fonction principale à exécuter : -Prétraite les données -Sépare Train/Test - Lance les modèles
def regression_model(data):
    X, y, target_column, problem_type = preprocess_data(data)

    #Séparation Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Entraînement des modèles
    regression_models(X_train, y_train, X_test, y_test, problem_type)