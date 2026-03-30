# 📌 Importation des bibliothèques nécessaires

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.svm import SVC, SVR                                   
from sklearn.model_selection import train_test_split              
from sklearn.metrics import accuracy_score, mean_squared_error    
from pre_traitement.clean import detect_target_column             

#Encoder les colonnes catégorielles
def encode_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

#Normaliser les colonnes numériques (moyenne=0, écart-type=1)
def normalize_data(data):
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

#Prétraitement complet : -Copie du dataset pour éviter les modifications directes - Encodage des colonnes catégorielles - Normalisation des colonnes numériques - Détection de la colonne cible et du type de problème
def preprocess_data(data):
    data = data.copy()
    
    #Encodage et normalisation
    data = encode_data(data)
    data = normalize_data(data)
    
    #Détection de la cible
    target_column, _ = detect_target_column(data)
    if not target_column:
        print("Impossible de détecter une colonne cible.")
        return None, None, None, None
        
    #Déterminer le type de problème
    n_unique = len(data[target_column].unique())
    problem_type = "classification" if n_unique <= 10 else "regression"
    
    #Séparation en X et y (cible)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    #Conversion en int si classification
    if problem_type == "classification":
        y = y.astype(int)
    
    return X, y, target_column, problem_type

#Appliquer SVM avec tous les noyaux : 'linear', 'poly', 'rbf', 'sigmoid'
def svm_all_kernels(X_train, y_train, X_test, y_test, problem_type):
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = []
    
    for kernel in kernels:
        try:
            print(f"\nSVM avec le noyau : {kernel}")
            
            #Initialisation du modèle selon le problème
            if problem_type == "classification":
                model = SVC(kernel=kernel, random_state=42)
                metric_name = "Accuracy"
            else:
                model = SVR(kernel=kernel)
                metric_name = "MSE"
                
            #Entraînement du modèle
            model.fit(X_train, y_train)
            
            #Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            #Calcul des métriques
            if problem_type == "classification":
                train_metric = accuracy_score(y_train, y_pred_train)
                test_metric = accuracy_score(y_test, y_pred_test)
            else:
                train_metric = mean_squared_error(y_train, y_pred_train)
                test_metric = mean_squared_error(y_test, y_pred_test)
                
            #Affichage des résultats
            print(f"{metric_name} (Train) : {train_metric:.2f}")
            print(f"{metric_name} (Test) : {test_metric:.2f}")
            
            #Stockage des résultats
            results.append({
                "kernel": kernel,
                "train_metric": train_metric,
                "test_metric": test_metric
            })
            
        except Exception as e:
            print(f"Erreur avec le noyau {kernel}: {str(e)}")
            continue
            
    return results

#Exécution du SVM : -Prétraitement -Division train/test -Application SVM avec tous les noyaux -Retour de la meilleure métrique et du noyau associé
def SVM(data):
    #Prétraitement
    X, y, target_column, problem_type = preprocess_data(data)
    if X is None or y is None or problem_type is None:
        print("Les données ne sont pas adaptées pour le modèle SVM.")
        return

    print(f"Colonne cible détectée : {target_column}")
    print(f"Type de problème détecté : {problem_type}")

    #Division en train/test (70%/30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Exécution SVM avec tous les noyaux
    print("\nExécution du modèle SVM avec tous les noyaux disponibles.")
    results = svm_all_kernels(X_train, y_train, X_test, y_test, problem_type)

    #Sélection du meilleur noyau selon la métrique test
    best_result = max(results, key=lambda x: x['test_metric'])
    
    metric = best_result['test_metric']
    params = {'best_kernel': best_result['kernel']}

    #Affichage détaillé de tous les résultats
    print("\nRésultats pour tous les noyaux:")
    for result in results:
        kernel = result["kernel"]
        train_metric = result["train_metric"]
        test_metric = result["test_metric"]
        print(f"Noyau : {kernel} | Train Metric : {train_metric:.2f} | Test Metric : {test_metric:.2f}")

    return metric, params
