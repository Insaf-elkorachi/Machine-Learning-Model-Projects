# Importation des bibliothèques nécessaires
from sklearn.preprocessing import LabelEncoder               
from sklearn.model_selection import train_test_split          
from sklearn.naive_bayes import GaussianNB                    
from sklearn.metrics import accuracy_score, classification_report 
from pre_traitement.clean import detect_target_column          

#Encodage des données
def encode_data(data):
    #Création d'un encodeur
    le = LabelEncoder()   
    #Parcours de chaque colonne contenant du texte
    for column in data.select_dtypes(include=['object']).columns:
        #Transformation en valeurs numériques
        data[column] = le.fit_transform(data[column])  
    return data  


#Fonction d'entraînement du modèle NB
def naive_bayes_model(X, y):
    #Division des données (70% entraînement / 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    #Initialisation du modèle Naive Bayes Gaussien
    model = GaussianNB()
    model.fit(X_train, y_train)   #Entraînement sur les données d'apprentissage
    
    #Prédictions sur le jeu de test
    y_pred = model.predict(X_test)
    
    #Calcul des métriques de performance
    #Précision globale du modèle
    accuracy = accuracy_score(y_test, y_pred)      
    #Rapport complet (precision, recall, f1-score)
    report = classification_report(y_test, y_pred) 
    
    #Renvoie le modèle et ses performances
    return model, accuracy, report  


#Prétraitement des données avant application NB
def preprocess_data(data):
    #Encodage des colonnes catégorielles
    data = encode_data(data)

    #Détection automatique de la cible + type de problème
    target_column, problem_type = detect_target_column(data)
    
    #Si aucune cible ou mauvais problème → on arrête
    if target_column is None or problem_type is None:
        return None, None, None, None

    #Naive Bayes ne s'applique qu'aux problèmes de classification
    if problem_type != "classification":
        return None, None, None, None

    #Séparation entre X et y (cible)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    #Si la cible est encore textuelle, on l'encode
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    #Données prêtes pour le modèle
    return X, y, target_column, problem_type  


#Fonction principale d'exécution Naive Bayes
def naive_bayes(data):
    try:
        #Prétraitement complet
        X, y, target_column, problem_type = preprocess_data(data)
        
        # Vérification si les données sont valides
        if X is None or y is None:
            return None, 0, "Error: Invalid data format"

        #Entraînement et Évaluation du modèle
        model, accuracy, report = naive_bayes_model(X, y)
        
        #Renvoie les résultats finaux
        return model, accuracy, report  
        
    except Exception as e:
        # Gestion des erreurs pour éviter un crash complet
        print(f"Error in naive_bayes: {e}")
        return None, 0, str(e)
