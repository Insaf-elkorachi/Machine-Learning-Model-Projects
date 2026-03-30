import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pre_traitement.clean import detect_target_column


#############################################
#             PREPROCESSING
#############################################


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes par la moyenne (numeriques)
    ou la valeur la plus frequente (categorielle).
    """
    data = data.copy()
    for col in data.columns:
        if data[col].dtype == "object" or str(data[col].dtype) == "category":
            mode = data[col].mode(dropna=True)
            if not mode.empty:
                data[col] = data[col].fillna(mode.iloc[0])
        else:
            data[col] = data[col].fillna(data[col].mean())
    return data


def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes de type 'object' en valeurs numeriques avec LabelEncoder.
    Attention : a eviter pour du texte libre !
    """
    data = data.copy()
    for column in data.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
    return data


def balance_data(X: pd.DataFrame, y: pd.Series, data: pd.DataFrame):
    """
    Utilise SMOTE si le dataset est en desequilibre et assez large.
    """
    _, problem_type = detect_target_column(data)

    if problem_type != "classification" or len(X) <= 20:
        print("[INFO] Pas d'equilibrage (regression ou dataset trop petit).")
        return X, y

    if y.nunique(dropna=True) < 2:
        print("[INFO] Pas d'equilibrage (une seule classe).")
        return X, y

    min_class_count = int(y.value_counts().min())
    k_neighbors = min(4, min_class_count - 1)
    if k_neighbors < 1:
        print("[INFO] Pas d'equilibrage (classe minoritaire trop petite pour SMOTE).")
        return X, y

    try:
        smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=k_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("[OK] SMOTE applique : donnees equilibrees.")
        return X_balanced, y_balanced
    except Exception as e:
        print(f"[INFO] SMOTE ignore (erreur: {e}).")
        return X, y


#############################################
#             MODELISATION
#############################################


def decision_tree_model(X_train, X_test, y_train, y_test, problem_type):
    if problem_type == "classification":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(y_test, predictions),
        }

        print("\n[RESULTATS] Classification")
        for k, v in results.items():
            if not isinstance(v, (list, dict)):
                print(f"{k} : {v}")
        print("\nMatrice de confusion :\n", results["confusion_matrix"])
        print("\nRapport de classification :\n", results["classification_report"])

        return model, results

    if problem_type == "regression":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2_score": r2_score(y_test, predictions),
        }

        print("\n[RESULTATS] Regression")
        for k, v in results.items():
            print(f"{k} : {v}")

        return model, results

    print("[ERREUR] Type de probleme inconnu (ni classification ni regression).")
    return None, None


#############################################
#             PIPELINE COMPLET
#############################################


def preprocess_data(data: pd.DataFrame):
    """
    Pipeline de pretraitement : verification, nettoyage, encodage, split...
    """
    try:
        print("\n[INFO] Pretraitement en cours...")

        data = handle_missing_values(data)  # Gestion des NaN
        data = encode_data(data)  # Encodage

        target_column, problem_type = detect_target_column(data)

        if not target_column:
            print("[ERREUR] Aucune colonne cible detectee.")
            return None, None, None, None

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X, y = balance_data(X, y, data)
        return X, y, target_column, problem_type

    except Exception as e:
        print(f"[ERREUR] Erreur dans preprocess_data : {str(e)}")
        return None, None, None, None


#############################################
#                 FONCTION PRINCIPALE
#############################################


def DT(data):
    """
    Fonction principale :
    - Nettoie
    - Detecte le probleme
    - Entraine l'arbre de decision
    - Retourne le modele + metriques
    """
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if data.empty:
            print("[ERREUR] Dataset vide.")
            return None, {"error": "Empty dataset"}

        _, problem_type = detect_target_column(data)
        if problem_type == "clustering":
            print("[ERREUR] Les arbres de decision ne supportent pas le clustering.")
            return None, {"error": "Unsupported type: clustering"}

        X, y, target_column, problem_type = preprocess_data(data)
        if X is None:
            return None, {"error": "Preprocessing failed"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, results = decision_tree_model(X_train, X_test, y_train, y_test, problem_type)
        return model, results

    except Exception as e:
        print(f"[ERREUR] Erreur dans DT() : {str(e)}")
        return None, {"error": str(e)}
