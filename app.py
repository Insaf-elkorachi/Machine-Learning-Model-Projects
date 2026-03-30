import pandas as pd
import streamlit as st
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

from pre_traitement.clean import detect_target_column, infer_problem_type

st.set_page_config(page_title="Decision Tree", layout="wide")

st.title("🌳 Decision Tree")
st.subheader("📌 Importez un dataset .csv puis entraînez un modèle (classification ou régression).")

uploaded_file = st.file_uploader("📂 Choisissez un fichier CSV", type=["csv"])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        else:
            try:
                df[col] = df[col].fillna(df[col].mean())
            except Exception:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder | None]:
    if y.dtype == "object" or str(y.dtype) == "category":
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return y_encoded, le

    if str(y.dtype) == "bool":
        return y.astype(int), None

    return y, None


def try_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, bool]:
    if len(X) <= 20:
        return X, y, False
    if y.nunique(dropna=True) < 2:
        return X, y, False

    min_class_count = int(y.value_counts().min())
    k_neighbors = min(4, min_class_count - 1)
    if k_neighbors < 1:
        return X, y, False

    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, True
    except Exception:
        return X, y, False


def train_decision_tree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
):
    if problem_type == "regression":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results = {
            "mse": float(mean_squared_error(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2_score": float(r2_score(y_test, preds)),
        }
        return model, results

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    return model, results


if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Impossible de lire le CSV : {e}")
        st.stop()

    if data.empty:
        st.error("❌ Dataset vide.")
        st.stop()

    st.success("✅ Fichier chargé avec succès !")
    st.write("📄 Aperçu du dataset :")
    st.dataframe(data.head())

    detected_target, detected_problem = detect_target_column(data)
    columns = list(data.columns)

    default_target = detected_target if detected_target in columns else columns[-1]
    target_column = st.selectbox("Colonne cible (y) :", columns, index=columns.index(default_target))

    inferred_problem = infer_problem_type(data[target_column])
    default_problem = inferred_problem or detected_problem or "classification"
    problem_type = st.selectbox(
        "Type de problème :",
        ["classification", "regression"],
        index=0 if default_problem == "classification" else 1,
    )

    use_smote = st.checkbox(
        "Appliquer SMOTE (classification)",
        value=True,
        disabled=(problem_type != "classification"),
    )

    if st.button("🚀 Entraîner le modèle"):
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]

            if X.shape[1] == 0:
                st.error("❌ Aucune feature disponible : le dataset ne contient qu'une colonne.")
                st.stop()

            # Retirer les lignes sans cible
            mask_y = y.notna()
            X = X.loc[mask_y]
            y = y.loc[mask_y]

            if len(X) < 2:
                st.error("❌ Pas assez de lignes exploitables (cible manquante).")
                st.stop()

            X = handle_missing_values(X)
            X = encode_features(X)

            y_encoder = None
            if problem_type == "regression":
                y = pd.to_numeric(y, errors="coerce")
                mask_num = y.notna()
                X = X.loc[mask_num]
                y = y.loc[mask_num]
                if len(X) < 2:
                    st.error("❌ La cible n'est pas numérique : impossible de faire une régression.")
                    st.stop()
            else:
                y, y_encoder = encode_target(y)

            did_smote = False
            if problem_type == "classification" and use_smote:
                X, y, did_smote = try_smote(X, y)
                if did_smote:
                    st.info("✅ SMOTE appliqué : classes équilibrées.")

            stratify = y if problem_type == "classification" and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )

            model, results = train_decision_tree(X_train, X_test, y_train, y_test, problem_type)
            st.success("🎯 Modèle entraîné avec succès !")

            if y_encoder is not None:
                mapping = {int(i): str(label) for i, label in enumerate(y_encoder.classes_)}
                st.write("🏷️ Encodage de la cible :", mapping)

            st.write("### 📊 Résultats :")
            st.json(results)

        except Exception as e:
            st.error(f"⚠️ Erreur lors de l'entraînement : {e}")
else:
    st.info("📥 Veuillez importer un fichier .csv pour commencer.")
