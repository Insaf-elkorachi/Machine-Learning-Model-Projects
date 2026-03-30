from __future__ import annotations

from collections.abc import Hashable
from typing import Optional, Tuple

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype

_TARGET_NAME_CANDIDATES = ("target", "label", "y", "class", "classe", "output")


def infer_problem_type(target: pd.Series) -> Optional[str]:
    """Infère le type de problème à partir de la cible.

    Retourne:
    - "classification" si la cible est catégorielle (texte/bool/catégorie) ou si
      elle a peu de valeurs uniques.
    - "regression" sinon.
    - None si la cible est vide/invalide.
    """
    if target is None:
        return None

    target_no_na = target.dropna()
    if target_no_na.empty:
        return None

    # Cible non-numérique => classification
    if is_object_dtype(target_no_na) or is_bool_dtype(target_no_na) or str(target_no_na.dtype) == "category":
        return "classification"

    # Cible numérique => heuristique (peu de classes / ratio faible)
    if is_numeric_dtype(target_no_na):
        n_rows = len(target_no_na)
        n_unique = int(target_no_na.nunique(dropna=True))
        unique_ratio = n_unique / max(1, n_rows)

        if n_unique <= 20 and unique_ratio < 0.2:
            return "classification"

        return "regression"

    # Fallback (datetime, etc.)
    return "regression"


def detect_target_column(df: pd.DataFrame) -> Tuple[Optional[Hashable], Optional[str]]:
    """Détecte une colonne cible et le type de problème.

    Stratégie:
    - Si une colonne dont le nom est l'un des candidats existe (insensible à la casse),
      on la choisit (ex: "target", "label", ...).
    - Sinon, on utilise la dernière colonne.

    Retourne: (target_column_name, problem_type) où problem_type est
    "classification" ou "regression". En cas d'entrée invalide, retourne (None, None).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None, None
    if df.shape[0] == 0 or df.shape[1] == 0:
        return None, None

    lower_to_original = {}
    for column in df.columns:
        if isinstance(column, str):
            lower_to_original[column.strip().lower()] = column

    target_column = None
    for candidate in _TARGET_NAME_CANDIDATES:
        if candidate in lower_to_original:
            target_column = lower_to_original[candidate]
            break

    if target_column is None:
        target_column = df.columns[-1]

    problem_type = infer_problem_type(df[target_column])
    return target_column, problem_type
