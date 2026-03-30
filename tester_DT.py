import pandas as pd

from Decision_Tree import DT


def tester_csv():
    print("=== Test du modèle Decision Tree ===")
    csv_path = input(
        "Entrez le chemin du fichier CSV (ex: data.csv) "
        "[défaut: Online Course Engagement.csv] : "
    ).strip()

    if not csv_path:
        csv_path = "Online Course Engagement.csv"

    try:
        data = pd.read_csv(csv_path)
        print("\n[OK] Dataset chargé avec succès !")
        print(data.head())

        model, results = DT(data)

        if model is None:
            print("\n[ERREUR] Erreur lors de l'entraînement :", results)
            return

        print("\n[OK] Modèle entraîné avec succès !")
        print("Résultats :\n", results)

    except FileNotFoundError:
        print("[ERREUR] Fichier introuvable : vérifie le chemin saisi.")
    except Exception as e:
        print("[ERREUR] Erreur lors du test :", str(e))


if __name__ == "__main__":
    tester_csv()
