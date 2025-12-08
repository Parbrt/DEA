import pandas as pd
import requests
import re
import os
import random
from rich.console import Console

console = Console()


def main():
    """Cette fonction prend un client dans la base de données puis appel l'API de docker et affiche si le prêt est accordé ou non"""
    API_URL = "http://127.0.0.1:1234/predict"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_path = os.path.join(project_root, "dataset", "processed", "test_final.parquet")

    console.print(f"[dim]Chargement d'un client réel depuis : {test_path}...[/dim]")

    try:
        df = pd.read_parquet(test_path)
    except Exception:
        console.print(
            "[bold red]Erreur : Impossible de lire le fichier Parquet.[/bold red]"
        )
        console.print("Vérifiez que vous avez bien généré les données à l'étape 1.")
        return

    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))

    random_index = random.randint(0, len(df) - 1)
    client_row = df.iloc[random_index]

    client_data = client_row.astype(float).to_dict()

    if "TARGET" in client_data:
        del client_data["TARGET"]

    payload = {"features": client_data}

    console.rule(
        f"[bold magenta]Test API pour le client Index {random_index}[/bold magenta]"
    )

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            console.print("\n[bold green]RÉPONSE API REÇUE :[/bold green]")
            console.print(
                f"Probabilité de faillite : [bold cyan]{result['probability']:.2%}[/bold cyan]"
            )
            console.print(f"Seuil appliqué          : {result['threshold']}")

            if result["decision"] == 1:
                console.print(
                    f"Décision                : [bold red]{result['decision_label']} (Refus)[/bold red]"
                )
            else:
                console.print(
                    f"Décision                : [bold green]{result['decision_label']} (Accordé)[/bold green]"
                )

        else:
            console.print(f"[bold red]Erreur API ({response.status_code}) :[/bold red]")
            console.print(response.text)

    except requests.exceptions.ConnectionError:
        console.print("[bold red]Impossible de se connecter à l'API.[/bold red]")
        console.print(
            "Vérifiez que le conteneur Docker tourne bien : 'docker run -p 1234:1234 credit-scoring-app'"
        )


if __name__ == "__main__":
    main()
