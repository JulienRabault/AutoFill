import pandas as pd
import sys
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_group(group):
    """Fonction exécutée en parallèle pour chaque groupe."""
    group_id, c2_values = group
    unique = len(c2_values) == len(set(zip(*c2_values)))
    return group_id, group_id if unique else None


def main(csv_path, c1_columns, c2_columns):
    # Chargement du CSV avec tqdm
    print("Chargement du fichier CSV...")
    with tqdm(desc="Progression", unit=" lignes") as pbar:
        df = pd.read_csv(csv_path)
        pbar.update(len(df))

    # Création des identifiants de groupe avec tqdm
    print("Création des identifiants de groupe...")
    tqdm.pandas(desc="Hashing des groupes")
    df['group_id'] = df[c1_columns].astype(str).progress_apply(
        lambda x: hashlib.sha256(x.sum().encode()).hexdigest(),
        axis=1
    )

    # Préparation des groupes pour le traitement parallèle
    grouped = df.groupby('group_id')[c2_columns].apply(lambda x: list(map(tuple, x.values)))
    groups = list(grouped.items())

    # Traitement parallèle avec multi-CPU et tqdm
    print("Traitement des groupes en parallèle...")
    pair_index_map = {}
    num_workers = mp.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_group, group): group[0] for group in groups}

        # Barre de progression pour le traitement parallèle
        with tqdm(total=len(groups), desc="Traitement des groupes", unit=" groupe") as pbar:
            for future in as_completed(futures):
                group_id, index = future.result()
                pair_index_map[group_id] = index
                pbar.update(1)

    # Application des résultats au DataFrame
    print("Application des résultats...")
    df['pair_index'] = df['group_id'].map(pair_index_map)

    # Nettoyage des colonnes temporaires
    df.drop('group_id', axis=1, inplace=True)

    # Sauvegarde du fichier avec tqdm
    print("Sauvegarde du fichier...")
    with tqdm(desc="Sauvegarde", total=1) as pbar:
        df.to_csv(csv_path, index=False)
        pbar.update(1)

    print("Traitement terminé avec succès !")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_path> <c1_columns> <c2_columns>")
        print("Exemple: python script.py data.csv material,shape concentration,date")
        sys.exit(1)

    csv_path = sys.argv[1]
    c1 = sys.argv[2].split(',')
    c2 = sys.argv[3].split(',')

    main(csv_path, c1, c2)