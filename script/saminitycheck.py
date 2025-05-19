import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="/projects/pnria/DATA/AUTOFILL/v2/all_data_les_v2.csv")
parser.add_argument("--basedir", type=str, default="/projects/pnria/DATA/AUTOFILL/")
args = parser.parse_args()

df = pd.read_csv(args.csv)
base = Path(args.basedir)

path_cols = [col for col in df.columns if 'path' in col.lower() and 'optical' not in col.lower()]
print(f"Colonnes contenant 'path' : {path_cols}")

missing_per_column = {}

for col in path_cols:
    print()
    print(f"--- Vérification pour la colonne : {col} ---")
    paths = df[col].astype(str).apply(lambda p: Path(p.replace('\\', '/')).as_posix())
    missing = []

    print(f"Nombre de fichiers à vérifier : {len(paths)}")
    print(f"Fichiers à vérifier (extrait) : {paths.tolist()[:5]}...")

    for rel_path in tqdm(paths, desc=f"Vérification ({col})"):
        rel_path = rel_path.lstrip("/")
        full_path = base / rel_path
        if not full_path.exists():
            print(f"Fichier manquant : {full_path}")
            missing.append(rel_path)

    missing_per_column[col] = missing
    print(f"Fichiers manquants pour {col} : {len(missing)} / {len(paths)}")

print("\n--- Résumé global ---")
for col, missing in missing_per_column.items():
    print(f"{col}: {len(missing)} fichiers manquants")
