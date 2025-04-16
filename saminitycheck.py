import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="/projects/pnria/DATA/AUTOFILL/v2/all_data_les_v2.csv")
parser.add_argument("--basedir", type=str, default="/projects/pnria/DATA/AUTOFILL/")
args = parser.parse_args()

df = pd.read_csv(args.csv)
for el in df.columns:
    if 'path' in el:
        df = df.rename(columns={el: 'path'})
print(df.columns)
df['path'] = df['path'].apply(lambda p: Path(str(p).replace('\\', '/')).as_posix())

base = Path(args.basedir)
missing = []
print()
print(f"Répertoire de base : {base}")
print(f"Nombre de fichiers à vérifier : {len(df)}")
print(f"Fichiers à vérifier : {df['path'].tolist()[:5]}...")
for rel_path in tqdm(df['path'], desc="Vérification des fichiers"):
    rel_path = rel_path.lstrip("/")
    full_path = base / rel_path
    if not full_path.exists():
        print(f"Fichier manquant : {full_path}")
        missing.append(rel_path)


print(f"\nFichiers manquants : {len(missing)} / {len(df)}")
