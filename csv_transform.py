import pandas as pd
import re
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Chemin du fichier source")
parser.add_argument("output", type=str, help="Chemin du fichier de sortie")
parser.add_argument("dir", type=str, help="Répertoire contenant les fichiers référencés dans la colonne path")
args = parser.parse_args()

shutil.copy(args.input, args.input + ".bak")
print(f"Copie de sauvegarde créée : {args.input}.bak")

df = pd.read_csv(args.input, sep=';', dtype=str)

def extract_d_h(dimension):
    match = re.search(r'd=(\d+(\.\d+)?)\s+l=(\d+(\.\d+)?)', str(dimension))
    if match:
        return float(match.group(1)), float(match.group(3))
    return None, None

df['d'], df['h'] = zip(*df['dimension'].apply(extract_d_h))

df['concentration'] = df['concentration'].astype(float)

df = df.drop(columns=['dimension'])

# df.insert(len(df.columns) - 1, 'pair_index', '')

valid_rows = []
removed_count = 0
base_dir = Path(args.dir)
for _, row in tqdm(df.iterrows(), total=len(df), desc="Vérification des fichiers"):
    row['path'] = Path(row['path'].replace('\\', '/')).as_posix()
    file_path = base_dir / row['path']
    if file_path.exists():
        valid_rows.append(row)
    else:
        removed_count += 1
        print(f"Fichier non trouvé, supprimé : {file_path}")
df = pd.DataFrame(valid_rows)

df.to_csv(args.output, index=False, sep=',')
print(f"Fichier traité sauvegardé sous : {args.output}")
print(f"Nombre total de lignes supprimées : {removed_count}")
