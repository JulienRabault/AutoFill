import argparse
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs='+', type=str, help="Chemins des fichiers source (plusieurs possibles)")
parser.add_argument("output", type=str, help="Chemin du fichier de sortie")
parser.add_argument("-s","sep", type=str, help="Seperateur pour les csv entrant", default=';')
args = parser.parse_args()

all_rows = []

for input_file in args.inputs:
    df = pd.read_csv(input_file, sep=args.sep, dtype=str) 


    def extract_d_h(dimension):
        match = re.search(r'd=(\d+(\.\d+)?)\s+l=(\d+(\.\d+)?)', str(dimension))
        if match:
            return float(match.group(1)), float(match.group(3))
        return None, None

    if "dimension" in list(df.columns):
        df['d'], df['h'] = zip(*df['dimension'].apply(extract_d_h))
        df = df.drop(columns=['dimension'])

    if "concentration" in list(df.columns):
        df['concentration'] = df['concentration'].astype(float)


    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Traitement de {input_file}"):
        raw_path = row['path'].replace('\\', '/')
        normalized_path = Path(raw_path).as_posix() if os.name != 'nt' else str(Path(raw_path))
        row['path'] = normalized_path
        all_rows.append(row)

final_df = pd.DataFrame(all_rows)
final_df.to_csv(args.output, index=False, sep=',')
print(f"Fichier traité sauvegardé sous : {args.output}")
print(f"Nombre total de lignes fusionnées : {len(final_df)}")
