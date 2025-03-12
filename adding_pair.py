import pandas as pd
import itertools
from tqdm import tqdm
import logging
import argparse
import shutil
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# adding_pair.py --input all_data.csv --output all_data_pair.csv --common shape material --diff technique

def process_group(pair_dict, pair_id, group, diff_column):
    group_pairs = []
    total_pairs = 0
    indices = group.index.tolist()
    logging.info("Processing group of %d rows", len(indices))
    for i, j in itertools.combinations(indices, 2):
        if group.at[i, diff_column] != group.at[j, diff_column]:
            pair_dict[i].append(pair_id)
            pair_dict[j].append(pair_id)
            group_pairs.append(pair_id)
            pair_id += 1
            total_pairs += 1
    return pair_dict, pair_id, total_pairs, group_pairs


def create_pairs_csv(input_file, output_file, common_columns, diff_column):
    # Copier le fichier avant traitement
    backup_file = f"{input_file}_backup.csv"
    shutil.copy(input_file, backup_file)
    logging.info("Backup created at %s", backup_file)

    df = pd.read_csv(input_file)

    missing_cols = [col for col in common_columns + [diff_column] if col not in df.columns]
    if missing_cols:
        logging.error("Missing columns in CSV: %s", missing_cols)
        exit(1)

    logging.info("Grouping by columns: %s", common_columns)
    pair_dict = {i: [] for i in df.index}
    pair_id = 1
    grouped = df.groupby(common_columns)

    total_pairs = 0
    all_group_pairs = []
    logging.info("")
    logging.info("Starting processing ")
    with Pool() as pool:
        # Traiter les groupes en parallèle
        results = list(tqdm(pool.starmap(process_group,
                                         [(pair_dict, pair_id, group, diff_column) for _, group in grouped]),
                            total=len(grouped), desc="Processing groups"))

        for result in results:
            pair_dict, pair_id, group_total_pairs, group_pairs = result
            total_pairs += group_total_pairs
            all_group_pairs.extend(group_pairs)

    df['pair_index'] = df.index.map(lambda i: ",".join(map(str, pair_dict[i])) if pair_dict[i] else "")

    df.to_csv(output_file, index=False)
    logging.info("CSV output saved to %s", output_file)
    logging.info("Total unique pairs created: %d", total_pairs)
    logging.info("Total rows processed: %d", len(df))
    logging.info("Rows with at least one pair: %d", sum(df['pair_index'] != ""))
    logging.info("Rows without a pair: %d", sum(df['pair_index'] == ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--common", nargs="+", required=True, help="List of common metadata columns")
    parser.add_argument("--diff", required=True, help="Column that must differ")
    args = parser.parse_args()

    logging.info("Starting processing")
    create_pairs_csv(args.input, args.output, args.common, args.diff)
    logging.info("Processing finished")
