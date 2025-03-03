import pandas as pd
import sys
import hashlib

def compute_pair_index(row, group_identifier, group_to_c2_values):
    group_key = row[group_identifier]
    c2_values = group_to_c2_values[group_key]

    if len(c2_values) == len(set(zip(*c2_values))):
        return group_key
    else:
        return None

def main(csv_path, c1_columns, c2_columns):
    df = pd.read_csv(csv_path)

    df['group_id'] = df[c1_columns].astype(str).sum(axis=1).apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )

    group_to_c2 = df.groupby('group_id')[c2_columns].apply(lambda x: list(map(tuple, x.values))).to_dict()
    df['pair_index'] = df.apply(
        lambda row: compute_pair_index(row, 'group_id', group_to_c2),
        axis=1
    )

    df.drop('group_id', axis=1, inplace=True)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_path> <c1_columns> <c2_columns>")
        sys.exit(1)

    csv_path = sys.argv[1]
    c1 = sys.argv[2].split(',')
    c2 = sys.argv[3].split(',')

    main(csv_path, c1, c2)