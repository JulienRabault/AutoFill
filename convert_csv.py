import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str, default="output.csv")
args = parser.parse_args()

df = pd.read_csv(args.input, sep=';', dtype=str)

def extract_d_h(dimension):
    match = re.search(r'd=(\d+(\.\d+)?)\s+l=(\d+(\.\d+)?)', str(dimension))
    if match:
        return float(match.group(1)), float(match.group(3))
    return None, None

df['d'], df['h'] = zip(*df['dimension'].apply(extract_d_h))

df['concentration'] = df['concentration'].astype(float)

df = df.drop(columns=['dimension'])

df.to_csv(args.output, index=False, sep=',')
