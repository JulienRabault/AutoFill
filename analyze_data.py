import csv
from collections import defaultdict, Counter


class CSVLoader:
    # Classe pour charger un fichier CSV
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> list:
        data = []
        with open(self.filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data


class CSVAnalyzer:
    # Classe pour analyser les colonnes material, technique et shape
    def __init__(self, data: list):
        self.data = data
        self.total = len(data)

    def count_individual(self, column: str) -> dict:
        counts = dict(Counter(row[column] for row in self.data))
        return counts

    def count_combinations(self) -> dict:
        # Combinaison de material, technique et shape uniquement
        combinations = [(row['material'], row['technique'], row['shape']) for row in self.data]
        return dict(Counter(combinations))

    def count_correlations(self, variable: str) -> int:
        """
        Compte le nombre de cas où seule la variable spécifiée change, peu importe les autres métadonnées.
        """
        if not self.data:
            return 0

        # Dictionnaire pour regrouper les valeurs de la variable en fonction des autres métadonnées
        groups = defaultdict(set)

        # Liste des en-têtes sauf la variable à analyser
        headers = [header for header in self.data[0].keys() if header != variable]

        for row in self.data:
            # Créer une clé basée sur toutes les métadonnées sauf la variable analysée
            key = tuple(row[header] for header in headers)
            # Ajouter la valeur de la variable analysée
            groups[key].add(row[variable])

        # Compter les cas où seule la variable change
        count = 0
        for key, values in groups.items():
            if len(values) > 1:
                print(f"Corrélation trouvée pour '{variable}':")
                print(f"  - Clé commune (autres métadonnées): {key}")
                print(f"  - Valeurs distinctes de '{variable}': {values}")
                count += 1

        return count


def main():
    loader = CSVLoader("../../DATA/AUTOFILL/merged_cleaned_data.csv")
    data = loader.load()

    analyzer = CSVAnalyzer(data)
    total = analyzer.total
    count_material = analyzer.count_individual('material')
    count_technique = analyzer.count_individual('technique')
    count_shape = analyzer.count_individual('shape')

    print("== Comptage individuel ==")
    print("Material:")
    for key, value in sorted(count_material.items()):
        pct = value / total * 100
        print(f"  - {key}: {value} ({pct:.2f}%)")
    print("\nTechnique:")
    for key, value in sorted(count_technique.items()):
        pct = value / total * 100
        print(f"  - {key}: {value} ({pct:.2f}%)")
    print("\nShape:")
    for key, value in sorted(count_shape.items()):
        pct = value / total * 100
        print(f"  - {key}: {value} ({pct:.2f}%)")

    # Ajout de l'analyse des corrélations
    print("\n== Analyse des corrélations ==")
    correlation_count_shape = analyzer.count_correlations('shape')
    print(f"Nombre de corrélations pour 'shape': {correlation_count_shape}")

    correlation_count_material = analyzer.count_correlations('material')
    print(f"Nombre de corrélations pour 'material': {correlation_count_material}")

    correlation_count_technique = analyzer.count_correlations('technique')
    print(f"Nombre de corrélations pour 'technique': {correlation_count_technique}")


if __name__ == "__main__":
    main()