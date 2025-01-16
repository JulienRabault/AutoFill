import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dataset import CustomDatasetVAE


def analyze_and_visualize(dataset, dataframe, categorical_cols, numerical_cols):
    print("\n=== Analyses et Visualisations Optimisées ===")

    # 1. Analyse Descriptive des Métadonnées
    print("\n=== Analyse Descriptive des Métadonnées ===")
    analyze_categorical(dataframe, categorical_cols)
    analyze_numerical(dataframe, numerical_cols)
    analyze_correlation(dataframe, numerical_cols)

    # 2. Initialiser les variables pour l'analyse des données brutes
    total_lengths = []
    length_distribution = defaultdict(int)
    longest_file_length = 0
    longest_file_index = -1
    all_data = []

    # Listes pour stocker les statistiques par échantillon
    current_length_list = []
    data_mean_list = []
    data_std_list = []
    data_max_list = []
    data_min_list = []

    # Pour la visualisation des échantillons
    sample_indices = np.linspace(0, len(dataset) - 1, num=5, dtype=int)
    sample_data = []
    sample_metadata = []

    print("\n=== Analyse des Données Brutes et Collecte de Statistiques ===")
    for idx in tqdm(range(len(dataset)), desc="Traitement des échantillons"):
        try:
            data, metadata = dataset[idx]

            # Collecte des statistiques sur la longueur des données
            current_length = (data != 0).sum(dim=0)[0].item()
            total_lengths.append(current_length)
            length_distribution[current_length] += 1

            if current_length > longest_file_length:
                longest_file_length = current_length
                longest_file_index = idx

            # Collecte des données brutes pour les statistiques globales
            if data.numel() > 0:
                all_data.append(data.numpy())
                # Calcul des statistiques par échantillon
                sample_mean = data.mean().item()
                sample_std = data.std().item()
                sample_max = data.max().item()
                sample_min = data.min().item()

                data_mean_list.append(sample_mean)
                data_std_list.append(sample_std)
                data_max_list.append(sample_max)
                data_min_list.append(sample_min)
            else:
                data_mean_list.append(np.nan)
                data_std_list.append(np.nan)
                data_max_list.append(np.nan)
                data_min_list.append(np.nan)

            current_length_list.append(current_length)

            # Collecte des échantillons pour la visualisation
            if idx in sample_indices:
                sample_data.append(data.numpy())
                sample_metadata.append(metadata.numpy())

        except Exception as e:
            print(f"Erreur lors du traitement de l'échantillon {idx}: {e}")
            # Remplir les listes avec des valeurs nan pour cet échantillon
            current_length_list.append(np.nan)
            data_mean_list.append(np.nan)
            data_std_list.append(np.nan)
            data_max_list.append(np.nan)
            data_min_list.append(np.nan)

    # 3. Intégration des statistiques dans le DataFrame
    print("\n=== Intégration des Statistiques dans le DataFrame ===")
    dataframe['data_length'] = current_length_list
    dataframe['data_mean'] = data_mean_list
    dataframe['data_std'] = data_std_list
    dataframe['data_max'] = data_max_list
    dataframe['data_min'] = data_min_list

    # 4. Statistiques sur les longueurs des fichiers
    if total_lengths:
        average_length = sum(total_lengths) / len(total_lengths)
        print(f"\nLongueur moyenne des fichiers: {average_length:.2f} lignes de données")
        print(
            f"Longueur du fichier le plus long: {longest_file_length} lignes de données (Index: {longest_file_index})")
        print(f"Nombre total de fichiers: {len(total_lengths)}")

        # Afficher la distribution des longueurs
        print("\nDistribution des longueurs des fichiers:")
        sorted_lengths = sorted(length_distribution.items())
        for length, count in sorted_lengths:
            print(f"  {length} lignes: {count} fichier(s)")

        # Afficher les 5 longueurs les plus fréquentes
        sorted_by_count = sorted(length_distribution.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 des longueurs les plus fréquentes:")
        for length, count in sorted_by_count[:5]:
            print(f"  {length} lignes: {count} fichier(s)")
    else:
        print("Aucune donnée valide trouvée pour les analyses supplémentaires.")

    # 5. Statistiques des données brutes
    if all_data:
        all_data = np.vstack(all_data)
        print("\n=== Statistiques des Données Brutes ===")
        print(f"Shape totale des données: {all_data.shape}")
        print(f"Moyenne par feature: {np.mean(all_data, axis=0)}")
        print(f"Écart-type par feature: {np.std(all_data, axis=0)}")
    else:
        print("Aucune donnée brute valide pour l'analyse.")

    # 6. Visualisation des échantillons
    if sample_data:
        plot_sample_data(sample_data, sample_metadata)
    else:
        print("Aucun échantillon valide pour la visualisation.")

    plot_categorical_distribution(dataframe, categorical_cols)
    plot_numerical_distribution(dataframe, numerical_cols)
    plot_correlation_matrix(dataframe, numerical_cols)

    group_cols = categorical_cols
    stats_cols = ['data_length', 'data_mean', 'data_std', 'data_max', 'data_min']

    analyze_grouped_data(dataframe, group_cols, stats_cols)

    # 9. Visualiser les différences avec des box plots
    for group_col in group_cols:
        for stat_col in stats_cols:
            plot_grouped_data(dataframe, group_col, stat_col)


def analyze_categorical(dataframe, categorical_cols):
    """Analyse et affiche la distribution des variables catégorielles."""
    print("\n=== Distribution des Variables Catégorielles ===")
    for col in categorical_cols:
        counts = dataframe[col].value_counts()
        print(f"\n{col}:\n{counts}")


def analyze_numerical(dataframe, numerical_cols):
    """Affiche les statistiques descriptives des variables numériques."""
    print("\n=== Statistiques Descriptives des Variables Numériques ===")
    desc = dataframe[numerical_cols].describe()
    print(desc)


def analyze_correlation(dataframe, numerical_cols):
    """Calcule et affiche la matrice de corrélation des variables numériques."""
    print("\n=== Matrice de Corrélation ===")
    corr = dataframe[numerical_cols].corr()
    print(corr)


def plot_categorical_distribution(dataframe, categorical_cols):
    """Trace les distributions des variables catégorielles."""
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=dataframe, x=col)
        plt.title(f"Distribution de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_numerical_distribution(dataframe, numerical_cols):
    """Trace les distributions des variables numériques."""
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=dataframe, x=col, kde=True)
        plt.title(f"Distribution de {col}")
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(dataframe, numerical_cols):
    """Trace la heatmap de corrélation des variables numériques."""
    plt.figure(figsize=(12, 10))
    corr = dataframe[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap de Corrélation")
    plt.show()


def plot_sample_data(sample_data, sample_metadata):
    for i, (data, metadata) in enumerate(zip(sample_data, sample_metadata)):
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"Échantillon {i} - Metadata: {metadata}")
        plt.xlabel("Temps")
        plt.ylabel("Valeur")
        plt.tight_layout()
        plt.show()


def analyze_grouped_data(dataframe, group_cols, stats_cols):
    for group_col in group_cols:
        print(f"\n=== Analyse par {group_col} ===")
        grouped = dataframe.groupby(group_col)[stats_cols].agg(['mean', 'std', 'min', 'max'])
        print(grouped)

        # Visualisations
        for stat in ['mean', 'std']:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=grouped.index, y=grouped[stat]['data_mean'])
            plt.title(f"Moyenne de la donnée par {group_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.barplot(x=grouped.index, y=grouped[stat]['data_std'])
            plt.title(f"Écart-type de la donnée par {group_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def plot_grouped_data(dataframe, group_col, stat_col):
    plt.figure(figsize=(14, 7))
    sns.boxplot(x=group_col, y=stat_col, data=dataframe)
    plt.title(f"Distribution de {stat_col} par {group_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def normalize_numerical(dataframe, numerical_cols):
    scaler = StandardScaler()
    dataframe[numerical_cols] = scaler.fit_transform(dataframe[numerical_cols].fillna(0.0))
    return scaler


def split_dataset(dataframe, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    dataframe = pd.read_csv('../AUTOFILL_data/datav2/merged_cleaned_data.csv')

    dataframe_LES = dataframe[dataframe['technique'] == 'les'].reset_index(drop=True).sample(frac=0.5)

    print(f"Nombre total d'échantillons dans le dataframe original: {len(dataframe)}")
    print(f"Nombre d'échantillons après filtrage et échantillonnage: {len(dataframe_LES)}")

    dataset = CustomDatasetVAE(dataframe_LES, data_dir='../AUTOFILL_data/datav2/Base de donnée')

    print(f"Nombre d'échantillons dans le CustomDataset: {len(dataset)}")
    print("\nVocabulaire des variables catégorielles:")
    for col, vocab in dataset.cat_vocab.items():
        print(f"{col}: {vocab}")

    # Analyses et Visualisations Optimisées
    analyze_and_visualize(
        dataset=dataset,
        dataframe=dataframe_LES,
        categorical_cols=dataset.categorical_cols,
        numerical_cols=dataset.numerical_cols,
        pad_size=dataset.pad_size
    )



if __name__ == "__main__":
    main()
