# Toolbox VAE/PairVAE

# TODO :
- [ ] Faire l'inference => utiliser les modeles
- [ ] Expliquer la sorti des train, poids + analyse de courbe 
- [ ] Bouger les fichiers/dossier
- [ ] Renforcer les explications sur le fonctionnement du pairVAE ?
- [ ] Essayer de faire les lien dans le "sommaire"

### Auteur : 
- **Julien Rabault** (julien.rabault@irit.fr)
- **Caroline De Pourtales** (caroline.de-pourtales@irit.fr)

## Structure du projet

```
AutoFill/
├─ dataset/             # code de gestion des données
├─ model/               # architectures et utilitaires d'entraînement
├─ scripts/             # pipeline CLI : prétraitement, conversion, entraînement
├─ config_vae.yml       # template config pour VAE (modifiable)
├─ config_pairvae.yml   # template config pour PairVAE (modifiable)
├─ requirements.txt     # dépendances
└─ README.md            # guide d’utilisation
```

## Installation

1. **Pré-requis**:

   * Python3.8+

2. **Clone du projet**:

   ```bash
   git clone https://github.com/JulienRabault/AutoFill.git
   cd AutoFill
   ```

3. **Environnement et dépendances**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   Si GPU disponible, installez PyTorch avec CUDA : consultez [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Pipeline

1. **[Prétraitement CSV](#1-prétraitement-csv)** : fusion et nettoyage → metadata_clean.csv
2. **Conversion .txt → HDF5** : séries temporelles + métadonnées → all_data.h5 + metadata_dict.json
3. **Entraînement du modèle** : filtre, configuration YAML → lancement du training (VAE ou PairVAE)
4. **Inference (optionnelle)** : analyse des résultats à partir des poids entraînés** : fusion et nettoyage → metadata_clean.csv
5. **Conversion .txt → HDF5 pour PAIRVAE** : séries temporelles + métadonnées → all_data.h5 + metadata_dict.json
6. **Entraînement du modèle PAIRVAE** : filtre, configuration YAML → lancement du training

### 1. Prétraitement CSV
`csv_pre_process.py`

Ce script fusionne et nettoie plusieurs fichiers CSV de métadonnées.

**Arguments:**

- `<inputs>` : un ou plusieurs chemins vers des fichiers CSV (séparateur ;).
- `<output>` : chemin du fichier CSV nettoyé de sortie (séparateur ,).

```bash
python scripts/csv_pre_process.py \
  data/raw_csv/file1.csv data/raw_csv/file2.csv \
  data/metadata_clean.csv
```

> **Exemple** : après exécution, le fichier `data/metadata_clean.csv` contient toutes les métadonnées normalisées. Vous pourrez l’utiliser à l’étape suivante pour la conversion au format HDF5.

### 2. Conversion `.txt` → HDF5 avec `txtTOhdf5.py`

Objectif : convertir les séries temporelles (`.txt`) et le CSV de métadonnées en un unique fichier HDF5.

Arguments:
* `--data_csv_path` : chemin vers le fichier CSV de métadonnées (doit contenir au moins une colonne path vers les fichiers .txt).
* `--data_dir` : dossier racine contenant les fichiers .txt.
* `--final_output_file` : chemin de sortie pour le fichier .h5 généré.
* `--json_output` : chemin de sortie pour le dictionnaire de conversion des métadonnées catégorielles (au format JSON).
* `--pad_size` : longueur maximale des séries temporelles (padding ou troncature appliqué si nécessaire). Default : 500.

```bash
python scripts/txtTOhdf5.py \
  --data_csv_path data/metadata_clean.csv \
  --data_dir data/txt/ \
  --final_output_file data/all_data.h5 \
  --json_output data/metadata_dict.json \
  --pad_size 900
```

> **Exemple** : en sortie, `data/all_data.h5` contient `data_q`, `data_y`, `len`, `csv_index` et toutes les métadonnées, et `data/metadata_dict.json` recense les encodages catégoriels. Vous utiliserez ces deux fichiers pour l’entraînement.

**Structure HDF5 générée :**

```text
final_output.h5
├── data_q          [N, pad_size]
├── data_y          [N, pad_size]
├── len             [N]
├── csv_index       [N]
├── metadata_field1 [N]
├── metadata_field2 [N]
...
```

**Attention aux chemins (path) dans le CSV :**

Les chemins indiqués dans la colonne path du CSV doivent être relatifs au répertoire --data_dir. Le script les concatène pour localiser les fichiers .txt. Toute incohérence entraînera des erreurs ou des fichiers ignorés.
Avant de lancer la conversion, vous pouvez utiliser saminitycheck.py pour valider que tous les fichiers .txt référencés dans le CSV existent réellement dans le répertoire --data_dir.

**Exécutez le script comme suit :**

```bash
python scripts/saminitycheck.py \
  --csv data/metadata_clean.csv \
  --basedir data/txt/
```

Ce script vérifiera que chaque chemin dans la colonne path (colonne contenant `"path"` dans son nom) du CSV correspond à un fichier existant dans le répertoire --basedir. Si des fichiers manquent, ils seront listés.


### 3. Entraînement du modèle à partir du fichier HDF5 `train.py`

Une fois le HDF5 et le JSON générés, lancez l’entraînement :

```bash
python scripts/train.py \
  --mode vae \
  --gridsearch off \
  --config model/VAE/config_vae.yml \
  --name AUTOFILL_SAXS_VAE \
  --hdf5_file data/all_data.h5 \
  --conversion_dict_path data/metadata_dict.json \
  --technique saxs \
  --material ag
```

> **Exemple** : ici `data/all_data.h5` et `data/metadata_dict.json` sont issus de l’étape précédente, et seront filtrés sur `technique=saxs` et `material=ag`.

### Paramètres minimum modifiables dans le YAML (config_vae.yml)

* `experiment_name` : nom de l’expérience (création de sous-dossier dans logdir).
* `logdir` : dossier où seront stockés logs et checkpoints.
* `dataset`

  * `hdf5_file` : chemin vers votre fichier .h5.
  * `conversion_dict_path` : chemin vers le JSON de mapping.
  * `metadata_filters` : filtres à appliquer sur les métadonnées (ex. material, technique, type, shape).
  * `sample_frac` : fraction d’échantillonnage (entre 0.0 et 1.0).
* **`transform`**

  * `q` et `y` – `PaddingTransformer.pad_size` doit correspondre à `pad_size` utilisé lors de la conversion `.txt`.
* **`training`**

  * `num_epochs` : nombre d’époques maximales.
  * `batch_size` : taille de batch.
  * `num_workers` : nombre de workers DataLoader.
  * `max_lr`, `T_max`, `eta_min` : planning de taux d’apprentissage.
  * `beta` : coefficient β du VAE.
* **model.args**

  * `latent_dim` : dimension latente.
  * `input_dim` : doit être égal à pad_size.
  * `down_channels`/`up_channels` : architecture, à conserver sauf si confortable avec PyTorch.

> **Note :** en dehors de ces clés, tout autre paramètre dans le YAML n’est pas nécessairement safe à modifier si vous débutez en IA. Respectez surtout la cohérence pad_size / input_dim et les chemins d’accès pour éviter les erreurs.

### 4. Inference (optionnelle)
...

### 5. Entraînement du modèle PAIRVAE à partir du fichier HDF5 `train.py`

De la même manière que pour le VAE, vous pouvez convertir vos séries temporelles en un fichier HDF5 pour l’entraînement du PairVAE. Le script `pairtxtTOhdf5.py` est conçu pour cela.

Arguments :

```bash
python scripts/pairtxtTOhdf5.py \
  --data_csv_path   data/metadata_clean.csv  \
  --data_dir        data/txt/               \
  --pad_size        900                     \
  --final_output_file data/all_pair_data.h5 \
  --json_output     data/pair_metadata_dict.json
```

> Les chemins dans `saxs_path` et `les_path` doivent être relatifs à `--data_dir`. Vous pouvez contrôler l’existence de chaque paire de fichiers avec `scripts/saminitycheck.py` au besoin.

Structure du HDF5 généré (final_output.h5) :

```text
final_output.h5
├── data_q_saxs    [N, pad_size]
├── data_y_saxs    [N, pad_size]
├── data_q_les     [N, pad_size]
├── data_y_les     [N, pad_size]
├── len            [N]
├── valid          [N]
├── csv_index      [N]
├── <metadata_1>   [N]
├── <metadata_2>   [N]
└── ...            ...
```
Une fois la conversion terminée, vous obtenez :

- `data/all_pair_data.h5` prêt pour l’entraînement ;
- `data/pair_metadata_dict.json` contenant vos mappings catégoriels.

### 6. Entraînement du modèle PAIRVAE

L’entraînement du PairVAE se fait de la même manière que pour le VAE, mais avec un fichier HDF5 différent et une configuration YAML différente.

Lancez l’entraînement en mode pairvae avec votre template `config_pairvae.yml` :

```bash
python scripts/train.py \
  --mode pairvae \
  --gridsearch off \
  --config model/PairVAE/config_pairvae.yml \
  --name AUTOFILL_SAXS_PAIRVAE \
  --hdf5_file data/all_data.h5 \
  --conversion_dict_path data/metadata_dict.json \
  --technique saxs \
  --material ag
```

### Contact

Pour toute question ou problème, n’hésitez pas à contacter :
- **Julien Rabault** (julien.rabault@irit.fr)
- **Caroline De Pourtales** (caroline.de-pourtales@irit.fr)
