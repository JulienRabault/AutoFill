#python3 srun.py --mode vae --config "model/VAE/vae_config_les.yaml" --mat ag --tech les
#python3 srun.py --mode vae --config "model/VAE/vae_config_saxs.yaml" --mat ag --tech saxs
#python3 srun.py --mode pair_vae --config "model/pair_vae2.yaml" --mat ag 
import argparse
import yaml 

#from model.VAE.trainer import train as train_vae
from model.VAE.grid_trainer import grid_search as train_vae
#from model.trainer import train as train_pairvae
from model.grid_pair_trainer import grid_search as train_pairvae

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un model")
    # Chemins et filtres dataset
    parser.add_argument("--mode", type=str, default="vae",choices = ["vae","pair_vae"])
    parser.add_argument("--config", type=str, default="model/VAE/vae_config_saxs.yaml",)
    parser.add_argument("--name", type=str, default="training", )
    parser.add_argument("--hdf5_file", type=str, default="/projects/pnria/julien/autofill/all_data.h5",
                        help="Chemin vers le H5 utilisé")
    parser.add_argument("--conversion_dict_path", type=str, default="conversion_dict_saxs.json",
                        help="Chemin vers le dictionnaire de conversion des métadonnées")
    parser.add_argument("--technique", type=str, default="saxs",
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default="ag",
                        help="Filtre pour la colonne 'material'")
    parser.add_argument("--devices")
    # Paramètres d'entraînement
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['name'] = args.name
    if args.material != "None" :
        config['dataset']["metadata_filters"]['material'] = args.material.split(",")
    if args.technique != "None" and args.mode != "pair_vae":
        config['dataset']["metadata_filters"]['technique'] = args.technique.split(",")
    config['devices'] = args.devices
    print(config)

    if args.mode == "pair_vae" :
        train_pairvae(config)
    elif args.mode == "vae" :
        train_vae(config)
    else :
        raise("Mode not existing")

if __name__ == "__main__":
    main()

    