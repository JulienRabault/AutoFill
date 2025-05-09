#python3 srun.py --mode vae  --config "model/VAE/vae_config_les.yaml" --mat ag --tech les
#python3 srun.py --mode vae --config "model/VAE/vae_config_saxs.yaml" --mat ag --tech saxs
#python3 srun.py --mode pair_vae --config "model/pair_vae2.yaml" --mat ag 
import argparse
import yaml
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.trainer import TrainPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un model")
    parser.add_argument("--mode", type=str, default="vae",choices = ["vae","pair_vae"])
    parser.add_argument("--gridsearch", action='store_true', default=False)
    parser.add_argument("--config", type=str, default="model/VAE/vae_config_saxs.yaml",)
    parser.add_argument("--name", type=str, default=None, )
    parser.add_argument("--hdf5_file", type=str,
                        help="Chemin vers le H5 utilisé, par default celui de la config", default=None)
    parser.add_argument("--conversion_dict_path", type=str,
                        help="Chemin vers le dictionnaire de conversion des métadonnées, par default celui de la config", default=None)
    parser.add_argument("--technique", type=str, default=None,
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default=None,
                        help="Filtre pour la colonne 'material'")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    if args.name is not None:
        config['run_name'] = args.name
    if args.material is not  None :
        config['dataset']["metadata_filters"]['material'] = args.material.split(",")
    if args.technique is not None and args.mode is not "pair_vae":
        config['dataset']["metadata_filters"]['technique'] = args.technique.split(",")
    config['model']['type'] = args.mode

    if args.hdf5_file is not None:
        config['dataset']['hdf5_file'] = args.hdf5_file
    if args.conversion_dict_path is not None:
        config['dataset']['conversion_dict_path'] = args.conversion_dict_path

    if args.gridsearch:
        raise("TODO : gridsearch")
    else:
        trainer = TrainPipeline(config)
        trainer.train()




if __name__ == "__main__":
    main()

    