import argparse

import torch

from src.model.inferencer import VAEInfer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Chemin vers le fichier HDF5 ou CSV")
    parser.add_argument("--conversion_dict_path", type=str, default=None,
                        help="Chemin vers le dictionnaire de conversion (HDF5 uniquement)")
    parser.add_argument("--technique", type=str, nargs='+', default=["saxs"],
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, nargs='+', default=["ag"],
                        help="Filtre pour la colonne 'material'")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ckpt = args.checkpoint
    hparams = torch.load(ckpt, map_location='cpu')['hyper_parameters']
    model_type = hparams['model']['type']
    if model_type != 'VAE':
        raise ValueError("Ce script ne supporte que les modèles VAE.")
    runner = VAEInfer(
        checkpoint_path=ckpt,
        data_path=args.data_path,
        technique=args.technique,
        material=args.material
    )
    runner.run()