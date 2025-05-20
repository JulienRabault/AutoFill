import argparse

import torch

from src.model.inferencer import PairVAEInferencer, VAEInferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-d","--data_path", type=str, required=True,
                        help="Chemin vers le fichier HDF5 ou CSV")
    parser.add_argument("-cdp","--conversion_dict_path", type=str, default=None,
                        help="Chemin vers le dictionnaire de conversion (HDF5 uniquement)")
    parser.add_argument('--mode', choices=['les_to_saxs', 'saxs_to_les'], required=False, default=None,
                        help='Mode de convertion pour le PairVAE')
    parser.add_argument("-bs", "--batch_size", type=int, default=32,)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = args.checkpoint
    hparams = torch.load(ckpt, map_location='cpu')['hyper_parameters']['config']
    model_type = hparams['model']['type']
    if model_type.lower() not in ['vae', 'pair_vae']:
        raise ValueError(f"Model type {model_type} is not supported for inference.")
    if model_type.lower() == 'pair_vae' and args.mode is None:
        raise ValueError("Please provide the mode for the PairVAE model.")
    if args.conversion_dict_path is None and args.data_path.endswith('.h5'):
        raise ValueError("Please provide the conversion dictionary path for HDF5 files.")
    print(f"Loading model of type {model_type} from checkpoint: {args.checkpoint}, hparams: {hparams}")
    if model_type == 'vae':
        runner = VAEInferencer(args.checkpoint, args.data_path, hparams, args.conversion_dict_path, args.batch_size)
    else:
        runner = PairVAEInferencer(args.checkpoint, args.data_path, hparams, args.conversion_dict_path, args.batch_size)
    runner.infer()
