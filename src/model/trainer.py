import lightning.pytorch as pl
import numpy as np
import torch
import yaml
import re
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.model.callbacks.inference_callback import InferencePlotCallback
from src.model.callbacks.metrics_callback import MAEMetricCallback
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE

class TrainPipeline:
    def __init__(self, config: dict):
        print("[Pipeline] Loading configuration")
        self.config = config
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False, allow_unicode=True))
        print("[Pipeline] Building components")
        self.log_path = self._safe_log_directory()
        self.model, self.dataset, self.extra_callback_list = self._initialize_components()
        print("[Pipeline] Preparing data loaders")
        self.training_loader, self.validation_loader = self._create_data_loaders()
        print("[Pipeline] Building trainer")
        self.trainer = self._configure_trainer()
        print("[Pipeline] Preparing log directory")
        self.log_directory = self._setup_log_directory()

    def _safe_log_directory(self) -> Path:
        base_dir = Path(self.config['experiment_name'])
        run_name = self.config['run_name']

        def extract_index(name: str) -> tuple[str, int]:
            match = re.match(rf"^(.*?)(?:_(\d+))?$", name)
            if match:
                base, idx = match.groups()
                return base, int(idx) if idx else 0
            return name, 0

        base_name, _ = extract_index(run_name)
        existing = [p.name for p in base_dir.glob(f"{base_name}*") if p.is_dir()]

        max_index = 0
        for name in existing:
            match = re.match(rf"^{re.escape(base_name)}(?:_(\d+))?$", name)
            if match and match.group(1):
                max_index = max(max_index, int(match.group(1)))

        next_name = f"{base_name}_{max_index + 1}" if max_index > 0 or run_name == base_name else run_name
        log_path = base_dir / next_name
        log_path.mkdir(parents=True)
        return log_path

    def _initialize_components(self):
        model_type = self.config['model']['type']
        common_cfg = {
            'num_samples': self.config['training'].get('num_samples', 4),
            'every_n_epochs': self.config['training'].get('every_n_epochs', 10),
            'artifact_file': 'val_plot.png'
        }
        callbacks = []
        if model_type.lower() == 'pair_vae':
            model = PlPairVAE(self.config)
            dataset = PairHDF5Dataset(**self.config['dataset'])
            curves_config = {
                'saxs': {'truth_key': 'data_y_saxs', 'pred_keys': ['recon_saxs', 'recon_les2saxs'], 'use_loglog': True},
                'les': {'truth_key': 'data_y_les', 'pred_keys': ['recon_les', 'recon_saxs2les']}
            }
        elif model_type.lower() == 'vae':
            model = PlVAE(self.config)
            dataset = HDF5Dataset(**self.config['dataset'])
            curves_config = {'recon': {'truth_key': 'data_y', 'pred_keys': ["recon"],
                                       'use_loglog': self.config['training']['use_loglog']}}
            callbacks.append(MAEMetricCallback())
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'pair' or 'vae'.")

        inf_cb = InferencePlotCallback(
            curves_config=curves_config,
            output_dir=self.config['training'].get('output_dir', 'inference_results'),
            **common_cfg
        )
        callbacks.append(inf_cb)
        if self.config['training'].get('plot_train', False):
            train_cfg = common_cfg.copy()
            train_cfg['artifact_file'] = 'train_plot.png'
            callbacks.insert(0, InferencePlotCallback(curves_config=curves_config,
                                                      output_dir=self.log_path / "inference_results",
                                                      **train_cfg))

        return model, dataset, callbacks

    def _create_data_loaders(self):
        dataset_instance = self.dataset
        total_samples = len(dataset_instance)
        train_count = int(0.8 * total_samples)
        validation_count = total_samples - train_count
        print(f"[Data] Splitting dataset: train={train_count}, validation={validation_count}")
        train_subset, validation_subset = random_split(dataset_instance, [train_count, validation_count])
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        print(f"[Data] Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
        training_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return training_loader, validation_loader

    def _configure_trainer(self):
        print("[Trainer] Configuring callbacks and logger")
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=self.config['training']['patience'],
                                            verbose=True, mode='min')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                              every_n_epochs=self.config['training'].get('save_every', 1),
                                              dirpath=self.log_path,
                                              filename="best")
        if "mlflow_uri" in self.config:
            from dotenv import load_dotenv
            load_dotenv()

        self.logger = MLFlowLogger(
            experiment_name=self.config['experiment_name'],
            run_name=self.config['run_name'],
            tracking_uri=self.config.get("mlflow_uri", f"file:{self.config['experiment_name']}/mlrun")
        )
        self.logger.log_hyperparams(self.config)
        self.all_callbacks = [early_stop_callback, checkpoint_callback] + self.extra_callback_list
        strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        devices = self.config['training']['num_gpus'] if torch.cuda.is_available() else 1
        return pl.Trainer(
            strategy=strategy,
            accelerator=accelerator,
            devices=devices,
            num_nodes=self.config['training']['num_nodes'],
            max_epochs=self.config['training']['num_epochs'],
            log_every_n_steps=10,
            callbacks=self.all_callbacks,
            logger=self.logger
        )

    def _setup_log_directory(self) -> str:

        file_path = self.log_path / "config_model.yaml"
        with file_path.open("w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

        print(f"Fichier YAML sauvegardé dans : {file_path}")

        self.trainer.logger.experiment.log_artifact(
            local_path=str(file_path),
            run_id=self.trainer.logger.run_id
        )

        np.save(self.log_path / 'train_indices.npy', self.training_loader.dataset.indices)
        np.save(self.log_path / 'val_indices.npy', self.validation_loader.dataset.indices)
        print(f"Indices sauvegardés dans : {self.log_path / 'train_indices.npy'} et {self.log_path / 'val_indices.npy'}")
        self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'train_indices.npy'),run_id=self.trainer.logger.run_id)
        self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'val_indices.npy'),run_id=self.trainer.logger.run_id)


    def train(self):
        print("[Pipeline] Starting training")
        self.trainer.fit(
            self.model,
            train_dataloaders=self.training_loader,
            val_dataloaders=self.validation_loader
        )
        print("[Pipeline] Training completed")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configuration = yaml.safe_load(f)
    pipeline = TrainPipeline(configuration)
    pipeline.run()
