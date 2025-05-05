# Installation

$python3 -m virtualenv venv
$source venv/bin/activate
$pip install -r requirements.txt

# Construction of dataset

## For Single VAE

Requirements : 
- csv with path to txt file, and metadata
- folder with txt files
  
Run notebook 'txt2h5.ipynb'. Adjust path as necessary.

## For Pair VAE

Requirements : 
- csv with saxs path and les path to txt file, and metadata
- folder with txt files
  
Run notebook 'txt2pairh5.ipynb'. Adjust path as necessary.


# Training models

## For Single VAE

### Configuration

You can modify yaml configurations in model/VAE as you like.
Each section of the YAML file is explained below.

```yaml
dataset:
  hdf5_file: Path to the HDF5 file containing the data.
  metadata_filters: Filters applied to the metadata to select a specific subset of the data.
    material: ["ag"]
    technique: ["saxs"]
    type: ["simulation"]
  requested_metadata: ["shape", "material"] Not use for the moment but for inference plotting
  conversion_dict_path: Path to the metadata conversion dictionary.
  sample_frac: Fraction of data to be used (between 0 and 1).
  transform: Transformations applied to the data, such as normalization and padding.
    y:
      MinMaxNormalizer: {} 
      PaddingTransformer:
        pad_size: 80
        value: 0
    q:
      PaddingTransformer:
        pad_size: 80
        value: 0
        
training:
  num_gpus: Number of GPUs used for training.
  num_nodes: Number of nodes used.
  num_epochs: Total number of training epochs.
  patience: Number of epochs without improvement before early stopping.
  batch_size: Batch size used for training.
  num_workers: Number of parallel workers for data loading.
  beta: Coefficient for the KLD loss.
  max_lr:  Maximum learning rate.
  T_max: Number of steps before resetting the learning rate (CosineAnnealingLR).
  eta_min: Minimum learning rate.

model:
  vae_class: Type of VAE between [VAE, ResVAE]
  output_transform_log: boolean, wheter to transform input and output to log before MSE (used for SAXS)
  args: Model architecture parameters:
    in_channels: Number of input channels.
    output_channels: Number of output channels.
    dilation: Dilation factor for convolutions.
    input_dim: Input data dimension.
    latent_dim: Latent space dimension.
    down_channels: Channels used in the encoder (downsampling in the network).
    up_channels: Channels used in the decoder (upsampling in the network).
    strat: Data strategy (here based on y only) -no other strat available for the moment.  
```
    
### Launching

- If you have OCCIDATA, you can : 
$python3 srun.py --mode vae --config "model/VAE/vae_config_les.yaml" --mat ag --tech les
$python3 srun.py --mode vae --config "model/VAE/vae_config_saxs.yaml" --mat ag --tech saxs

- if you don't :
$train.py --name training_vae --mode vae --config {config} --technique {tech} --material {mat} --devices {gpu}

where config if the path to the yaml config, technique is saxs or les ...

- you can also simply use the train function in model/VAE/trainer.py with the correct config.
  
  
## For Pair VAE





# Visualize results

## Tensorboard

Each trained model will create a subfolder in lightning_logs.
Inside, the model configuration, weights, and the metrics board are stored.
You can display the latter with:
$tensorboard --logdir lightning_logs

## Loading model and testing

## For Single VAE

You can use 'plot_result_vae.ipynb' notebook. Modify path_config and path_checkpoints with your trained model.

## For Pair VAE

You can use 'plot_result_pairvae.ipynb' notebook. Modify path_config and path_checkpoints with your trained model.

