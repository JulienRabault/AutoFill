from model.VAE.submodel.VAE import VAE
from model.VAE.submodel.ResVAE import ResVAE

MODEL_REGISTRY = {
    "VAE": VAE,
    "ResVAE": ResVAE,
}
