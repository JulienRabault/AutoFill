from model.VAE.submodel.VAE import VAE
from model.VAE.submodel.ResVAE import ResVAE
from model.VAE.submodel.ResVAE2 import ResVAE2

MODEL_REGISTRY = {
    "VAE": VAE,
    "ResVAE": ResVAE,
    "ResVAE2": ResVAE2,
}
