from src.model.vae.submodel.ResVAE import ResVAE
from src.model.vae.submodel.ResVAEBN import ResVAEBN
from src.model.vae.submodel.VAE import VAE

MODEL_REGISTRY = {
    "VAE": VAE,
    "ResVAE": ResVAE,
    "ResVAEBN": ResVAEBN,
}
