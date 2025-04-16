from model.VAE.submodel.VAE import VAE
from model.VAE.submodel.ResVAEBN import ResVAEBN
from model.VAE.submodel.ResVAE2 import ResVAE2
from model.VAE.submodel.ResVAE import ResVAE

MODEL_REGISTRY = {
    "VAE": VAE,
    "ResVAE": ResVAE,
    "ResVAE2": ResVAE2,
    "ResVAEBN": ResVAEBN,
}
