from ray.rllib.models import ModelCatalog

from architectures.dense_latent_space import DenseLatentSpaceModel
from architectures.minimal_model import MinimalModel
from architectures.minimal_latent_space import MinimalLatentSpaceModel


def register_architectures():
    ModelCatalog.register_custom_model('minimal_model', MinimalModel)
    ModelCatalog.register_custom_model('minimal_latent_space_model', MinimalLatentSpaceModel)
    ModelCatalog.register_custom_model('dense_latent_space', DenseLatentSpaceModel)
