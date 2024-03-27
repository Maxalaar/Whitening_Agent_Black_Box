from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class DenseLatentSpaceModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.latent_space_size = 10

        self.input = None
        self.latent_space = None

        observation_size = get_preprocessor(observation_space)(observation_space).size
        action_size = get_preprocessor(action_space)(action_space).size

        num_hidden_layers = model_config.get('number_hidden_layers', 1)

        layers = [nn.Linear(observation_size, self.latent_space_size), nn.ReLU()]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(self.latent_space_size, self.latent_space_size))
            layers.append(nn.ReLU())  # Adding activation function after each hidden layer

        self.latent_space_layer = nn.Sequential(*layers)

        self.action_layer = nn.Linear(self.latent_space_size, action_size)
        self.value_function_layer = nn.Linear(self.latent_space_size, 1)

    def forward(self, input_dict, state, seq_lens):
        self.input = input_dict['obs']
        self.latent_space = self.latent_space_layer(self.input)

        action = self.action_layer(self.latent_space)
        return action, []

    def value_function(self):
        value_function = self.value_function_layer(self.latent_space)
        return torch.reshape(value_function, [-1])

    def get_latent_space(self, observations):
        with torch.no_grad():
            input_dict = {'obs': observations}
            self.forward(input_dict, None, None)
            self.value_function()
            return self.latent_space
