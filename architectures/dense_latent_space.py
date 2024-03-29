import gymnasium
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class DenseLatentSpaceModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.size_layers = 64

        self.flatten_observation = None
        self.latent_space = None

        observation_size = get_preprocessor(observation_space)(observation_space).size
        action_size = get_preprocessor(action_space)(action_space).size

        num_hidden_layers = model_config.get('number_hidden_layers', 2)

        actor_layers = [nn.Linear(observation_size, self.size_layers), nn.ReLU()]
        critic_layers = [nn.Linear(observation_size, self.size_layers), nn.ReLU()]

        for _ in range(num_hidden_layers):
            actor_layers.append(nn.Linear(self.size_layers, self.size_layers))
            actor_layers.append(nn.ReLU())  # Adding activation function after each hidden layer

            critic_layers.append(nn.Linear(self.size_layers, self.size_layers))
            critic_layers.append(nn.ReLU())

        critic_layers.append(nn.Linear(self.size_layers, 1))

        self.actor_layers = nn.Sequential(*actor_layers)
        self.critic_layers = nn.Sequential(*critic_layers)

        self.action_layer = nn.Linear(self.size_layers, action_size)
        self.value_function_layer = nn.Linear(self.size_layers, 1)

    def forward(self, input_dict, state, seq_lens):
        self.flatten_observation = input_dict['obs_flat']
        self.latent_space = self.actor_layers(self.flatten_observation)

        action = self.action_layer(self.latent_space)
        return action, []

    def value_function(self):
        value_function = self.critic_layers(self.flatten_observation)
        return torch.reshape(value_function, [-1])

    def get_latent_space(self, observations):
        with torch.no_grad():
            input_dict = {'obs': observations, 'obs_flat': observations}
            self.forward(input_dict, None, None)
            self.value_function()
            return self.latent_space
