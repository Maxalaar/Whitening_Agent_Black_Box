import gymnasium
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class DenseLatentSpaceModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        model_configuration = model_config['custom_model_config']
        self.configuration_hidden_layers = model_configuration.get('configuration_hidden_layers', [32, 32])
        self.num_hidden_layers = len(self.configuration_hidden_layers)
        self.layers_use_clustering = model_configuration.get('layers_use_clustering', [True for _ in range(self.num_hidden_layers)])

        self.flatten_observation = None
        self.latent_space = None

        observation_size = get_preprocessor(observation_space)(observation_space).size
        action_size = get_preprocessor(action_space)(action_space).size

        actor_layers = [nn.Linear(observation_size, self.configuration_hidden_layers[0]), nn.ReLU()]
        critic_layers = [nn.Linear(observation_size, self.configuration_hidden_layers[0]), nn.ReLU()]

        for i in range(0, self.num_hidden_layers - 1):
            actor_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i+1]))
            actor_layers.append(nn.ReLU())

            critic_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i+1]))
            critic_layers.append(nn.ReLU())

        critic_layers.append(nn.Linear(self.configuration_hidden_layers[-1], 1))

        self.actor_layers = nn.Sequential(*actor_layers)
        self.critic_layers = nn.Sequential(*critic_layers)

        self.action_layer = nn.Linear(self.configuration_hidden_layers[-1], action_size)

        self.hook_current_index_layer = None
        self.hook_activations = None
        self.already_initialized_get_latent_space = False

    def forward(self, input_dict, state, seq_lens):
        self.flatten_observation = input_dict['obs_flat']
        self.latent_space = self.actor_layers(self.flatten_observation)

        action = self.action_layer(self.latent_space)
        return action, []

    def value_function(self):
        value_function = self.critic_layers(self.flatten_observation)
        return torch.reshape(value_function, [-1])

    def hook(self, module, input, output):
        if not isinstance(module, nn.Linear):
            self.hook_activations[self.hook_current_index_layer] = output
            self.hook_current_index_layer += 1

    def initialisation_get_latent_space(self):
        index_layer = 0
        for layer in self.actor_layers:
            if not isinstance(layer, nn.Linear):
                if self.layers_use_clustering[index_layer]:
                    layer.register_forward_hook(self.hook)
                index_layer += 1
        self.already_initialized_get_latent_space = True

    def get_latent_space(self, observations):
        if not self.already_initialized_get_latent_space:
            self.initialisation_get_latent_space()

        with torch.no_grad():
            self.hook_current_index_layer = 0
            self.hook_activations = {}
            input_dict = {'obs': observations, 'obs_flat': observations}
            self.forward(input_dict, None, None)
            self.value_function()
            return torch.cat(list(self.hook_activations.values()), dim=1)
