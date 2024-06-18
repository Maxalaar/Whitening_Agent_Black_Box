import os
import ray

import environments.environment_configurations as environment_configurations
from pipeline.generation_clustered_episode_videos import generation_clustered_episode_videos
from utilities.global_include import project_initialisation
from pipeline.generate_episode_videos import generate_video_episodes
from pipeline.generation_cluster_videos import generation_cluster_videos
from pipeline.generate_latent_space_dataset import generate_latent_space_dataset
from pipeline.generate_observation_dataset import generate_observation_dataset
from pipeline.training_agent import training_agent
from pipeline.latent_space_clustering import latent_space_clustering


if __name__ == '__main__':
    experiment_name = 'entire_system_v3'
    rllib_trial_name = 'rllib_base_trial'
    environment_name = 'PongSurvivor'     # 'CartPole-v1'

    environment_configuration = environment_configurations.classic_one_ball
    training_time = 60 * 60 * 5
    architecture_name = 'dense_latent_space'      # 'minimal_latent_space_model', 'dense_latent_space'
    architecture_configuration = {
        'configuration_hidden_layers': [32, 32],
        'layers_use_clustering': [True, True],
    }

    execution_directory = os.getcwd()
    results_directory = os.path.join(execution_directory, 'results')
    experiment_directory = os.path.join(results_directory, experiment_name)
    rllib_directory = os.path.join(experiment_directory, 'rllib')
    videos_directory = os.path.join(experiment_directory, 'videos')
    episode_videos_directory = os.path.join(videos_directory, 'episodes')
    cluster_videos_directory = os.path.join(videos_directory, 'clusters')
    clustered_episode_videos_directory = os.path.join(videos_directory, 'clustered_episode')
    datasets_directory = os.path.join(experiment_directory, 'datasets')
    classifiers_directory = os.path.join(experiment_directory, 'sklearn')
    visualization_directory = os.path.join(experiment_directory, 'visualization')
    rllib_trial_path = os.path.join(rllib_directory, rllib_trial_name)

    ray.shutdown()
    ray.init(local_mode=False)
    project_initialisation()

    training_agent(
        rllib_directory=rllib_directory,
        rllib_trial_name=rllib_trial_name,
        environment_name=environment_name,
        environment_configration=environment_configuration,
        architecture_name=architecture_name,
        architecture_configuration=architecture_configuration,
        training_time=training_time,
    )

    generate_video_episodes(
        video_directory=episode_videos_directory,
        rllib_trial_path=rllib_trial_path,
        number_video_per_worker=3,
    )

    generate_observation_dataset(
        datasets_directory=datasets_directory,
        rllib_trial_path=rllib_trial_path,
        number_iteration=30,
        number_episode_per_worker=2,
    )

    generate_latent_space_dataset(
        datasets_directory=datasets_directory,
        rllib_trial_path=rllib_trial_path,
    )

    latent_space_clustering(
        datasets_directory=datasets_directory,
        sklearn_directory=classifiers_directory,
    )

    generation_cluster_videos(
        video_directory=cluster_videos_directory,
        datasets_directory=datasets_directory,
        classifiers_directory=classifiers_directory,
        number_episode=50,
    )

    generation_clustered_episode_videos(
        video_directory=clustered_episode_videos_directory,
        datasets_directory=datasets_directory,
        classifiers_directory=classifiers_directory,
        number_episode=50,
    )

    ray.shutdown()


