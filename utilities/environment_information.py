import ray
import sys
import os
import torch


def environment_information():
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        print('Conda environment name:', conda_env)
    else:
        print('You are not in a Conda environment.')
    print('Python version:', sys.version)
    print('Ray version:', ray.__version__)
    print('PyTorch version:', torch.__version__)
    print()

    cuda_available = torch.cuda.is_available()
    print(f'Is CUDA available? {cuda_available}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device in use: {device}')
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f'Number of available GPUs: {gpu_count}')

        current_gpu = torch.cuda.current_device()
        print(f'Current GPU ID: {current_gpu}')

        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f'GPU Name: {gpu_name}')
    print()

    resources = ray.cluster_resources()
    print('Number of CPUs accessible to Ray:', resources.get('CPU', 0))
    print('Number of GPUs accessible to Ray:', resources.get('GPU', 0))
