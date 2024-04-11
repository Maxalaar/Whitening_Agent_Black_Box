import os
from typing import Dict, List, Union
import dask.array as da
import h5py
import numpy as np

from utilities.global_include import create_directory


class DatasetHandler:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.dataset_path = self.path + '/' + self.name + '.h5'

    def save(self, data: Dict):
        create_directory(self.path)
        with h5py.File(self.dataset_path, 'a') as hf:
            for key_data in data.keys():
                if key_data not in hf:
                    da.to_hdf5(self.dataset_path, str(key_data), data[key_data], maxshape=(None, *data[key_data].shape[1:]))
                else:
                    dataset = hf[key_data]
                    dataset.resize((dataset.shape[0] + data[key_data].shape[0]), axis=0)
                    dataset[-data[key_data].shape[0]:] = data[key_data]

    def load(self, keys: List[str], number_data: Union[int, None] = None):
        dataset = h5py.File(self.dataset_path)

        if number_data is None:
            data = {}
            for key in keys:
                data[key] = da.from_array(dataset[key])
            return data
        else:
            total_values = dataset[keys[0]].shape[0]
            random_indices = np.random.choice(total_values, number_data)    # , replace=False
            random_indices.sort()
            random_indices = np.unique(random_indices)

            data = {}
            for key in keys:
                data[key] = da.from_array(dataset[key][random_indices])
            return data

    def load_episode(self, keys: List[str], number_episode):
        dataset = h5py.File(self.dataset_path)
        total_values = dataset['index_episodes'].shape[0]
        random_indices = np.random.choice(total_values, number_episode, replace=False)
        random_indices.sort()

        data = {}
        for key in keys:
            data[key] = []

        for index in random_indices:
            start_index = dataset['index_episodes'][index][0]
            end_index = dataset['index_episodes'][index][1]
            dataset[keys].append(dataset['index_episodes'][start_index:end_index])

        return dataset

    def print_info(self):
        dataset = h5py.File(self.dataset_path)
        print('dataset name: ' + self.name)

        print('dataset subgroups:')
        for subgroups_name in dataset:
            print(' ' + subgroups_name + ':')
            print('  number data: ' + str(dataset[subgroups_name].shape[0]))
            print('  shape data: ' + str(dataset[subgroups_name].shape[1:]))

    def size(self, key: str):
        if os.path.exists(self.dataset_path):
            dataset = h5py.File(self.dataset_path)
            return dataset[key].shape[0]
        else:
            return 0
