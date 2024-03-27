from typing import Dict, List, Union
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
                    hf.create_dataset(key_data, data=data[key_data], chunks=True, maxshape=(None, *data[key_data].shape[1:]))
                else:
                    dataset = hf[key_data]
                    dataset.resize((dataset.shape[0] + data[key_data].shape[0]), axis=0)
                    dataset[-data[key_data].shape[0]:] = data[key_data]

    def load(self, keys: List[str], number_data: Union[int, None] = None):
        dataset = h5py.File(self.dataset_path)

        if number_data is None:
            data = {}
            for key in keys:
                data[key] = dataset[key]
            return data
        else:
            total_values = dataset[keys[0]].shape[0]
            random_indices = np.random.choice(total_values, number_data)    # , replace=False
            random_indices.sort()
            random_indices = np.unique(random_indices)

            data = {}
            for key in keys:
                data[key] = dataset[key][random_indices]
            return data

    def load_index(self, keys: List[str], start_index: int, stop_index):
        dataset = h5py.File(self.dataset_path)
        data = {}
        for key in keys:
            data[key] = dataset[key][start_index:stop_index]
        return data

    def print_info(self):
        dataset = h5py.File(self.dataset_path)
        print('dataset name: ' + self.name)

        print('dataset subgroups:')
        for subgroups_name in dataset:
            print(' ' + subgroups_name + ':')
            print('  number data: ' + str(dataset[subgroups_name].shape[0]))
            print('  shape data: ' + str(dataset[subgroups_name].shape[1:]))

    def size(self, key: str):
        dataset = h5py.File(self.dataset_path)
        return dataset[key].shape[0]
