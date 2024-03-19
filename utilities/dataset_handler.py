from typing import Dict
import h5py


class DatasetHandler:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.dataset_path = self.path + '/' + self.name + '.h5'

    def save(self, data: Dict):
        with h5py.File(self.dataset_path, 'a') as hf:
            for key_data in data.keys():
                if key_data not in hf:
                    hf.create_dataset(key_data, data=data[key_data], chunks=True, maxshape=(None, *data[key_data].shape[1:]))
                else:
                    dataset = hf[key_data]
                    dataset.resize((dataset.shape[0] + data[key_data].shape[0]), axis=0)
                    dataset[-data[key_data].shape[0]:] = data[key_data]

    def load(self, key_data):
        dataset = h5py.File(self.dataset_path)
        return dataset[key_data]

    def print_info(self):
        dataset = h5py.File(self.dataset_path)
        print('dataset name: ' + self.name)

        print('dataset subgroups:')
        for subgroups_name in dataset:
            print(' ' + subgroups_name + ':')
            print('  number data: ' + str(dataset[subgroups_name].shape[0]))
            print('  shape data: ' + str(dataset[subgroups_name].shape[1:]))
