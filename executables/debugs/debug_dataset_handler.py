
from utilities.dataset_handler import DatasetHandler

dataset_handler: DatasetHandler = DatasetHandler('./results/data/', 'test')
dataset_handler.print_info()