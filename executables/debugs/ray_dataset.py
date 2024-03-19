import ray
import numpy as np

# Initialize Ray
ray.init()

# Define a function to save data in the Ray object store
@ray.remote
class DataStore:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

# Generate some sample data
data = np.random.rand(1000)

# Save data in the Ray object store
data_ref = DataStore.remote(data)

# Retrieve data from the object store
data_from_store = ray.get(data_ref.get_data.remote())

# Now you have the data from the object store in the variable data_from_store

# Clean up resources
ray.shutdown()
