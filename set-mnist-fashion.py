import tensorflow_datasets as tfds
import numpy as np
import pickle
import torch

# Load Fashion MNIST dataset
(train_ds, test_ds), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Convert the tf.data.Dataset to numpy arrays
features = []
labels = []
for image, label in tfds.as_numpy(train_ds):
    features.append(image)
    labels.append(label)

features = np.stack(features)  # shape: (num_samples, 28, 28)
labels = np.array(labels)      # shape: (num_samples,)

# Convert to torch tensors for compatibility with dataset_viewer.py
features = torch.from_numpy(features)
labels = torch.from_numpy(labels)

# If you want to flatten images, uncomment the next line
# features = features.reshape(features.shape[0], -1)

# Save as pickle file compatible with dataset_viewer.py
with open('fashion_mnist.pkl', 'wb') as f:
    pickle.dump({'features': features, 'labels': labels}, f)

print('Saved fashion_mnist.pkl with', features.shape[0], 'samples.')
