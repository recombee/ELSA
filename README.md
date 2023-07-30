[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ELSA

This is an official implementation of our paper [Scalable Linear Shallow Autoencoder for Collaborative Filtering](https://dl.acm.org/doi/10.1145/3523227.3551482).

### Requirements

PyTorch in version >=10.1 (along with compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)) must be installed in the system. If not, one can install [PyTorch](https://pytorch.org/get-started/locally/) with 
```
pip install torch==1.10.2+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Instalation

ELSA can be installed from [pypi](https://pypi.org/project/elsarec/) with:

```
pip install elsarec
```

## Basic usage

```python
from elsa import ELSA
import torch
import numpy as np

device = torch.device("cuda")

X_csr = ... # load your interaction matrix (scipy.sparse.csr_matrix with users in rows and items in columns)
X_test = ... # load your test data (scipy.sparse.csr_matrix with users in rows and items in columns)

items_cnt = X_csr.shape[1]
factors = 256 
num_epochs = 5
batch_size = 128

model = ELSA(n_items=items_cnt, device=device, n_dims=factors)

model.fit(X_csr, batch_size=batch_size, epochs=num_epochs)

# save item embeddings into np array
A = torch.nn.functional.normalize(model.get_items_embeddings(), dim=-1).cpu().numpy()

# get predictions in PyTorch
predictions = model.predict(X_test, batch_size=batch_size)

# get predictions in numpy
predictions = ((X_test @ A) @ (A.T)) - X_test

# find related items for a subset of items
itemids = np.array([id1, id2, ...])  # id1, id2 are indices of items in the X_csr
related = model.similar_items(N=100, batch_size=128, sources=itemids)
```

## Notes

### Reproducibility

Instructions for reproducing the results from the paper on the MovieLens20M dataset are in the branch `reproduce_movielens` https://github.com/recombee/ELSA/tree/reproduce_movielens.

Reproducibility instructions for Netflix and Goodbooks10k datasets will follow soon.

### Tensorflow users

We decided to implement ELSA in PyTorch, but implementation in TensorFlow is simple and straightforward. One can, for example, implement ELSA as a Keras layer:

```python
class ELSA(tf.keras.layers.Layer):
    def __init__(self, latent, nr_of_items):
        super(ELSA, self).__init__()
        w_init = tf.keras.initializers.HeNormal()
        self.A = tf.Variable(
            initial_value=w_init(shape=(nr_of_items, latent), dtype="float32"),
            trainable=True,
        )
    
    def get_items_embeddings(self):
        A = tf.math.l2_normalize(self.A, axis=-1)
        return A.numpy()
    
    @tf.function
    def call(self, x):
        A = tf.math.l2_normalize(self.A, axis=-1)
        xA = tf.matmul(x, A, transpose_b=False)
        xAAT = tf.matmul(xA, A, transpose_b=True)
        return xAAT - x
```

### Licence
[MIT licence](https://github.com/recombee/ELSA/blob/main/LICENCE)

### Troubleshooting
If you encounter a problem or have a question about ELSA, do not hesitate to create an issue and ask. In case of an implementation problem, please include the Python, PyTorch and CUDA versions in the description of the issue.

### Cite us

Please consider citing our paper:

```
@inproceedings{10.1145/3523227.3551482,
author = {Van\v{c}ura, Vojt\v{e}ch and Alves, Rodrigo and Kasalick\'{y}, Petr and Kord\'{\i}k, Pavel},
title = {Scalable Linear Shallow Autoencoder for Collaborative Filtering},
year = {2022},
isbn = {9781450392785},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3523227.3551482},
doi = {10.1145/3523227.3551482},
abstract = {Recently, the RS research community has witnessed a surge in popularity for shallow autoencoder-based CF methods. Due to its straightforward implementation and high accuracy$
booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
pages = {604–609},
numpages = {6},
keywords = {Linear models, Shallow autoencoders, Implicit feedback recommendation},
location = {Seattle, WA, USA},
series = {RecSys '22}
}
```
