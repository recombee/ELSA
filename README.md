# Reproducing results on MovieLens20M dataset 

To reproduce the results from our paper "Scalable Linear Shallow Autoencoder for Collaborative Filtering":

1. run `MovieLens - preprocessing.ipynb` to download and preprocess the MovieLens20M dataset
2. then run `python train_movielens.py --factors=800 --batch_size=32` to reproduce the results from the paper.

Logged results of training are stored in `train_movielens.ipynb`.

Please consider citating our paper:

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
abstract = {Recently, the RS research community has witnessed a surge in popularity for shallow autoencoder-based CF methods. Due to its straightforward implementation and high accuracy on item retrieval metrics, EASE is potentially the most prominent of these models. Despite its accuracy and simplicity, EASE cannot be employed in some real-world recommender system applications due to its inability to scale to huge interaction matrices. In this paper, we proposed ELSA, a scalable shallow autoencoder method for implicit feedback recommenders. ELSA is a scalable autoencoder in which the hidden layer is factorizable into a low-rank plus sparse structure, thereby drastically lowering memory consumption and computation time. We conducted a comprehensive offline experimental section that combined synthetic and several real-world datasets. We also validated our strategy in an online setting by comparing ELSA to baselines in a live recommender system using an A/B test. Experiments demonstrate that ELSA is scalable and has competitive performance. Finally, we demonstrate the explainability of ELSA by illustrating the recovered latent space.},
booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
pages = {604–609},
numpages = {6},
keywords = {Linear models, Shallow autoencoders, Implicit feedback recommendation},
location = {Seattle, WA, USA},
series = {RecSys '22}
}
```

and the paper from we used the framework for offline experiments:

```
@InProceedings{10.1007/978-3-030-86383-8_11,
author="Van{\v{c}}ura, Vojt{\v{e}}ch
and Kord{\'i}k, Pavel",
editor="Farka{\v{s}}, Igor
and Masulli, Paolo
and Otte, Sebastian
and Wermter, Stefan",
title="Deep Variational Autoencoder with Shallow Parallel Path for Top-N Recommendation (VASP)",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="138--149",
abstract="The recently introduced Embarrasingelly Shallow Autoencoder (EASE) algorithm presents a simple and elegant way to solve the top-N recommendation task. In this paper, we introduce Neural EASE to further improve the performance of this algorithm by incorporating techniques for training modern neural networks. Also, there is a growing interest in the recsys community to utilize variational autoencoders (VAE) for this task. We introduce Focal Loss Variational AutoEncoder (FLVAE), benefiting from multiple non-linear layers without an information bottleneck while not overfitting towards the identity. We show how to learn FLVAE in parallel with Neural EASE and achieve state-of-the-art performance on the MovieLens 20M dataset and competitive results on the Netflix Prize dataset.",
isbn="978-3-030-86383-8"
}
```
