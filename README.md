# Reproducing results on MovieLens20M dataset 

To reproduce the results from our paper "Scalable Linear Shallow Autoencoder for Collaborative Filtering":

1. run `MovieLens - preprocessing.ipynb` to download and preprocess the MovieLens20M dataset
2. then run `python train_movielens.py --factors=800 --batch_size=32` to reproduce the results from the paper.
