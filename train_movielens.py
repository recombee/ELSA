import numpy as np
import pandas as pd

from utils import *
from elsa import *

import tensorflow as tf
import tensorflow_addons as tfa

from time import time
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=0, type=int, help="Default device to run on (tf)")

parser.add_argument("--factors", default=64, type=int, help="Number of ELSA factors")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size for ELSA training")
parser.add_argument("--lr", default=.1, type=float, help="Laerning rate for ELSA training")
parser.add_argument("--epochs", default=10, type=int, help="Epochs of ELSA training")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
    tf.config.set_visible_devices([physical_devices[args.device]], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[args.device], True)
    set_seed(args.seed)
    
    dataset = Data(d='', pruning='u5')
    dataset.splits = []
    dataset.create_splits(1, 10000, shuffle=False, generators=False)
    dataset.split.train_users = pd.read_json("train_users.json").userid.apply(str).to_frame()
    dataset.split.validation_users = pd.read_json("val_users.json").userid.apply(str).to_frame()
    dataset.split.test_users = pd.read_json("test_users.json").userid.apply(str).to_frame()
    dataset.split.generators(
        batch_size=args.batch_size,
        random_batching=True,
        prevent_identity=False,
        full_data=True,
        p50_splits=False,
        p2575_splits=False,
        p7525_splits=False,
        p2525_splits=False,
        p7575_splits=False,                        
    )
    name="ELSA_f_"+str(args.factors)+"_lr_"+str(args.lr)+"_ep_"+str(args.epochs)
    m = ELSA(dataset.split, name=name)
    m.create_model(latent=args.factors)
    m.model.summary()
    print("=" * 80)
    print("Train for 10 epochs with lr 0.1")
    m.compile_model(lr=args.lr)
    m.train_model(args.epochs)
    m.model.load_weights(m.name +"_best_ncdg_100/" + m.name)
    history_df=m.mc.get_history_df()
    
    test_r20s = []
    test_r50s = []
    test_n100s = []

    for fold in range(1,6):
        ev=Evaluator(m.split, method=str(fold)+'_20')
        ev.update(m.model)

        test_n100s.append(ev.get_ncdg(100))
        test_r20s.append(ev.get_recall(20))
        test_r50s.append(ev.get_recall(50))

    ncdg100 = round(sum(test_n100s) / len(test_n100s),4)    
    recall20 = round(sum(test_r20s) / len(test_r20s),4)
    recall50 = round(sum(test_r50s) / len(test_r50s),4)

    print("TEST SET (MEAN)")
    print("5-fold mean NCDG@100", ncdg100)
    print("5-fold mean Recall@20", recall20)
    print("5-fold mean Recall@50", recall50)

