import random
import json
import logging
import sys
import os
import torch
import argparse
import pickle
import numpy as np
from utils.metrics import *
from utils.helper import *
from model.pfn import PFN
from dataloader.dataloader import dataloader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
from helpers_al import VAE, query_samples, Discriminator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="which dataset to use")

    parser.add_argument("--epoch", default=100, type=int,
                        help="number of training epoch")

    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")

    parser.add_argument("--batch_size", default=20, type=int,
                        help="number of samples in one training batch")
    
    parser.add_argument("--eval_batch_size", default=10, type=int,
                        help="number of samples in one testing batch")
    
    parser.add_argument("--do_train", action="store_true",
                        help="whether or not to train from scratch")

    parser.add_argument("--do_eval", action="store_true",
                        help="whether or not to evaluate the model")

    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="BERT or ALBERT pretrained embedding")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")

    parser.add_argument("--lr", default=None, type=float,
                        help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")
    
    parser.add_argument("--linear_warmup_rate", default=0.0, type=float,
                        help="warmup at the start of training")
    
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed initiation")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")

    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")

    parser.add_argument("--output_file", default="test", type=str, required=True,
                        help="name of result file")

    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")

    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="maximum length of sequence")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


       
    with open(args.data + "/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open(args.data + "/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    if args.embed_mode == "albert":
            input_size = 4096
    else:
        input_size = 768

    

    train_dataset, test_dataset, dev_dataset, collate_fn, train_unlabeled = dataloader(args, ner2idx, rel2idx)
    no_train = len(train_dataset)
    indices = list(range(no_train))
    model = PFN(args, input_size, ner2idx, rel2idx)
    model = torch.load('predictor-backbone-cycle-6.pth')
    labeled_setn = np.load('labelled-head5.npy')
    labeled_setprev_ind = np.load('labelled-head4.npy')

    unlabeled_set = [x for x in indices if x not in labeled_setn]
    unlabeled_set = unlabeled_set[:400]

    # labeled_setn = np.setdiff1d(labeled_setn, labeled_setprev_ind)


    labeled_batchn = DataLoader(dataset=train_dataset, batch_size=len(labeled_setn), sampler=SubsetRandomSampler(labeled_setn), 
                                        pin_memory=True, collate_fn=collate_fn)
    labeled_setprev = DataLoader(dataset=train_dataset, batch_size=len(labeled_setprev_ind), sampler=SubsetRandomSampler(labeled_setprev_ind), 
                                        pin_memory=True, collate_fn=collate_fn)
    unlabeled_batch = DataLoader(dataset=train_dataset, batch_size=len(unlabeled_set), sampler=SubsetRandomSampler(unlabeled_set), 
                                        pin_memory=True, collate_fn=collate_fn)
    with torch.no_grad():
        for data in labeled_batchn:
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_score, re_core, features, out4 = model(text, mask)

            lab_features = features[-1]

        for data in unlabeled_batch:
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_score, re_core, features, out4 = model(text, mask)


            unlab_features = features[-1]

        for data in labeled_setprev:
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_score, re_core, features, out4 = model(text, mask)

            lab_featuresprev = features[-1]
        tsne = TSNE(n_components=2,random_state=42,perplexity=10)
        print(lab_features.shape)
        lab_features = lab_features.cpu()
        unlab_features = unlab_features.cpu()
        lab_featuresprev = lab_featuresprev.cpu()
        lab_features = lab_features.reshape(lab_features.shape[0],-1)
        unlab_features = unlab_features.reshape(unlab_features.shape[0],-1)
        lab_featuresprev = lab_featuresprev.reshape(lab_featuresprev.shape[0],-1)

        
        lab_features = lab_features.numpy()
        unlab_features = unlab_features.numpy()
        lab_featuresprev = lab_featuresprev.numpy()
        X = np.concatenate([lab_features,unlab_features,lab_featuresprev],axis=0)
        train_tsne = tsne.fit_transform(X)
        # test_tsne = tsne.fit_transform(unlab_features)
        # labprevtsne = tsne.fit_transform(lab_featuresprev)

        # Step 5: Visualize t-SNE plots
        plt.figure(figsize=(10, 5))

        # Train t-SNE plot
        plt.scatter(train_tsne[:len(lab_features), 0], train_tsne[:len(lab_features), 1],label='Newly added samples',c='red')

        # Test t-SNE plot
        plt.scatter(train_tsne[len(lab_features):len(lab_features)+len(unlab_features), 0], train_tsne[len(lab_features):len(lab_features)+len(unlab_features), 1], label='Unlabeled samples',c='blue')

        plt.scatter(train_tsne[len(lab_features)+len(unlab_features):, 0], train_tsne[len(lab_features)+len(unlab_features):, 1], label='Prev labeled samples',c='green')

        plt.title('WEBNLG, Ours w DV - Cycle 6')
        plt.tight_layout()
        plt.legend()
        plt.savefig('tsne_plot_dyn_weight_cycle6.png')
        plt.show()

        print("done plotting")