import os

import numpy as np
import pandas as pd
import scipy.linalg as LA
import scipy.sparse as sps
from scipy.io import loadmat

import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

# animation
from matplotlib import animation
from celluloid import Camera

# visualisation of the weights matrix
import r1svd

##########################################################################
# Data Handling
##########################################################################


def fcps_reader(name, data_dir='../input/FCPS'):
    points = pd.read_csv(f"{data_dir}/{name}.lrn",
                         sep='\t', comment='%', index_col=0, header=None)
    labels = pd.read_csv(f"{data_dir}/{name}.cls",
                         sep='\t', comment='%', index_col=0, header=None)

    points = points.rename(columns=lambda x: f"x{x}")
    labels = labels.rename(columns={labels.columns[0]: 'y'})

    return pd.concat([points, labels], axis=1)


class FCPSDataset(Dataset):
    """FCPS Dataset loader for PyTorch"""

    def __init__(self, name, fcps_dir='../input/FCPS', transform=None):
        data = fcps_reader(name, fcps_dir)
        self.X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float)
        self.y = torch.tensor(data.y.values, dtype=torch.int8)
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageMatDataset(Dataset):
    """Image Dataset from .mat files  
    reads the files for: ORL, MNIST, 
        COIL20, COIL100, Yale and USPS datasets
    """

    def __init__(self, name, img_dir='../input', transform=None):
        mat = loadmat(f'{img_dir}/{name}.mat')
        self.X = torch.tensor(mat['fea'], dtype=torch.float)
        self.y = torch.tensor(mat['gnd'], dtype=torch.int8)
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


##########################################################################
# AutoEncoder
##########################################################################

class AutoEncoder(nn.Module):
    def __init__(self, encode_sizes, decode_sizes=None):
        super(AutoEncoder, self).__init__()
        if decode_sizes is None:
            decode_sizes = encode_sizes[::-1]

        self.encoder = AutoEncoder.make_sequential(encode_sizes)
        self.decoder = AutoEncoder.make_sequential(decode_sizes)
        self.d_in = encode_sizes[0]
        self.d_latent = encode_sizes[-1]

    @staticmethod
    def make_sequential(layer_sizes):
        n = len(layer_sizes)
        ls = layer_sizes
        # create fully connected layers with ReLU activations
        layers = [(nn.Linear(ls[i-1], ls[i]), nn.ReLU())
                  for i in range(1, n-1)]
        # unwrap tuples
        layers = [fn for layer in layers for fn in layer]
        # add the last layer
        layers += [nn.Linear(ls[n-2], ls[n-1])]
        # transform to sequential
        return nn.Sequential(*layers)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


##########################################################################
# LLE implementations
##########################################################################
def get_neighbors(X, k):
    n = X.shape[0]
    knn = NearestNeighbors(algorithm='kd_tree').fit(X)
    neighborhood = knn.kneighbors(X, n_neighbors=k+1, return_distance=False)

    # filter points that are their own neighbors
    x = np.arange(n)
    mask = neighborhood != x[:, np.newaxis]
    self_found = np.sum(~mask, axis=1) == 1
    mask[:, -1] &= self_found

    return neighborhood[mask].reshape((n, -1))

    # dists = squareform(pdist(X))
    # idx = np.arange(n)
    # dists[idx == idx[:, np.newaxis]] = np.Inf
    # # neighborhood = np.argsort(dists, axis=1)[:, :self.k]
    # neighborhood = np.argpartition(dists, self.k+1, axis=1)[:, :self.k]


class LLE:
    def __init__(self, n_neighbors=5, d=2, reg=1E-3):
        self.reg = reg
        self.k = n_neighbors
        self.d = d

    def fit(self, X):
        n, r = X.shape

        # Step 1: compute pairwise distances
        neighborhood = get_neighbors(X, self.k)

        # Step 2: solve for the reconstruction weights W
        W = np.zeros((n, n))
        ones = np.ones(self.k)

        for i in range(n):
            nbrs = neighborhood[i, :]
            z = X[nbrs, :] - X[i, :]
            C = z @ z.T

            # regularization
            trace = float(C.trace())
            C.flat[::self.k+1] += self.reg * trace if trace > 0 else self.reg

    #         W[i, nbrs] = LA.pinv(C) @ ones
            W[i, nbrs] = LA.solve(C, ones)

    #     print(np.round(W, 2))
        W = W / W.sum(axis=1)[:, np.newaxis]
        self.S_ = W.copy()

        # Step 3: compute the embedding from the eigenvectors
        W.flat[::W.shape[0]+1] -= 1  # W - I
        # calculate the embeddings
        u, w, v = LA.svd(W)

        idx = np.argsort(w)[1:(self.d+1)]
        Y = v.T[:, idx]  # * n**.5

        self.B_ = Y

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.S_, self.B_


class SparseLLE:
    def __init__(self, n_neighbors=5, d=2, reg=1E-3):
        self.reg = reg
        self.k = n_neighbors
        self.d = d

    def fit(self, X):
        n, r = X.shape

        # Step 1: compute pairwise distances
        neighborhood = get_neighbors(X, self.k)

        # Step 2: solve for the reconstruction weights W
        W = np.zeros((n, n))
        ones = np.ones(self.k)

        rows = np.empty(n*self.k, dtype=int)
        cols = np.empty(n*self.k, dtype=int)
        vals = np.empty(n*self.k)
        for i in range(n):
            nbrs = neighborhood[i, :]
            z = X[nbrs, :] - X[i, :]
            C = z @ z.T

            # regularization
            trace = float(C.trace())
            C.flat[::self.k+1] += self.reg * trace if trace > 0 else self.reg

            values = LA.solve(C, ones)

            s, e = i*self.k, (i+1)*self.k
            vals[s:e] = values
            rows[s:e] = np.repeat(i, self.k)
            cols[s:e] = nbrs

        W = sps.csr_matrix((vals, (rows, cols)), shape=(n, n))
    #     print(np.round(W.toarray(), 2))

        w_sum = W.sum(axis=1).A.ravel()
        w_sum_inv = sps.diags(1/w_sum)

        W = w_sum_inv @ W

        self.S_ = W.toarray().copy()

        # Step 3: compute the embedding from the eigenvectors
        I = sps.eye(n)
        W = W - I
        # calculate the embeddings
        u, w, v = sps.linalg.svds(W, k=n-1, which='LM')
    #     u, w, v = LA.svd(W.toarray())
    #     w, v = sps.linalg.eigsh(W.T @ W, k=d+1, which='SM', sigma=0)
    #     print(w.shape, v.shape)
    #     v = v.T

        idx = np.argsort(w)[:self.d]
        Y = v.T[:, idx]  # * n**.5

        self.B_ = Y

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.S_, self.B_


##########################################################################
# Model Trainer
##########################################################################

class Trainer:

    def __init__(self, net, loader, dataset, optimizer,
                 n_epochs, n_neighbors=9, l=1, sparse_lle=False):
        self.net = net
        self.loader = loader
        self.dataset = dataset
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.n_neighbors = n_neighbors
        self.l = l
        self.LLE = SparseLLE if sparse_lle else LLE

    def train_model(self, track_W=False, track_Y=False):
        if track_W:
            w_fig, w_ax = plt.subplots()
            self.w_camera_ = Camera(w_fig)
            r = r1svd.RankOneSvd()
        else:
            self.y_camera_ = None

        if track_Y:
            y_fig, y_ax = plt.subplots()
            self.y_camera_ = Camera(y_fig)
            r = r1svd.RankOneSvd()
        else:
            self.y_camera_ = None

        losses = []
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.n_epochs)):

            for inputs, labels in self.loader:

                # encode
                enc, dec = self.net(inputs)
                # compute
                X_enc = enc.detach().numpy()
                S, _ = self.LLE(self.n_neighbors,
                                X_enc.shape[1]).fit_transform(X_enc)

                cost = self.criterion(inputs, enc, dec, torch.tensor(
                    S, dtype=torch.float),
                    l=self.l if epoch <= 10 else 1 + .5*(epoch - 10))
            #     print(f"Loss:\t{cost.item():.3f}")
                losses.append(cost.item())

                # update autoencoder weights
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                if track_W:
                    tmp = r.fit_transform(np.abs(S))
                    w_ax.spy(tmp)
                    self.w_camera_.snap()
                if track_Y:
                    color = labels.detach().numpy().ravel()
                    y_ax.scatter(X_enc[:, 0], X_enc[:, 1],
                                 c=color, cmap='Set1')
                    self.y_camera_.snap()
        if track_W:
            plt.close(w_fig)
        if track_Y:
            plt.close(y_fig)

        return losses

    def predict(self, loader):
        with torch.no_grad():
            n = len(self.dataset)
            d_ae = self.net.d_latent
            d_in = self.net.d_in

            X_ae = np.empty((n, d_ae))
            batch_size = self.loader.batch_size
            labels = np.empty(n)

            X = np.empty((n, d_in))

            for i, (inputs, outputs) in enumerate(loader):
                x, _ = self.net(inputs)
                x = x.numpy()
                s = i*batch_size
                e = min((i+1)*batch_size, n)
                X[s:e, :] = inputs.numpy()
                X_ae[s:e, :] = x
                out = outputs.numpy().ravel()
                if i == 0:
                    labels = labels.astype(out.dtype)
                labels[s:e] = out

            S, Y = self.LLE(self.n_neighbors, d_ae).fit_transform(X_ae)

            inp = torch.tensor(X, dtype=torch.float)
            enc, dec = self.net(inp)
            cost = Trainer.criterion(X, enc.numpy(), dec.numpy(), S, l=self.l)
#             Y = LocallyLinearEmbedding(self.n_neighbors, d_ae).fit_transform(X_ae)
        return S, X_ae, Y, labels, cost

    @staticmethod
    def criterion(X, encoded, decoded, S, l):
        """Computes the autoencoder loss
        loss = ||X - decoded||^2 + l * ||encoded - S*encoded||^2
        the autoencoder should keep as much information while also
        keeping the neighborhoods as separate as possible // could be better phrased
        """
        sim = X - decoded  # how similar the result is to the input
        sep = encoded - S @ encoded  # how well separated the embedding is

        return (sim ** 2).sum() + l * (sep ** 2).sum()
