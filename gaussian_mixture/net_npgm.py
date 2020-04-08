#!/usr/bin/env python
# coding: utf-8

## Neural networks for the non-parametric gaussian mixture
import os
import datetime
import warnings
import numpy as np
from scipy.stats import wishart
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm


warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def generate_clusters(nb_samples=1, n_obs=1000, alpha=0.01,
                      prior_mean_scale=10, prior_covariance_scale=np.eye(2)/.2):
    '''
    Generation function for the clustering problem

    Args:
        nb_samples(int): number of samples
        n_obs(int): for each sample, the number of observations
        alpha (float): the parameter which regulates the creation of new samples.
            the greater alpha, the more clusters will be created
        prior_mean_scale (float) : hyper-parameter for the prior of the means
            of the gaussian distribution. The greater prior_mean_scale is
            the more the means of the gaussian distributions will be
            spread out
        prior_covariance_scale (matrix) : hyper-parameter for the prior of
            the covariance matrix of the gaussian distributions

    Return:
        X - a numpy array of size (nb_sample, n_obs, 2).
            For each sample, X gives the 2d coordinates of each observation
        Z - a numoy array of size (nb_sample, n_obs, 2).
            For each sample, Z gives the cluster of each observation
        nb_current_clusters - a numpy array of size (nb_sample, n_obs).
            For each sample, K gives the number of clusters
            as a function of the number of past observations available
    '''
    # initialize output variables
    X = np.zeros([nb_samples, n_obs, 2])
    Z = np.zeros([nb_samples, n_obs])
    nb_current_clusters = np.ones([nb_samples, n_obs], dtype=np.int)
    # initialize dictionaries for means and covariances of each cluster
    dict_of_means = {}
    dict_of_covariances = {}
    # sample mean and covariance for the first observation of each sample
    # and assign cluster 0 to that observation
    for k in range(nb_samples):
        new_mean = np.random.normal(size=2, scale=prior_mean_scale)
        new_covariance = wishart.rvs(2, prior_covariance_scale)
        X[k, 0] = np.random.multivariate_normal(new_mean, new_covariance)
        Z[k, 0] = 0
        dict_of_means[k, nb_current_clusters[k, 0] - 1] = new_mean
        dict_of_covariances[k, nb_current_clusters[k, 0] - 1] = new_covariance
    # iterate over all over observations
    for it_n_obs in range(1, n_obs):
        # probability to start a new cluster
        proba_new_cluster = alpha / (alpha + nb_current_clusters[:, it_n_obs])
        new_cluster = (np.random.rand(nb_samples) < proba_new_cluster)
        # for each observation, iterate over samples
        for k in range(nb_samples):
            if new_cluster[k]:
                # increment number of clusters
                nb_current_clusters[k, it_n_obs:] += 1
                # sample mean and covariance for new cluster
                new_mean = np.random.normal(size=2, scale=prior_mean_scale)
                new_covariance = wishart.rvs(2, prior_covariance_scale)
                # sample new observation and assign new cluster
                X[k, it_n_obs] = np.random.multivariate_normal(new_mean,
                                                               new_covariance)
                Z[k, it_n_obs] = nb_current_clusters[k, it_n_obs] - 1
                # save new cluster
                dict_of_means[k, nb_current_clusters[k, it_n_obs] - 1] \
                    = new_mean
                dict_of_covariances[k, nb_current_clusters[k, it_n_obs] - 1] \
                    = new_covariance
            else:
                # compute the probability of choosing each existing cluster
                counts = np.array([(Z[k, :it_n_obs] == j).sum()
                                   for j in range(int(Z[k, :it_n_obs].max() + 1))
                                  ])
                probability_oldcluster_unnormalized = (sum(counts)/counts)
                probability_oldcluster = probability_oldcluster_unnormalized \
                    / sum(probability_oldcluster_unnormalized)
                # sample cluster from existing ones
                old_cluster_id = np.random.choice(len(probability_oldcluster),
                                                  p=probability_oldcluster)
                # sample observation and assign cluster
                X[k, it_n_obs] = np.random.multivariate_normal(
                    dict_of_means[k, old_cluster_id],
                    dict_of_covariances[k, old_cluster_id])
                Z[k, it_n_obs] = old_cluster_id
    return X, Z, nb_current_clusters


class ClusterDataset(Dataset):
    """cluster dataset object"""

    def __init__(self, length=10000, transformation=None,
                 n_obs=100, alpha=0.01, prior_mean_scale=10,
                 prior_covariance_scale=np.eye(2)/.2):
        """
        the length parameter is completely arbitrary, as this object ignores the
        idx entirely and keeps producing new random datasets.

        Args:
            length (int,optional) : artificial length of the dataset (default=10000)
            transform (callable, optional) : Optional transform to be applied
                on a sample.
            n_obs(int or list of int) : number of points. If list is given it is randomly
                sampled from (default=100)
            prior_mean_scale(float) : scale of the prior for the clusters (default=10)
            prior_cov_scale(np.2darray) : scale of the Wishart for the covariances.
                Should be a 2x2 matrix (default=np.eye(2)/.2)

        """
        self.transform = transformation
        self.length = length
        self.n_obs = n_obs
        self.alpha = alpha
        self.prior_mean_scale = prior_mean_scale
        self.prior_covariance_scale = prior_covariance_scale

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X, Z, K = generate_clusters(n_obs=self.n_obs, alpha=self.alpha,
                                    prior_mean_scale=self.prior_mean_scale,
                                    prior_covariance_scale=self.prior_covariance_scale
                                   )
        np.random.shuffle(X)
        sample = {'X': X[0].astype(np.float32), 'K':K[0, -1]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ModelBasic(nn.Module):
    """parallel passing of data, categorical output with one unit per number of clusters"""
    def __init__(self, n_obs, n_units=100, n_timesteps=10, max_K=10):
        super(ModelBasic, self).__init__()
        self.fc_input = nn.Linear(2*n_obs, n_units)
        self.fc_output = nn.Linear(n_units, max_K)
        self.fc_recurrent = nn.Linear(n_units, n_units)
        self.n_obs = n_obs
        self.n_units = n_units
        self.n_timesteps = n_timesteps
        self.max_K = max_K

    def forward(self, x):
        inp = x.view(-1, 2 * self.n_obs)
        hidden = torch.tanh(self.fc_input(inp))
        for i_time in range(self.n_timesteps):
            hidden = hidden + torch.tanh(self.fc_recurrent(hidden)) \
                + torch.tanh(self.fc_input(inp))
        out = self.fc_output(hidden)
        return out

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_input.weight)
        nn.init.xavier_uniform_(self.fc_output.weight)
        nn.init.xavier_uniform_(self.fc_recurrent.weight)
        self.fc_input.bias.data.fill_(0)
        self.fc_output.bias.data.fill_(0)
        self.fc_recurrent.bias.data.fill_(0)


def accuracy(x, y):
    _, predicted = torch.max(x, 1)
    correct = (predicted == y).sum()
    return correct


loss = torch.nn.CrossEntropyLoss()


def optimize(model, N, optimizer, batch_size=20, clip=np.inf, n_obs=100,
             smooth_display=0.9, loss_file=None, t_max=10000, smooth_l=0,
             device='cpu', check_dir=None, filename=None):
    d = ClusterDataset(length=t_max, alpha=0.05, n_obs=n_obs)
    dataload = DataLoader(d, batch_size=batch_size, num_workers=20)
    print('starting optimization\n')
    if loss_file:
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
        else:
            losses = np.array([])
    with tqdm.tqdm(total=t_max * N,
                   dynamic_ncols=True, smoothing=0.01) as pbar:
        k0 = len(losses)
        k = k0
        losses = np.concatenate((losses,
                                 np.zeros(int(N * len(d) / batch_size))))
        for i_epoch in range(N):
            for i, samp in enumerate(dataload):
                k = k+1
                x_tensor = samp['X'].to(device)
                y_tensor = samp['K'].to(device)
                optimizer.zero_grad()
                y_est = model(x_tensor)
                l_batch = loss(y_est, y_tensor)
                l_batch.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                smooth_l = (smooth_display * smooth_l
                            + (1 - smooth_display) * l_batch.item())
                losses[k-1] = l_batch.item()
                pbar.postfix = ',  loss:%0.5f' % (smooth_l
                                                  / (1-smooth_display**(k-k0)))
                pbar.update(batch_size)
                if loss_file and not k % 25:
                    np.save(loss_file, losses)
            if ((check_dir is not None) and (filename is not None)):
                save_checkpoint(model, filename, check_dir, n_obs=n_obs,
                                batch_size=batch_size, device=device)


def overtrain(model, optimizer, n_obs=100, batch_size=20, clip=np.inf,
              smooth_display=0.9, loss_file=None, t_max=np.inf, smooth_l=0,
              device='cpu'):
    d = ClusterDataset(length=t_max, alpha=0.05, n_obs=n_obs)
    dataload = DataLoader(d, batch_size=batch_size, num_workers=20)
    print('starting optimization\n')
    with tqdm.tqdm(total=min(len(d), batch_size * t_max),
                   dynamic_ncols=True, smoothing=0.01) as pbar:
        losses = np.zeros(int(len(d)/batch_size))
        k0 = 0
        k = 0
        for i, samp in enumerate(dataload):
            k = k+1
            if i == 0:
                x_tensor = samp['X'].to(device)
                y_tensor = samp['K'].to(device)
            optimizer.zero_grad()
            y_est = model(x_tensor)
            l_batch = loss(y_est, y_tensor)
            l_batch.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            smooth_l = (smooth_display * smooth_l
                        + (1 - smooth_display) * l_batch.item())
            losses[k-1] = l_batch.item()
            pbar.postfix = ',  loss:%0.5f' % (smooth_l
                                              / (1-smooth_display**(k-k0)))
            pbar.update(batch_size)
            if k >= t_max:
                return


def evaluate(model, batch_size=20, device='cpu', n_max=1000,
             n_obs=100):
    d = ClusterDataset(length=n_max, alpha=0.05, n_obs=n_obs)
    dataload = DataLoader(d, batch_size=batch_size, num_workers=20)
    with tqdm.tqdm(total=n_max, dynamic_ncols=True, smoothing=0.01) as pbar:
        with torch.no_grad():
            losses = np.zeros(int(min(len(d), n_max) / batch_size))
            accuracies = np.zeros(int(min(len(d), n_max) / batch_size))
            for i, samp in enumerate(dataload):
                if i >= (n_max / batch_size):
                    break
                x_tensor = samp['X'].to(device)
                y_tensor = samp['K'].to(device)
                y_est = model(x_tensor)
                l_batch = loss(y_est, y_tensor)
                acc = accuracy(y_est, y_tensor)
                losses[i] = l_batch.item()
                accuracies[i] = acc.item() / batch_size
                pbar.postfix = ',  loss:%0.5f' % np.mean(losses[:(i+1)])
                pbar.update(batch_size)
    return losses, accuracies


def save_checkpoint(model, filename, check_dir, batch_size=20, n_obs=100,
                    device='cpu'):
    losses, accuracies = evaluate(model, n_obs=n_obs,
                                  batch_size=batch_size,
                                  device=device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename + '_' + timestamp
    path = check_dir + filename + '.pt'
    path_l = check_dir + filename + '_l.npy'
    path_acc = check_dir + filename + '_acc.npy'
    torch.save(model.state_dict(), path)
    np.save(path_l, np.array(losses))
    np.save(path_acc, np.array(accuracies))


def plot_loss(path_loss, smooth_n=25):
    losses = np.load(path_loss)
    plt.figure()
    plt.plot(np.convolve(losses, np.ones(smooth_n) / smooth_n, 'valid'))
    plt.show()


def get_model(model_name, n_obs, time, n_neurons, device):
    if model_name == 'basic':
        model = ModelBasic(n_obs).to(device)
    return model


def get_filename(model_name, n_obs, n_neurons, time):
    filename = 'model_%s' % model_name
    filename = filename + '_nobs%02d' % n_obs
    filename = filename + '_nn%02d' % n_neurons
    filename = filename + '_t%03d' % time
    return filename


def main(model_name, action, device='cpu', weight_decay=10**-3, epochs=1,
         lr=0.001, t_max=np.inf, batch_size=20, time=5, n_neurons=100,
         n_obs=100, path_models='nn_models/'):
    model = get_model(model_name, n_obs, time, n_neurons, device)
    filename = get_filename(model_name, n_obs, n_neurons, time)
    check_dir = 'check_points/'
    path = path_models + filename + '.pt'
    path_opt = path_models + filename + '_opt.pt'
    path_loss = path_models + filename + '_loss.npy'
    path_l = path_models + filename + '_l.npy'
    path_acc = path_models + filename + '_acc.npy'
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 amsgrad=True,
                                 weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),lr = lr,weight_decay=weight_decay)
    if action == 'reset':
        os.remove(path)
        os.remove(path_opt)
        os.remove(path_loss)
        os.remove(path_l)
        os.remove(path_acc)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        optimizer.load_state_dict(torch.load(path_opt, map_location=torch.device('cpu')))
        optimizer.param_groups[0]['lr'] = lr
    if action == 'train':
        optimize(model, epochs, optimizer, batch_size=batch_size, clip=np.inf,
                 smooth_display=0.9, loss_file=path_loss, t_max=t_max,
                 device=device, check_dir=check_dir, filename=filename,
                 n_obs=n_obs)
        torch.save(model.state_dict(), path)
        torch.save(optimizer.state_dict(), path_opt)
    elif action == 'eval':
        l, acc = evaluate(model, batch_size=batch_size, device=device)
        np.save(path_l, np.array(l))
        np.save(path_acc, np.array(acc))
    elif action == 'overtrain':
        overtrain(model, optimizer, n_obs=n_obs,
                  batch_size=batch_size, clip=np.inf, smooth_display=0.9,
                  loss_file=None, t_max=t_max, device=device)
    elif action == 'print':
        print(filename)
        if os.path.isfile(path_l):
            print('negative log-likelihood:')
            print('%.6f' % np.mean(np.load(path_l)))
            print('accuracy:')
            print('%.4f %%' % (100*np.mean(np.load(path_acc))))
        else:
            print('not yet evaluated!')
    elif action == 'plot_loss':
        plot_loss(path_loss, smooth_n=batch_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device",
                        help="device to run on [cuda,cpu]",
                        choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument("-E", "--epochs",
                        help="numer of epochs",
                        type=int, default=1)
    parser.add_argument("-b", "--batch_size",
                        help="size of a batch",
                        type=int, default=20)
    parser.add_argument("-w", "--weight_decay",
                        help="how much weight decay?",
                        type=float, default=0.01)
    parser.add_argument("-l", "--learning_rate",
                        help="learning rate",
                        type=float, default=10**-3)
    parser.add_argument("-s", "--t_max",
                        help="number of training steps",
                        type=int, default=10000)
    parser.add_argument("-t", "--time",
                        help="number of timesteps",
                        type=int, default=5)
    parser.add_argument("-n", "--n_neurons",
                        help="number of neurons",
                        type=int, default=100)
    parser.add_argument("-o", "--n_obs",
                        help="number of neurons",
                        type=int, default=100)
    parser.add_argument("action",
                        help="what to do? [train, eval, overtrain, print"
                            + ", reset, plot_loss]",
                        choices=['train', 'eval', 'overtrain', 'print',
                                 'reset', 'plot_loss'])
    parser.add_argument("model_name",
                        help="model to be trained ['basic']",
                        choices=['basic'])
    args = parser.parse_args()
    main(args.model_name, args.action, device=args.device,
         weight_decay=float(args.weight_decay), epochs=args.epochs,
         lr=args.learning_rate, t_max=args.t_max, batch_size=args.batch_size,
         time=args.time, n_neurons=args.n_neurons, n_obs=args.n_obs)
