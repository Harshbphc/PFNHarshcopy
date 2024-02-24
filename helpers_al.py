import torch
from utils.metrics import *
from utils.helper import *
import numpy as np
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import pairwise_distances
import abc

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def select_batch_unc_(self, **kwargs):
      return self.select_batch_unc_(**kwargs)

  def to_dict(self):
    return None
  
class kCenterGreedy(SamplingMethod):

    def __init__(self, X,  metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
          self.min_distances = None
        if only_new:
          cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if cluster_centers:
          x = self.features[cluster_centers]
          # Update min_distances for all examples given new cluster center.
          dist = pairwise_distances(self.features, x, metric=self.metric)#,n_jobs=4)

          if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
          else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
          if self.already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)
          # New examples should not be in already selected since those points
          # should have min_distance of zero to a cluster center.
          assert ind not in already_selected

          self.update_distances([ind], only_new=True, reset_dist=False)
          new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))


        self.already_selected = already_selected

        return new_batch

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(0):
        features = torch.tensor([]).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    SUBSET = 50
    ADDENDUM = 50
    with torch.no_grad():
        for data in unlabeled_loader:                       
            inputs = data[0]
            mask = data[-1]
            mask = mask.to(device)
            _, _,_, features_batch = models['backbone'](inputs,mask)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return  other_idx + batch

def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args, collate_fn):
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True,collate_fn= collate_fn)

    arg = get_kcg(model, 50+30*cycle, unlabeled_loader)

    return arg

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for data in dataloader:
                yield data
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD