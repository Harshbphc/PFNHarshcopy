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

class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):  
        z = torch.cat([z, r], 1)
        return self.net(z)
    
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(                                                   #   B 3 96 96
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32     B 128 48 48
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16      B 256 24 24
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8        B 512 12 12 
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4  B 1024 6 6
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*6*6)),     #1024*14*12                            # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 1024*6*6),                           # B, 1024*8*8
            View((-1, 1024, 6, 6)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 2, 2),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, r, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z,r],1)
        x_recon = self._decode(z)

        return  x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, args):
    
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']
    
    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    vae = vae.to(device)
    discriminator = discriminator.to(device)
    task_model = task_model.to(device)
    ranker = ranker.to(device)

    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = 100 # int( (args['INCREMENTAL']*cycle+ args['SUBSET']) * args['EPOCHV'] / args['BATCH'] )
    print('Num of Iteration:', str(train_iterations))
    
    for iter_count in range(train_iterations): #2
        data_sub_label = next(labeled_data)
        labeled_imgs = data_sub_label[0]
        lab_img_dup = labeled_imgs
        data_sub_unlabel = next(unlabeled_data)
        unlabeled_imgs = data_sub_unlabel[0]
        label_mask = data_sub_label[-1]
        unlabel_mask = data_sub_unlabel[-1]
        unlabeled_imgs_dup = unlabeled_imgs

        if args.embed_mode == 'albert':
            tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            bert = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif args.embed_mode == 'bert_cased':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            bert = AutoModel.from_pretrained("bert-base-cased")
        elif args.embed_mode == 'scibert':
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        labeled_imgs = tokenizer(labeled_imgs, return_tensors="pt",
                                  padding='longest',
                                  is_split_into_words=True)#.to(device)

        labeled_imgs = bert(**labeled_imgs)[0]
        unlabeled_imgs = tokenizer(unlabeled_imgs, return_tensors="pt",
                                  padding='longest',
                                  is_split_into_words=True)#.to(device)
        unlabeled_imgs = bert(**unlabeled_imgs)[0]


        labeled_imgs = labeled_imgs.to(device)
        unlabeled_imgs = unlabeled_imgs.to(device)
        # labels = labels.to(device)
        label_mask = label_mask.to(device)
        unlabel_mask = unlabel_mask.to(device)

        if iter_count == 0 :
            r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
            r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
        else:
            with torch.no_grad():
                _,_,features_l = task_model(lab_img_dup,label_mask)
                _,_,feature_u = task_model(unlabeled_imgs_dup,unlabel_mask)
                r_l = ranker(features_l)
                r_u = ranker(feature_u)
        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.sigmoid(r_l).detach()
            r_u_s = torch.sigmoid(r_u).detach()   
        # print(labeled_imgs.shape)
        desired_size = (labeled_imgs.shape[0], 100, 768)
        pad_dimensions = []
        for original_size, desired_size in zip(labeled_imgs.size(), desired_size):
            pad_size = max(0, desired_size - original_size)
            pad_dimensions.append(0)  # Pad with zeros at the end
            pad_dimensions.append(pad_size)
        # Pad the tensor
        pad_dimensions = tuple(pad_dimensions)
        labeled_imgs= torch.nn.functional.pad(labeled_imgs, pad_dimensions)
        # print(labeled_imgs.shape)
        # print(unlabeled_imgs.shape)
        pad_dimensions = []
        desired_size = (unlabeled_imgs.shape[0], 100, 768)
        for original_size, desired_size in zip(unlabeled_imgs.size(), desired_size):
            pad_size = max(0, desired_size - original_size)
            pad_dimensions.append(0)  # Pad with zeros at the end
            pad_dimensions.append(pad_size)
        # Pad the tensor
        pad_dimensions = tuple(pad_dimensions)
        unlabeled_imgs= torch.nn.functional.pad(unlabeled_imgs, pad_dimensions)

        labeled_imgs = labeled_imgs[:,:100,:]
        unlabeled_imgs = unlabeled_imgs[:,:100,:]
        labeled_imgs = labeled_imgs.reshape([labeled_imgs.shape[0], 3, 128, 200])
        unlabeled_imgs = unlabeled_imgs.reshape([unlabeled_imgs.shape[0], 3, 128, 200])
        labeled_imgs = torch.nn.functional.interpolate(labeled_imgs, size=(96, 96), mode='bilinear', align_corners=False)
        unlabeled_imgs = torch.nn.functional.interpolate(unlabeled_imgs, size=(96, 96), mode='bilinear', align_corners=False)
        # print(unlabeled_imgs.shape)
        
        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            lab_real_preds = lab_real_preds.to(device)
            unlab_real_preds = unlab_real_preds.to(device)            

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]
                
                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                # labels = labels.to(device)                

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s,labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
            
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)            
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                # labels = labels.to(device)                
                
            if iter_count % 50 == 0:
                # print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                SummaryWriter('logs/SCIERC_Train').add_scalar(str(cycle) + ' Total VAE Loss ',
                        total_vae_loss.item(), iter_count)
                SummaryWriter('logs/SCIERC_Train').add_scalar(str(cycle) + ' Total DSC Loss ',
                        dsc_loss.item(), iter_count)

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

    arg = get_kcg(model, 50*(cycle+1), unlabeled_loader)

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