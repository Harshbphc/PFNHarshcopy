import torch

from sklearn.manifold import TSNE

# from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt

num_classes = 2

# cf = CIFAR10('')

def test_tensor_batch():
    return torch.rand([8,num_classes])

def train_tensor_batch():
    return torch.rand([8,num_classes])

train_features = []
train_predictions = []

for i in range(30):
    feats = train_tensor_batch()
    preds = torch.argmax(feats,dim=1)
    train_features.append(feats)
    train_predictions.append(preds)

test_features = []
test_predictions = []

for i in range(5):
    feats = train_tensor_batch()
    preds = torch.argmax(feats,dim=1)
    test_features.append(feats)
    test_predictions.append(preds)

train_features = torch.cat(train_features, dim=0)  # Concatenate features from all batches
train_predictions = torch.cat(train_predictions, dim=0)  # Concatenate predictions from all batches


test_features = torch.cat(test_features, dim=0)  # Concatenate features from all batches
test_predictions = torch.cat(test_predictions, dim=0)  # Concatenate predictions from all batches

tsne = TSNE(n_components=num_classes,random_state=42)

train_tsne = tsne.fit_transform(train_features)
test_tsne = tsne.fit_transform(test_features)

# Step 5: Visualize t-SNE plots
plt.figure(figsize=(10, 5))

# Train t-SNE plot
plt.subplot(1, 2, 1)
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_predictions, cmap='viridis')
plt.title('Train t-SNE Plot')

# Test t-SNE plot
plt.subplot(1, 2, 2)
plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_predictions, cmap='viridis')
plt.title('Test t-SNE Plot')

plt.tight_layout()
plt.show()
# plt.savefig('trial2.png')