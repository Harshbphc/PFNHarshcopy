import matplotlib as plt
import torch

plt.figure(figsize=(10, 5))

    # Train t-SNE plot
train_tsne = torch.rand([20,2])
ner_labpred = torch.rand([20,1])
re_labpred = torch.rand([20,1])
plt.subplot(1, 2, 1)
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=ner_labpred+re_labpred, cmap='viridis')
plt.title('Train t-SNE Plot')

    # Test t-SNE plot
# plt.subplot(1, 2, 2)
# plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=ner_unlabpred+re_unlabpred, cmap='viridis')
# plt.title('Test t-SNE Plot')

plt.tight_layout()
plt.show()

print("done plotting")