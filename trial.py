from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from datasets import Dataset
from datasets import concatenate_datasets
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the directory where you want to store the dataset
cache_dir = "/mnt/MIG_Store/Datasets/cc100/cc100/en-lang=en/0.0.0/8159941b93eb06d0288bb80be26ddfe8213c0c5e33286619c85ad8e1ee0eb91c"

# Load the dataset and specify the cache directory
#dataset = load_dataset("cc100", lang="en", cache_dir=cache_dir, split="train")
dataset = Dataset.from_file(cache_dir+"/cc100-train-00000-of-00720.arrow")
for i in range (1,5): #change it to 720 from 20, this was for test
    print("Reading..."+str(i)+" of 720")
    temp = Dataset.from_file(cache_dir+"/cc100-train-"+format(i, '05d')+"-of-00720.arrow")
    dataset = concatenate_datasets([dataset, temp])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert = AutoModel.from_pretrained("bert-base-cased")

for i in range (0,4):
    x = tokenizer(dataset[i]['text'], return_tensors="pt",
                                  padding='longest',
                                  is_split_into_words=True).to(device)

    x = bert(**x)[0]
    print(x)