# PFN (Partition Filter Network)
This repository contains codes of the NLP experiments carried out MJSR-AL paper.

## Quick links
* [Model Overview](#Model-Overview)
  * [Framework](#Framework)
  * [Equation Explanation](#Equation-Explanation)
* [Preparation](#Preparation)
  * [Environment Setup](#Environment-setup)
  * [Data Acquisition and Preprocessing](#Data-Acquisition-and-Preprocessing)
  * [Custom Dataset](#Custom-Dataset)
* [Quick Start](#Quick-Start)
  * [Model Training](#Model-Training)
  * [Evaluation on Pre-trained Model](#Evaluation-on-Pre-trained-Model)
  * [Inference on Customized Input](#Inference-on-Customized-Input)
* [Evaluation on CoNLL04](#Evaluation-on-CoNLL04)
* [Pre-trained Models and Training Logs](#Pre-trained-Models-and-Training-Logs)
  * [Download Links](#Download-Links)
  * [Result Display](#Result-Display)
* [Extension on Ablation Study](#Extension-on-Ablation-Study)
* [Robustness Against Input Perturbation](#Robustness-Against-Input-Perturbation)
* [Citation](#Citation)


## Model Overview

### Framework

![](./fig/model.png)
In this work, we present a new framework equipped with a novel recurrent encoder named **partition
filter encoder** designed for multi-task learning.


### Equation Explanation

The explanation for equation 2 and 3 is displayed here.
![](./fig/gate.png)
![](./fig/partition.png)


## Preparation

### Environment Setup
The experiments were performed using one single NVIDIA-RTX3090 GPU. The dependency packages can be installed with the command:
```
pip install -r requirements.txt
```
Other configurations we use are:  
* python == 3.7.10
* cuda == 11.1
* cudnn == 8


### Data Acquisition and Preprocessing
This is the first work that covers all the mainstream English datasets for evaluation, including **NYT**, **WebNLG**, **ADE**, **ACE05**, **ACE04**, **SCIERC**, **CoNLL04**. 

Please follow the instructions of reademe.md in each dataset folder in ./data/ for data acquisition and preprocessing.  

### Custom Dataset
We suggest that you use **PFN-nested** for other datasets, especially Chinese datasets.  
**PFN-nested** is an enhanced version of PFN. It is better in leveraging entity tail information and capable of handling nested triple prediction.

**---Reasons for Not Using the Original Model**

The orignal one will not be able to decode **triples with head-overlap entities**. For example, if **New York** and **New York City** are both entities, and there exists a RE prediction such as (New, cityof, USA), we cannot know what **New** corresponds to.  

Luckily, the impact on evaluation of English dataset is limited, since such triple is either filtered out (for ADE) or rare (one in test set of SciERC, one in ACE04, zero in other datasets).  

**---Usage**

Replace the files (except for readme.md) in the root directory with the files in the PFN-nested folder, then follow the directions in Quick Start. 





## Quick Start


### Model Training
The training command-line is listed below (command for CONLL04 is in [Evaluation on CoNLL04](#Evaluation-on-CoNLL04)):  
```
python main.py \
--data ${NYT/WEBNLG/ADE/ACE2005/ACE2004/SCIERC} \
--do_train \
--do_eval \
--embed_mode ${bert_cased/albert/scibert} \
--batch_size ${20 (for most datasets) /4 (for SCIERC)} \
--lr ${0.00002 (for most datasets) /0.00001 (for SCIERC)} \
--output_file ${the name of your output files, e.g. ace_test} \
--eval_metric ${micro/macro} 
```

After training, you will obtain three files in the ./save/${output_file}/ directory:     
  * **${output_file}.log** records the logging information.  
  * **${output_file}.txt** records loss, NER and RE results of dev set and test set for each epoch.  
  * **${output_file}.pt** is the saved model with best average F1 results of NER and RE in the dev set.  


### Evaluation on Pre-trained Model

The evaluation command-line is listed as follows:

```
python eval.py \
--data ${NYT/WEBNLG/ADE/ACE2005/ACE2004/SCIERC} \
--eval_metric ${micro/macro} \
--model_file ${the path of saved model you want to evaluate. e.g. save/ace_test.pt} \
--embed_mode ${bert_cased/albert/scibert}
```

### Inference on Customized Input

If you want to evaluate the model with customized input, please run the following code:  

```
python inference.py \
--model_file ${the path of your saved model} \
--sent ${sentence you want to evaluate, str type restricted}
```

**model_file** must contain two kinds of keywords:
* The dataset the model trained on - (web, nyt, ade, ace, sci)
* Pretrained embedding the model uses - (albert, bert, scibert)

For example, model_file could be set as "web_bert.pt"  
 
  
**---Example**

```
input:
python inference.py \
--model_file save/sci_test_scibert.pt \
--sent "In this work , we present a new framework equipped with a novel recurrent encoder   
        named partition filter encoder designed for multi-task learning ."

result:
entity_name: framework, entity type: Generic
entity_name: recurrent encoder, entity type: Method
entity_name: partition filter encoder, entity type: Method
entity_name: multi-task learning, entity type: Task
triple: recurrent encoder, Used-for, framework
triple: recurrent encoder, Part-of, framework
triple: recurrent encoder, Used-for, multi-task learning
triple: partition filter encoder, Hyponym-of, recurrent encoder
triple: partition filter encoder, Used-for, multi-task learning



input:  
python inference.py \
--model_file save/ace_test_albert.pt \
--sent "As Williams was struggling to gain production and an audience for his work in the late 1930s ,  
        he worked at a string of menial jobs that included a stint as caretaker on a chicken ranch in   
        Laguna Beach , California . In 1939 , with the help of his agent Audrey Wood , Williams was 
        awarded a $1,000 grant from the Rockefeller Foundation in recognition of his play Battle of 
        Angels . It was produced in Boston in 1940 and was poorly received ."



## Evaluation on CoNLL04
We also run the test on the dataset CoNLL04, and our model surpasses previous SoTA table-sequence in micro/macro RE by 1.4%/0.9%.  

but we did not report the results in our paper due to several reasons:  
* Since the experiment setting is very confusing, we are unsure that the baseline results are reported in the same way as we did. The problems are discussed in detail in [Let's Stop Incorrect Comparisons in End-to-end Relation Extraction!](https://arxiv.org/pdf/2009.10684.pdf).
* Hyper-parameter tuning affects the performance considerably in this dataset.
* Page limits



The command for running CoNLL04 is listed below:

```
python main.py \
--data CONLL04 \
--do_train \
--do_eval \
--embed_mode albert \
--batch_size 10 \
--lr 0.00002 \
--output_file ${the name of your output files} \
--eval_metric micro \
--clip 1.0 \
--epoch 200
```
## Citation
Please cite our paper if it's helpful to you in your research.






