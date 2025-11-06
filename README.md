```python
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pyfaidx import Fasta
from torch.nn.functional import one_hot
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)
```


```python
# Read the train data, assuming you have downloaded them
# Here, I assumed you downloaded them into the same directory/folder as this notebook file
# However, if that is not the case, just modify the file paths accordingly

train_terms = pd.read_csv("train_terms.tsv", sep='\t')
seq = Fasta("train_sequences.fasta")
```


```python
# Run this cell to only use a small subnet of the data if needed (to quickly test things, etc.)
# Adjust the frac parameter to set the percentage of data you want

train_terms = train_terms.sample(frac=0.0001)
```


```python
## The entire point of this code cell is to encode the response/target variable (the GO terms) into a data format that ML algorithms can understand

# Collect unique GO terms in our training data
unique_terms = train_terms['term'].unique()

# Map GO terms to numeric values (machine learning only works with numbers, so anything non-numeric has to be converted at some point) and vice versa
id2label = {idx: term for idx, term in enumerate(unique_terms)}
label2id = {term: idx for idx, term in enumerate(unique_terms)}

# In the FASTA file, a sequence's key includes more than its EntryID in the train_terms.tsv file
# So we want to build a mapping from EntryID to the raw sequence for quick lookup
seqs = {seq[key].name.split('|')[1] : seq[key][:].seq for key in seq.keys()}

# Add the proteins' sequences to the training data
# These will be the feature/predictor/X that our model will learn to predict the target/response/y from
train_terms['seq'] = train_terms['EntryID'].map(lambda x: seqs[x])

# Since we want to learn multiple labels for each protein sequence, we need to associate each of them to a list of labels
# We also need to encode that list into an array of real numbers in order to feed it into our ML model (as mentioned above, ML only works with numbers and multidimensional objects (arrays, matrices, tensors) of numbers)
# In the original data, each protein-term pair is a single row, we want to "collapse" them into a single protein-(list of terms) row for each protein

# Create a label column that is constructed as follow:
# - For each value in the term column, find the integer assigned to it using our label2id mapping above
# - Encode that value into a 1-hot array. Read more about one-hot encoding here:
#   - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
#   - https://en.wikipedia.org/wiki/One-hot
train_terms['label'] = train_terms['term'].map(lambda x: one_hot(torch.tensor(label2id[x]), num_classes=unique_terms.size).numpy().astype(float))

# "Collapse" the many labels for each protein by summing them up
train_terms = train_terms.groupby('EntryID').agg({'label': 'sum', 'seq': 'first'})
```


```python
# You might have learned that machine learning (and deep learning) is about learning high-dimensional features of data
# Therefore, the very first step, conceptually, is "bringing" that data into a "space" that has enough dimensions to represent it
# Practically, this means choosing a numerical space with enough dimensions and encode our data into elements in it
# (because ML only works with- Ok, ok, I'll stop saying it)
# In the latest lingo, this is called tokenization (I swear it was called embedding last time)
# 
# For data that we have understood well, we can manually design rules to encode them
# For example, if you only need to classify codons, then you may be able to hard-code arrays to represent them (since there are only a fixed amount of them)
# But for data that we have not yet have a good understanding of, like proteins, we can also ask the machine to "understand" them for us (nothing can go wrong!)
# Here, we use Facebook AI's Evolutionary Scale Modelling model, which was trained to do exactly that
# (There are many other "protein language" models, but this one integrates well with the Transformers framework)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

def batch_tokenize(examples):
    return tokenizer(examples['seq'])

dataset = Dataset.from_pandas(train_terms[['seq', 'label']])
dataset = dataset.map(batch_tokenize, batched=True)

# After tokenizing, we split our dataset into a training set and a test set
dataset = dataset.train_test_split(test_size=0.1)
```


```python
# Each of these Tokenizer, AutoModel, TrainingArguments, and Trainer objects have a ton of parameters (see Transformers' documentation)
# But these are the absolute minimum required to successfully run a multiclass classification trainer
# It may perform terribly with these default params, but I want to give you a minimal template to work with
# (Many tutorials provide initial values like learning rate, epoch, etc. without much explanation. I hate that)

# We don't train a model from scratch, but only fine-tune Facebook's model for our data
model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/esm2_t6_8M_UR50D", problem_type="multi_label_classification", id2label=id2label, label2id=label2id
)
    
training_args = TrainingArguments(
    output_dir="./model",

    # These parameters are for displaying progress in the notebook
    # Transformers (the library) has some questionable software design choices
    disable_tqdm=False,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],

    # This is for resizing all inputs (numerical representations of protein sequences) in a traning batch into the same size, usually the size of the biggest sequence of that batch
    # Because...the algorithm requires it (to be most efficient)
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
```


```python
trainer.train()
```


```python
# After training (fine-tuning), the fine-tuned model will be saved in a "checkpoint-*" subdirectory of the "./model" directory
# We will use that for predicting GO terms for a protein now
# Edit the model parameter to point to the directory of the fine-tuned model
# The classifier, when given an amino acid sequence, will return a dictionary of GO terms and the probabilities that each of those terms are associated with said sequence
# The top_k parameter controls how many labels (GO terms) we want the classifier to return. "None" means all (that the model was fine-tuned on earlier)
classifier = pipeline(task="text-classification", model="./model/<checkpoint>", top_k=None)
```


```python
classifier('<some amino acid sequence>')
```
