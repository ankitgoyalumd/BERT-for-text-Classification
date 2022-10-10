import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import bert_helper
from bert_helper import disasterdataset, disaster_classify

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
config = transformers.BertConfig()
config.vocab_size = tokenizer.vocab_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_train_val_data(file_path, split_frac):
    data = pd.read_csv(file_path)
    data = data[['text', 'target']]
    df_train, df_val = train_test_split(data, test_size = split_frac, random_state=42)
    return df_train, df_val

Max_len = 512
Epochs = 20


def create_dataloader(df, tokenizer, max_length, batch_size):
    ds = disasterdataset(
        tweet = df['text'].to_numpy(),
        target = df['target'].to_numpy(),
        tokenizer = tokenizer,
        max_length = max_length
        )
    
    return DataLoader(
        ds,
        batch_size = batch_size,
        num_workers = 0)


df_train, df_val = get_train_val_data(file_path= './data/train.csv', split_frac = 0.01)

train_dataloader = create_dataloader(df_train, tokenizer, 70, 16)
val_dataloader = create_dataloader(df_val, tokenizer, 70, 16)

#data = next(iter(train_dataloader))

Pre_trained_model_name = 'bert-base-cased'

def model(model_name):
    bert_model = BertModel.from_pretrained(model_name)
    return bert_model

bert_model = model(Pre_trained_model_name)

model = disaster_classify(2, Pre_trained_model_name)
model.to(device)

## Training

optimizer = AdamW(model.parameters(), lr = 2e-5, correct_bias = False)

total_steps  = len(train_dataloader)*Epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    scheduler,
    n_examples,
    device
    ) :
        model = model.train()
        losses = []
        correct_predictions = 0

        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            #print(attention_mask.size())
            #print(input_ids.size())
           # print(d['review_text'])
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )

            _, pred = torch.max(outputs, dim =1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(pred == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return correct_predictions/n_examples, np.mean(losses)


#evaluate model

def model_eval(model, data_loader, loss_fn, device, examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            #print(attention_mask.size())
            #print(input_ids.size())
           # print(d['review_text'])
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )

            _, pred = torch.max(outputs, dim =1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(pred == targets)
            losses.append(loss.item())
    return correct_predictions/examples, np.mean(losses)






history = defaultdict(list)
best_accuracy = 0
count = 0
for epoch in range(Epochs):

    print(f'Epoch {epoch + 1}/{Epochs}')
    print('-'*10)
   # print(df_train.shape[0])
    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        scheduler,
        df_train.shape[0],
        device
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = model_eval(
        model,
        val_dataloader,
        loss_fn,
        device,
        df_val.shape[0]
    )
    print(f'Val loss {val_loss} accuracy {val_acc}')
    print(f'end of epoch # {epoch}')
    history['train_accuracy'].append(train_acc)
    history['train_loss'].append(train_loss)

    history['val_accuracy'].append(val_acc)
    history['val_loss'].append(val_loss)

    if epoch%10 >= count:
        model_name = f'{model}_{val_acc}_{epoch}.pth'
        torch.save(model, model_name)
        count +=1
    



        


            
