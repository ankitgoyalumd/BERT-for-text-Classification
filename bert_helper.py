from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
config = transformers.BertConfig()

config.vocab_size = tokenizer.vocab_size


class disasterdataset(Dataset):
    def __init__(self, tweet, target, tokenizer, max_length):
        self.tweet = tweet
        self.target = target
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        encoding = tokenizer.encode_plus(
        tweet,
        max_length= self.max_length,
        add_special_tokens=True,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_token_type_ids = False,
        truncation= True,
        return_tensors = 'pt')

        return {
            'review_text' : tweet,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.target[item], dtype = torch.long)
        }

class disaster_classify(nn.Module):
    def __init__(self, n_classes, model_name):

        super ( disaster_classify, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, input_ids, attention_mask):
       # print(torch.max(input_ids))
        #print(torch.min(input_ids))
        ##print(input_ids)
        #print(torch.min(attention_mask))
        #print(torch.max(attention_mask))
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)
