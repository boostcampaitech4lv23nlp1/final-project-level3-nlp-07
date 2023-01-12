import argparse
import random
from transformers import BertTokenizer
import re
import torch
# from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset
from transformers import BertForNextSentencePrediction, AdamW, BertConfig
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import pandas as pd


def MarginRankingLoss(p_scores, n_scores):
    margin = 1
    scores = margin - p_scores + n_scores
    scores = scores.clamp(min=0)

    return scores.mean()

device = 0
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('klue/bert-base', do_lower_case=True)

sample_num_memory = []
id_inputs = []

#for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_sample_num.txt'):
# for line in open('/ubc/cs/research/nlp/Linzi/dailydial/dailydial_sample_num.txt'):
#     line = line.strip()
#     sample_num_memory.append(int(line))
class DTSDataset(Dataset):
    def __init__(self,path,tokenizer) -> None:
        super(DTSDataset,self).__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.dataset = self._tokenizing(self.data)
        self.label = [1]
    
    def _tokenizing(self,df):
        output =[]
        for idx, item in df.iterrows():
            token = self.tokenizer(item['Message'],item['Message2'],add_special_tokens = True, max_length = 128, padding = 'max_length',return_tensors = 'pt')
            output.append(token)
        return output

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {'input_ids' : self.dataset[idx]['input_ids'],
                 'attention_mask' : self.dataset[idx]['attention_mask'],
                 'label' : torch.tensor(self.label)}
PATH = '/opt/ml/input/data/poc/KakaoTalk_Chat.csv'
dataset = DTSDataset(PATH,tokenizer)
print('The group number is: '+ str(len(dataset)))
# generate pos/neg pairs ....
print('start generating pos and neg pairs ... ')

####################### 여기부터 미완 ###################################
# TODO : Sampling 방법 고안해보기
# 
pos_neg_pairs = []; pos_neg_masks = []
for i in range(len(dataset)):
    if len(dataset[i]) == 2:
        pos_neg_pairs.append(dataset[i])
        pos_neg_masks.append(dataset[i])
    else:
        pos_neg_pairs.append([dataset[i][0], dataset[i][1]])
        pos_neg_pairs.append([dataset[i][0], dataset[i][2]])
        pos_neg_pairs.append([dataset[i][1], dataset[i][2]])
        pos_neg_masks.append([dataset[i][0], dataset[i][1]])
        pos_neg_masks.append([dataset[i][0], dataset[i][2]])
        pos_neg_masks.append([dataset[i][1], dataset[i][2]])

print('there are '+str(len(pos_neg_pairs))+' samples been generated...')
fake_labels = [0]*len(pos_neg_pairs)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(pos_neg_pairs, fake_labels, random_state=42, test_size=0.8)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(pos_neg_masks, fake_labels, random_state=42, test_size=0.8)


batch_size = 16
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our validation set.

validation_sampler = SequentialSampler(dataset)
validation_dataloader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch_size)

coherence_prediction_decoder = []
coherence_prediction_decoder.append(nn.Linear(768, 768))
coherence_prediction_decoder.append(nn.ReLU())
coherence_prediction_decoder.append(nn.Dropout(p=0.1))
coherence_prediction_decoder.append(nn.Linear(768, 2))
coherence_prediction_decoder = nn.Sequential(*coherence_prediction_decoder)
coherence_prediction_decoder.to(device)

model = BertForNextSentencePrediction.from_pretrained("klue/bert-base", num_labels = 2, output_attentions = False, output_hidden_states = True)
model.cuda(device)
optimizer = AdamW(list(model.parameters())+list(coherence_prediction_decoder.parameters()), lr = 2e-5, eps = 1e-8)

epochs = 10
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


for epoch_i in range(0, epochs):

    total_loss = 0

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0

    model.train()
    coherence_prediction_decoder.train()

    for step, batch in enumerate(train_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        model.zero_grad()
        coherence_prediction_decoder.zero_grad()

        pos_scores = model(b_input_ids[:,0,:], attention_mask=b_input_mask[:,0,:])
        pos_scores = pos_scores[1][-1][:,0,:]
        pos_scores = coherence_prediction_decoder(pos_scores)

        neg_scores = model(b_input_ids[:,1,:], attention_mask=b_input_mask[:,1,:])
        neg_scores = neg_scores[1][-1][:,0,:]
        neg_scores = coherence_prediction_decoder(neg_scores)

        #loss = MarginRankingLoss(pos_scores[0][:,0], neg_scores[0][:,0])
        loss = MarginRankingLoss(pos_scores[:,0], neg_scores[:,0])

        total_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(coherence_prediction_decoder.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print('=========== the loss for epoch '+str(epoch_i)+' is: '+str(avg_train_loss))

    print("")
    print("Running Validation...")

    model.eval()
    coherence_prediction_decoder.eval()

    all_pos_scores = []
    all_neg_scores = []

    for step, batch in enumerate(validation_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        with torch.no_grad():
            pos_scores = model(b_input_ids[:,0,:], attention_mask=b_input_mask[:,0,:])
            pos_scores = pos_scores[1][-1][:,0,:]
            pos_scores = coherence_prediction_decoder(pos_scores)
            neg_scores = model(b_input_ids[:,1,:], attention_mask=b_input_mask[:,1,:])
            neg_scores = neg_scores[1][-1][:,0,:]
            neg_scores = coherence_prediction_decoder(neg_scores)

        #all_pos_scores += pos_scores[0][:,0].detach().cpu().numpy().tolist()
        #all_neg_scores += neg_scores[0][:,0].detach().cpu().numpy().tolist()
        all_pos_scores += pos_scores[:,0].detach().cpu().numpy().tolist()
        all_neg_scores += neg_scores[:,0].detach().cpu().numpy().tolist()

    labels = []

    for i in range(len(all_pos_scores)):
        if all_pos_scores[i] > all_neg_scores[i]:
            labels.append(1)
        else:
            labels.append(0)

    print(sum(labels)/float(len(all_pos_scores)))

    '''
    PATH = '/scratch/linzi/bert_'+str(epoch_i)
    torch.save(model.state_dict(), PATH)
    '''
    #model.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')
    #tokenizer.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')