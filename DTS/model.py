import argparse
from transformers import BertTokenizer
import re
import torch
# from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForNextSentencePrediction, AdamW, BertConfig, AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from load_dataset import DTSDataset

def MarginRankingLoss(p_scores, n_scores):
    margin = 1
    scores = margin - p_scores + n_scores
    scores = scores.clamp(min=0)

    return scores.mean()

device = 0
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
coherence_prediction_decoder = []
coherence_prediction_decoder.append(nn.Linear(768, 768))
coherence_prediction_decoder.append(nn.ReLU())
coherence_prediction_decoder.append(nn.Dropout(p=0.1))
coherence_prediction_decoder.append(nn.Linear(768, 2)) # label : 2개 positive, negtive
coherence_prediction_decoder = nn.Sequential(*coherence_prediction_decoder)
coherence_prediction_decoder.to(device)

tokenizer = AutoTokenizer.from_pretrained('klue/bert-base', do_lower_case=True)
model = BertForNextSentencePrediction.from_pretrained("klue/bert-base", num_labels = 2, output_attentions = False, output_hidden_states = True)
model.cuda(device)
optimizer = AdamW(list(model.parameters())+list(coherence_prediction_decoder.parameters()), lr = 2e-5, eps = 1e-8)

sample_num_memory = []
id_inputs = []

#for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_sample_num.txt'):
# for line in open('/ubc/cs/research/nlp/Linzi/dailydial/dailydial_sample_num.txt'):
#     line = line.strip()
#     sample_num_memory.append(int(line))

PATH = '/opt/ml/input/data/poc/KakaoTalk_Chat_IT개발자 구직:채용 정보교류방 (비번 2186)_2023-01-11-12-07-28.csv'
df = pd.read_csv(PATH)

print('The group number is: '+ str(len(df)))
# generate pos/neg pairs ....
print('start generating pos and neg pairs ... ')
# def collate_fn(batch):
#     print('batch', batch)
#     data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
#     # pos_ids, neg_ids = item['input_ids'] for item in batch
#     pos_mask, neg_mask = batch['attention_mask']
#     labels = batch['label']
#     pos_ids = data_collator({'input_ids': pos_ids})['input_ids']
#     neg_ids = data_collator({'input_ids': neg_ids})['input_ids']

#     return {'input_ids' : [pos_ids, neg_ids],
#                 'attention_mask' : [pos_mask,neg_mask],
#                 'label' : labels}
####################### 여기부터 미완 ###################################
# TODO : Sampling 방법 고안해보기
# 
# pos_neg_pairs = []; pos_neg_masks = []
# for i in range(len(dataset)):
#     if len(dataset[i]) == 2:
#         pos_neg_pairs.append(dataset[i])
#         pos_neg_masks.append(dataset[i])
#     else:
#         pos_neg_pairs.append([dataset[i][0], dataset[i][1]])
#         pos_neg_pairs.append([dataset[i][0], dataset[i][2]])
#         pos_neg_pairs.append([dataset[i][1], dataset[i][2]])
#         pos_neg_masks.append([dataset[i][0], dataset[i][1]])
#         pos_neg_masks.append([dataset[i][0], dataset[i][2]])
#         pos_neg_masks.append([dataset[i][1], dataset[i][2]])

# print('there are '+str(len(pos_neg_pairs))+' samples been generated...')
fake_labels = [0]*len(df)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(df, fake_labels, random_state=42, test_size=0.8)
# Do the same for the masks.
train_dataset = DTSDataset(train_inputs,tokenizer)
valid_dataset = DTSDataset(validation_inputs,tokenizer)
print(f'train_len : {len(train_dataset)}, valid_len : {len(valid_dataset)} now loading ...')
data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
batch_size = 16
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,collate_fn=data_collator)
# Create the DataLoader for our validation set.

validation_sampler = SequentialSampler(valid_dataset)
validation_dataloader = DataLoader(valid_dataset, sampler=validation_sampler, batch_size=batch_size,collate_fn=data_collator)


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

    for step, batch in tqdm(enumerate(train_dataloader),desc = 'train_step', total = len(train_dataset)//batch_size):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        pos_input_ids = batch['input_ids'].to(device)
        pos_input_mask = batch['attention_mask'].to(device)
        neg_input_ids = batch['neg_input_ids'].to(device)
        neg_input_mask = batch['neg_attention_mask'].to(device)
        # pos_input_ids, neg_input_ids = batch['input_ids'][0].to(device), batch['input_ids'][1].to(device)
        # pos_input_mask, neg_input_mask = batch['attention_mask'][0].to(device), batch['attention_mask'][1].to(device)

        model.zero_grad()
        coherence_prediction_decoder.zero_grad()

        pos_scores = model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0] #batch_size, sequence_length, hidden_size
        pos_scores = pos_scores[:,0,:]
        pos_scores = coherence_prediction_decoder(pos_scores)

        neg_scores = model(input_ids = neg_input_ids, attention_mask=neg_input_mask).hidden_states[0] #batch_size, sequence_length, hidden_size
        neg_scores = neg_scores[:,0,:]
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

    for step, batch in tqdm(enumerate(validation_dataloader),desc = 'validation_step',total = len(valid_dataset)//batch_size):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        pos_input_ids = batch['input_ids'].to(device)
        pos_input_mask = batch['attention_mask'].to(device)
        neg_input_ids = batch['neg_input_ids'].to(device)
        neg_input_mask = batch['neg_attention_mask'].to(device)
        # pos_input_ids, neg_input_ids = batch['input_ids'][0].to(device), batch['input_ids'][1].to(device)
        # pos_input_mask, neg_input_mask = batch['attention_mask'][0].to(device), batch['attention_mask'][1].to(device)

        with torch.no_grad():
            pos_scores = model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0]
            pos_scores = pos_scores[:,0,:]
            pos_scores = coherence_prediction_decoder(pos_scores)

            neg_scores = model(input_ids =neg_input_ids, attention_mask=neg_input_mask).hidden_states[0]
            neg_scores = neg_scores[:,0,:]
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

    print('label must be',sum(labels)/float(len(all_pos_scores)))

    '''
    PATH = '/scratch/linzi/bert_'+str(epoch_i)
    torch.save(model.state_dict(), PATH)
    '''
    #model.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')
    #tokenizer.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')