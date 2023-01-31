import argparse
from transformers import BertTokenizer
import re
import torch
import os
# from keras.preprocessing.sequence import pad_sequences
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from collections import OrderedDict
import logging
from utils import TqdmLoggingHandler
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())
def MarginRankingLoss(p_scores, n_scores):
    margin = 1
    scores = margin - p_scores + n_scores
    scores = scores.clamp(min=0)

    return scores.mean()
logger.info(f'The accuracy for this validation  :%')
device = 'cuda:0'
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
class CSModel(nn.Module):
    def __init__(self,pretrained_id, model_config = None) -> None:
        super(CSModel, self).__init__()
        if model_config == None:
            model_config = AutoConfig.from_pretrained(pretrained_id)
        self.hidden_size = model_config.hidden_size
        self.plm = AutoModel.from_pretrained(pretrained_id, add_pooling_layer = False)
        self.cs_model = nn.Sequential(OrderedDict({'Linear' : nn.Linear(self.hidden_size, self.hidden_size),
                                        'Active_fn' : nn.ReLU(),
                                        'Dropout' : nn.Dropout(p=0.1),
                                        'cls_layer' : nn.Linear(self.hidden_size, 1),
                                        'Active_fn' : nn.Tanh(),}))
    def forward(self,input):
        required_keys = ['input_ids', 'attention_mask', 'neg_input_ids', 'neg_attention_mask']
        if not all(key in input for key in required_keys):
            raise KeyError("Input is missing required keys: {}".format(required_keys))

        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        neg_input_ids = input['neg_input_ids']
        neg_attention_mask = input['neg_attention_mask']
        pos_scores = self.plm(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).last_hidden_state
        neg_scores = self.plm(
            input_ids = neg_input_ids,
            attention_mask = neg_attention_mask
        ).last_hidden_state
        pos_scores = self.cs_model(pos_scores[:,0,:])
        neg_scores = self.cs_model(neg_scores[:,0,:])

        return {'output' : {'pos' : pos_scores, 'neg' : neg_scores}}
    
# coherence_prediction_decoder = CSModel().to(device)

# tokenizer = AutoTokenizer.from_pretrained('klue/bert-base', do_lower_case=True)
# model = BertForNextSentencePrediction.from_pretrained("klue/bert-base", num_labels = 2, output_attentions = False, output_hidden_states = True)
# model.cuda(device)

# sample_num_memory = []
# id_inputs = []

# PATH = '/opt/ml/input/data/dialogue/train.csv'
# df = pd.read_csv(PATH)

# # generate pos/neg pairs ....
# print('The group number is: '+ str(len(df)))
# print('start generating pos and neg pairs ... ')
# train_inputs = df
# validation_inputs = pd.read_csv('/opt/ml/input/data/dialogue/valid.csv')

# train_dataset = trainDataset(train_inputs,tokenizer)
# valid_dataset = trainDataset(validation_inputs,tokenizer)
# print(f'train_len : {len(train_dataset)}, valid_len : {len(valid_dataset)} now loading ...')
# data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
# batch_size = 16
# train_sampler = RandomSampler(train_dataset)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,collate_fn=data_collator)
# # Create the DataLoader for our validation set.

# validation_sampler = SequentialSampler(valid_dataset)
# validation_dataloader = DataLoader(valid_dataset, sampler=validation_sampler, batch_size=batch_size,collate_fn=data_collator)


# epochs = 10
# # Total number of training steps is number of batches * number of epochs.
# total_steps = len(train_dataloader) * epochs
# # Create the learning rate scheduler.
# optimizer = optim.AdamW([
#             {'params': model.parameters()},
#             {'params': coherence_prediction_decoder.parameters(), 'lr': 5e-5},
#                 ], lr=2e-5,eps = 1e-8)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=3, eta_min=1e-7)
# # optimizer = AdamW(list(model.parameters())+list(coherence_prediction_decoder.parameters()), lr = 2e-5, eps = 1e-8)
# # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
# print(f'Total len for training set {len(train_dataloader)}')
# print(f'Total len for validation set {len(validation_dataloader)}')
# min_loss = []
# max_acc = []
# for epoch_i in range(0, epochs):

#     total_loss = 0

#     print("")
#     print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#     print('Training...')

#     total_loss = 0

#     model.train()
#     coherence_prediction_decoder.train()

#     for step, batch in tqdm(enumerate(train_dataloader),desc = 'train_step', total = len(train_dataset)//batch_size):

#         if step % 1000 == 0 and not step == 0:
#             logger.info(str(step)+' steps done....')

#         pos_input_ids = batch['input_ids'].to(device)
#         pos_input_mask = batch['attention_mask'].to(device)
#         neg_input_ids = batch['neg_input_ids'].to(device)
#         neg_input_mask = batch['neg_attention_mask'].to(device)
#         # pos_input_ids, neg_input_ids = batch['input_ids'][0].to(device), batch['input_ids'][1].to(device)
#         # pos_input_mask, neg_input_mask = batch['attention_mask'][0].to(device), batch['attention_mask'][1].to(device)

#         model.zero_grad()
#         coherence_prediction_decoder.zero_grad()

#         pos_scores = model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0] #batch_size, sequence_length, hidden_size
#         pos_scores = pos_scores[:,0,:]
#         pos_scores = coherence_prediction_decoder(pos_scores)

#         neg_scores = model(input_ids = neg_input_ids, attention_mask=neg_input_mask).hidden_states[0] #batch_size, sequence_length, hidden_size
#         neg_scores = neg_scores[:,0,:]
#         neg_scores = coherence_prediction_decoder(neg_scores)

#         #loss = MarginRankingLoss(pos_scores[0][:,0], neg_scores[0][:,0])
#         loss = MarginRankingLoss(pos_scores[:,0], neg_scores[:,0])
#         if step % 200 == 0 and not step == 0:
#             logger.info(f'log for loss in {step} steps : [{loss}]')
#         total_loss += loss.item()
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(coherence_prediction_decoder.parameters()), 1.0)
#         optimizer.step()
#         scheduler.step()

#     avg_train_loss = total_loss / len(train_dataloader)
#     logger.info('=========== the loss for epoch '+str(epoch_i)+' is: '+str(avg_train_loss))
#     min_loss.append(avg_train_loss)
#     print("")
#     print("Running Validation...")

#     model.eval()
#     coherence_prediction_decoder.eval()

#     all_pos_scores = []
#     all_neg_scores = []

#     for step, batch in tqdm(enumerate(validation_dataloader),desc = 'validation_step',total = len(valid_dataset)//batch_size):

#         if step % 1000 == 0 and not step == 0:
#             logging.info(str(step)+' steps done....')

#         pos_input_ids = batch['input_ids'].to(device)
#         pos_input_mask = batch['attention_mask'].to(device)
#         neg_input_ids = batch['neg_input_ids'].to(device)
#         neg_input_mask = batch['neg_attention_mask'].to(device)
#         # pos_input_ids, neg_input_ids = batch['input_ids'][0].to(device), batch['input_ids'][1].to(device)
#         # pos_input_mask, neg_input_mask = batch['attention_mask'][0].to(device), batch['attention_mask'][1].to(device)

#         with torch.no_grad():
#             pos_scores = model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0]
#             pos_scores = pos_scores[:,0,:]
#             pos_scores = coherence_prediction_decoder(pos_scores)

#             neg_scores = model(input_ids =neg_input_ids, attention_mask=neg_input_mask).hidden_states[0]
#             neg_scores = neg_scores[:,0,:]
#             neg_scores = coherence_prediction_decoder(neg_scores)
#         all_pos_scores += pos_scores[:,0].detach().cpu().numpy().tolist()
#         all_neg_scores += neg_scores[:,0].detach().cpu().numpy().tolist()

#     labels = []

#     for i in range(len(all_pos_scores)):
#         if all_pos_scores[i] > all_neg_scores[i]:
#             labels.append(1)
#         else:
#             labels.append(0)
#     logger.info(f'The accuracy for this validation  : {sum(labels)/len(labels)*100}%')
#     max_acc.append(sum(labels)/len(labels)*100)

#     # PATH = '/scratch/linzi/bert_'+str(epoch_i)
#     # torch.save(model.state_dict(), PATH)
#     model.save_pretrained('/opt/ml/input/poc/BERT/bert_'+str(epoch_i + 1)+'/')
#     torch.save(coherence_prediction_decoder.state_dict(),'/opt/ml/input/poc/CS/cs' + str(epoch_i +1) + '.pt')
# logger.info(f'training is finished and there is min_loss happend in {min_loss.index(min(min_loss)) + 1} epochs')
# logger.info(f'training is finished and there is max_acc happend in {max_acc.index(min(max_acc)) + 1} epochs')