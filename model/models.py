# coding=utf-8
import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, ElectraModel
import torch.autograd as autograd
from torch.autograd import Variable

PRETRAINED_MODEL = {
    'bert': BertModel,
    'roberta': RobertaModel,
}
class BERTModel(nn.Module):

    def __init__(self, bert_model=None):
        super(BERTModel, self).__init__()
        if 'roberta' in bert_model:
            self.model_type = 'roberta'
        else:
            self.model_type = 'bert'
        self.bert = PRETRAINED_MODEL[self.model_type].from_pretrained(bert_model)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        h_embedding, pooled = outputs[0],outputs[1]
        out_mean = torch.mean(h_embedding, dim=1)
        return out_mean

class TripletModel(nn.Module):
    def __init__(self, bert_model=None):
        super(TripletModel, self).__init__()
        self.BERTModel = BERTModel(bert_model)
        self.loss_fct = torch.nn.TripletMarginLoss(margin=3)
    def forward(self, anchor_ids,anchor_mask,positive_ids,positive_mask,negative_ids,negative_mask):
        anchor_out = self.BERTModel(anchor_ids, anchor_mask)
        positive_out = self.BERTModel(positive_ids, positive_mask)
        negative_out = self.BERTModel(negative_ids, negative_mask)
        loss = self.loss_fct(anchor_out, positive_out, negative_out)
        return loss

