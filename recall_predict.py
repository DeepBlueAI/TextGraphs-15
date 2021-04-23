import argparse
import logging
import os
import random
import time
import numpy as np
import torch
import itertools
from typing import Any, Callable, Dict, List, NewType, Tuple
from tqdm.auto import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_pt_utils import SequentialDistributedSampler
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.data.data_collator import *
from transformers.data.datasets import *
from dataset import *
from sklearn.model_selection import KFold
from models.models import *
from models.adversarial import FGM
from sklearn import metrics
import logging
from models.callbacks import *
import torch.nn.functional as F
logger = logging.getLogger()
from evaluate import eval_ndcg,eval_ndcg_train
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
TOKENIZER = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
}

def model_test_predict(args, model):
    if os.path.isfile(args.save_model_path + '/recall_scores_test.pkl'):
        return
    test_examples = get_test_examples()
    test_dataset = BertDataset(test_examples)
    model.eval()
    if 'roberta' in args.bert_path:
        mode_type = 'roberta'
    else:
        mode_type = 'bert'
    tokenizer = TOKENIZER[mode_type].from_pretrained(args.bert_path)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    facts_preds = []
    fact_loader = DataLoader(all_facts_keys, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)
    dev_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForTest(tokenizer, mode='test'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)

    scores = []
    for index,(pred, data) in enumerate(zip(val_preds, test_dataset)):
        query_id = test_dataset.__getitem__(index)
        score = F.pairwise_distance(pred, facts_preds, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/recall_scores_test.pkl')

def model_train_predict(args, model):
    # if os.path.isfile(args.save_model_path + '/recall_scores_train.pkl'):
    #     return
    train_examples = get_train_predict_examples()

    train_dataset = BertDataset(train_examples)
    model.eval()
    if 'roberta' in args.bert_path:
        mode_type = 'roberta'
    else:
        mode_type = 'bert'
    tokenizer = TOKENIZER[mode_type].from_pretrained(args.bert_path)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    facts_preds = []
    fact_loader = DataLoader(all_facts_keys, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)
    dev_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForTest(tokenizer, mode='train'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)
    scores = []
    for index, (pred, data) in enumerate(zip(val_preds, train_dataset)):

        score = F.pairwise_distance(pred, facts_preds, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/recall_scores_train.pkl')

def model_val_predict(args, model):
    if os.path.isfile(args.save_model_path + '/recall_scores_val.pkl'):
        return
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    model.eval()
    if 'roberta' in args.bert_path:
        mode_type = 'roberta'
    else:
        mode_type = 'bert'
    tokenizer = TOKENIZER[mode_type].from_pretrained(args.bert_path)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    facts_preds = []
    fact_loader = DataLoader(all_facts_keys, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForTest(tokenizer, mode='val'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)
    scores = []
    for index, (pred, data) in enumerate(zip(val_preds, val_dataset)):

        score = F.pairwise_distance(pred, facts_preds, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/recall_scores_val.pkl')





def main_predict():
    args = get_argparse()

    dir_paths = [
        'save_model/recall',
    ]
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            print(params)
            args.output_dir = params['output_dir']
            args.bert_path = params['bert_path']
            args.save_model_path = args.output_dir
            model = TripletModel(
                bert_model=args.bert_path,
            )
            args.device = torch.device("cuda")
            save_model_path = os.path.join(args.output_dir,'model_best.bin')
            state_dict = torch.load(save_model_path)
            model.load_state_dict(state_dict)
            model.cuda()
            model_val_predict(args,model)
            model_test_predict(args, model)
            model_train_predict(args, model)



def get_result_train():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_train.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_train.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    print(merge_scores.shape)
    train_examples = get_train_predict_examples()
    val_dataset = BertDataset(train_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/predict_tem.txt', 'w')
    val_top_2000 = defaultdict(list)
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    score = eval_ndcg_train('data/predict_tem.txt')

    pd.to_pickle(val_top_2000,'data/train_top_2000.pkl')


def get_result_val():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_val.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_val.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/val_predict_tem.txt', 'w')
    idtopositives = get_idtopositives_val()
    val_top_2000 = defaultdict(list)
    totals = 0
    preds_num = 0
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        totals += len(idtopositives[query_id])
        for recall_id in recall_subject_ids:
            if recall_id in idtopositives[query_id]:
                preds_num+=1
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    print(preds_num/totals)
    score = eval_ndcg('data/val_predict_tem.txt')

    pd.to_pickle(val_top_2000,'data/val_top_2000.pkl')


def get_result_test():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_test.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_test.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    val_examples = get_test_examples()
    val_dataset = BertDataset(val_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/test_predict2000.txt', 'w')
    val_top_2000 = defaultdict(list)
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        for recall_id in recall_subject_ids:

            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    pd.to_pickle(val_top_2000,'data/test_top_2000.pkl')

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2021, type=int,
                        help="")
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument("--per_gpu_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training or evaluation.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training or evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Bert.")
    parser.add_argument("--lr", default=5e-4, type=float,
                        help="The initial learning rate")
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--do_eval", default=True, type=bool, )

    parser.add_argument("--do_adv", default=True, type=bool)
    parser.add_argument('--dropout_num', default=1, type=int)
    parser.add_argument('--num_hidden_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout_prob1', default=0.2, type=float)
    parser.add_argument('--dropout_prob2', default=0.1, type=float)

    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument("--bert_path", default='', type=str, )
    '''
    bert-base-uncased
    ernie-2.0-base-en
    roberta-base
    bert-large-uncased
    roberta-large
    ernie-2.0-large-en

    '''
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    main_predict()
    get_result_val()
    get_result_test()
    get_result_train()
    '''
    mlm acc,f1 : 0.3422150755243601 0.1364769099050118

    mlm acc,f1 : 0.32729968512121305 0.13474287074221317
    mlm acc,f1 : 0.3212296492926155 0.13913180514054874


    '''
