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
    # if os.path.isfile(args.save_model_path + '/sort_scores_test.pkl'):
    #     return
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

    facts_preds_dict = {}
    for i, j in zip(all_facts_keys, facts_preds):
        facts_preds_dict[i] = j

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
    test_recall_top2000 = pd.read_pickle('data/test_top_2000.pkl')
    scores = []
    for index,(pred, data) in enumerate(zip(val_preds, test_dataset)):
        query_id = test_dataset.__getitem__(index)
        recall_ids = test_recall_top2000[query_id][:2000]
        facts_pred_recall = [facts_preds_dict[rid] for rid in recall_ids]
        facts_pred_recall = torch.stack(facts_pred_recall, dim=0)
        score = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/sort_scores_test.pkl')

def model_val_predict(args, model):
    # if os.path.isfile(args.save_model_path + '/sort_scores_val.pkl'):
    #     return
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

    facts_preds_dict = {}
    for i, j in zip(all_facts_keys, facts_preds):
        facts_preds_dict[i] = j
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
    val_recall_top2000 = pd.read_pickle('data/val_top_2000.pkl')

    for index, (pred, data) in enumerate(zip(val_preds, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        recall_ids = val_recall_top2000[query_id][:2000]
        facts_pred_recall = [facts_preds_dict[rid] for rid in recall_ids]
        facts_pred_recall = torch.stack(facts_pred_recall, dim=0)
        score = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/sort_scores_val.pkl')





def main_predict():
    args = get_argparse()
    dir_paths = [
        'save_model/sort',
    ]

    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            args.output_dir = params['output_dir']
            args.bert_path = params['bert_path']
            args.save_model_path = args.output_dir
            model = TripletModel(
                bert_model=args.bert_path,
            )

            args.device = torch.device("cuda")

            save_model_path = os.path.join(args.output_dir,'model_best.bin')
            if not os.path.isfile(save_model_path):
                continue
            state_dict = torch.load(save_model_path)
            model.load_state_dict(state_dict)
            model.cuda()
            model_val_predict(args,model)
            model_test_predict(args, model)


def get_result_val():
    dir_paths = [
        'save_model/sort',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/sort_scores_val.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/sort_scores_val.pkl')
            merge_scores.append(scores)
    print(len(merge_scores))
    merge_scores = np.mean(merge_scores, axis=0)
    print(merge_scores.shape)
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/val_predict_tem.txt', 'w')
    val_top_2000 = defaultdict(list)
    val_recall_top2000 = pd.read_pickle('data/val_top_2000.pkl')
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        recall_ids = val_recall_top2000[query_id]

        indices = scores.argsort()
        recall_subject_ids = [recall_ids[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    score = eval_ndcg('data/val_predict_tem.txt')
    print('score',score)


def get_result_test():
    dir_paths = [
        'save_model/sort',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            scores = pd.read_pickle(params['output_dir'] + '/sort_scores_test.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    test_examples = get_test_examples()
    test_dataset = BertDataset(test_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('result/predict.txt', 'w')

    test_recall_top2000 = pd.read_pickle('data/test_top_2000.pkl')

    for index, (scores, data) in enumerate(zip(merge_scores, test_dataset)):
        query_id = test_dataset.__getitem__(index)

        recall_ids = test_recall_top2000[query_id]

        indices = scores.argsort()
        recall_subject_ids = [recall_ids[index] for index in indices]

        for recall_id in recall_subject_ids[:2000]:
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()


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

    parser.add_argument("--output_dir", default="/data/pan/models/naacl_sort/bert-base-uncased", type=str)
    parser.add_argument("--bert_path", default="/data/pan/embedding/pytorch/english/bert-base-uncased", type=str, )
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
    print(0)
    main_predict()
    get_result_val()
    get_result_test()


