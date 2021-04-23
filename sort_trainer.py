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
from evaluate import eval_ndcg
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
TOKENIZER = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
}
11,2,4,6
def get_optimizers(args, model, num_training_steps: int):
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion * num_training_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler



def model_eval(args, model, val_dataset):
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
        if torch.cuda.device_count() > 1:
            anchor_out = model.module.BERTModel(**inputs).detach()
        else:
            anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)

    facts_preds_dict = {}
    for i,j in zip(all_facts_keys,facts_preds):
        facts_preds_dict[i] = j

    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForTest(tokenizer, mode='val'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        if torch.cuda.device_count() > 1:
            anchor_out = model.module.BERTModel(**inputs).detach()
        else:
            anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)
    val_predict = open(args.save_model_path +'/val_predict.txt','w')
    val_recall_top2000 = pd.read_pickle('data/val_top_2000.pkl')
    for index,(pred, data) in enumerate(zip(val_preds, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        recall_ids = val_recall_top2000[query_id]
        facts_pred_recall = [facts_preds_dict[rid] for rid in recall_ids]
        facts_pred_recall = torch.stack(facts_pred_recall, dim=0)
        scores = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
        indices = scores.argsort()[:2000]
        recall_subject_ids = [recall_ids[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_predict.write('{}\t{}\n'.format(query_id,recall_id))
    val_predict.close()
    score = eval_ndcg(args.save_model_path +'/val_predict.txt')
    logger.info('val score: {0}'.format(str(score)))
    return {'score': score}



def train(args, model, train_dataset, eval_dataset=None, data_collator=None):
    if args.local_rank != -1:
        args.n_gpu = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.batch_size = args.per_gpu_batch_size
        torch.distributed.init_process_group(backend="nccl")
        logger.info('local rank: {0}'.format(torch.distributed.get_rank()))
        args.num_workers = max(int(args.batch_size / 4), 1)

    else:
        args.n_gpu = torch.cuda.device_count()
        args.batch_size = args.n_gpu * args.per_gpu_batch_size
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.num_workers = max(int(args.batch_size / 4), 1)

    train_sampler = (RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset))
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=data_collator)

    t_total = int(len(train_dataloader) * args.num_train_epochs)

    model.to(args.device)

    optimizer, scheduler = get_optimizers(args, model, t_total)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  learning rate = %f", args.learning_rate)
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", args.per_gpu_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  num workers = %d", args.num_workers)
    if args.do_adv:
        fgm = FGM(model, emb_name='word_embeddings.', )
    train_iterator = tqdm(range(args.num_train_epochs), desc="Epoch")
    eval_scores = {}
    model_checkpoint_score = ModelCheckpoint(model, args.save_model_path + '/model_best.bin',
                                          mode="max")
    # model_test(args, model, 0)
    # score = model_eval(args, model, eval_dataset)
    # model_eval(args, model, eval_dataset)
    for epoch_index in train_iterator:
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", )
        for batch_index, inputs in enumerate(epoch_iterator):

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)
            outputs = model(**inputs)
            loss = outputs
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()

            if args.do_adv:
                fgm.attack()
                outputs = model(**inputs)
                loss_adv = outputs
                if torch.cuda.device_count() > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # if args.local_rank in [-1, 0]:
        #     if epoch_index>0:
        #         model_to_save = model.module if hasattr(model, 'module') else model
        #         torch.save(model_to_save.state_dict(),
        #                    os.path.join(args.save_model_path, 'pytorch_model.bin' + str(epoch_index)))

        if args.do_eval and args.local_rank in [-1, 0]:
            logger.info('val eval')
            # model_test(args, model,epoch_index)
            score = model_eval(args, model, eval_dataset)
            eval_scores[epoch_index] = score
            model_checkpoint_score.epoch_step(score['score'])
    return eval_scores


def main():
    args = get_argparse()
    bert_path = args.bert_path
    logger.info('main')
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isfile(os.path.join(args.output_dir, 'scores.json')):
        return
    init_logger(os.path.join(args.output_dir, 'log.txt'))
    args_dict = vars(args)
    print(args_dict)
    json.dump(args_dict, open(args.output_dir + '/args.json', 'w'))

    if 'roberta' in bert_path:
        mode_type = 'roberta'
    else:
        mode_type = 'bert'
    tokenizer = TOKENIZER[mode_type].from_pretrained(args.bert_path)
    scores = {}
    train_examples = get_train_examples(args.relevance_int)
    val_examples = get_dev_examples()
    train_dataset = BertDataset(examples=train_examples)
    eval_dataset = BertDataset(examples=val_examples)
    logger.info('train dataset len: {0}'.format(len(train_dataset)))
    logger.info('val dataset len: {0}'.format(len(eval_dataset)))
    model = TripletModel(
        bert_model=bert_path,
    )
    data_collator = DataCollatorForTrainSort(tokenizer)
    args.save_model_path = args.output_dir
    eval_scores = train(args, model, train_dataset, eval_dataset, data_collator)





def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2021, type=int,
                        help="")
    parser.add_argument('--num_train_epochs', default=15, type=int)
    parser.add_argument("-per_gpu_batch_size", default=8, type=int,
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
    parser.add_argument('--relevance_int', default=5, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--bert_path", default="", type=str, )
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
    main()
    # CUDA_VISIBLE_DEVICES=0,1 python sort_trainer.py --output_dir=save_model/sort/roberta --bert_path=/home/panchg/embedding/roberta-base

    '''
    mlm acc,f1 : 0.3422150755243601 0.1364769099050118
    mlm acc,f1 : 0.32729968512121305 0.13474287074221317
    mlm acc,f1 : 0.3212296492926155 0.13913180514054874


    '''
