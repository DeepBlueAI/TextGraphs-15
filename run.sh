CUDA_VISIBLE_DEVICES=0,1 python recall_trainer.py --output_dir=save_model/recall/roberta --bert_path=roberta-large
CUDA_VISIBLE_DEVICES=0,1 python recall_trainer.py --output_dir=save_model/recall/ernie --bert_path=ernie-2.0-large-en
CUDA_VISIBLE_DEVICES=0 python recall_predict.py
CUDA_VISIBLE_DEVICES=0,1 python sort_trainer.py --output_dir=save_model/sort/roberta --bert_path=roberta-large
CUDA_VISIBLE_DEVICES=0,1 python sort_trainer.py --output_dir=save_model/sort/ernie --bert_path=ernie-2.0-large-en
CUDA_VISIBLE_DEVICES=0 python sort_predict.py
