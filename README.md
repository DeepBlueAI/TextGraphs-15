## Multi-Hop Inference Explanation Regeneration (TextGraphs-15)



This is the code for the first place in the textgraphs-15 competition

paper ：DeepBlueAI at TextGraphs 2021 Shared Task: Treating Multi-HopInference Explanation Regeneration as A Ranking Problem

### requirement

1. pytorch=1.6
2. transformers=4.5.1
3. pandas=1.2.3
4. cuda=10.1
5. python=3.8.5



### pre-training model

**roberta-large**

https://huggingface.co/roberta-large/tree/main

**ernie-2.0-large-en**

https://huggingface.co/nghuyong/ernie-2.0-large-en/tree/main



### run the code

**recall train**

```bash
CUDA_VISIBLE_DEVICES=0,1 python recall_trainer.py --output_dir=save_model/recall/roberta --bert_path=roberta-large
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python recall_trainer.py --output_dir=save_model/recall/ernie --bert_path=ernie-2.0-large-en
```

**recall predict**

```bash
CUDA_VISIBLE_DEVICES=0 python recall_predict.py
```

**sort train**

```bash
CUDA_VISIBLE_DEVICES=0,1 python sort_trainer.py --output_dir=save_model/sort/roberta --bert_path=roberta-large
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python sort_trainer.py --output_dir=save_model/sort/ernie --bert_path=ernie-2.0-large-en
```

**sort predict**

```bash
CUDA_VISIBLE_DEVICES=0 python sort_predict.py
```



### result

The result is  "result/predict.txt"
