
import numpy as np
import torch
import os
class LRWarmup(object):
    '''
    Bert模型内定的学习率变化机制
    Example:
    '''

    def __init__(self, optimizer,num_warmup_steps):

        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, training_step):
        if training_step<self.num_warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = lr * (training_step / self.num_warmup_steps)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,model=None, patience=2,mode='min', verbose=False,min_delta=0, restore_best_weights=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model = model
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.counter = 0
        self.best_score = self.best
        self.early_stop = False

    def epoch_step(self, current):
        '''
        正常模型
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        # 是否保存最好模型
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model

        if self.monitor_op(current, self.best):
            # logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
            self.best = current
            # Only save the model it-self
            torch.save(model_to_save.state_dict(), self.base_path)

            if self.model_recover:
                if os.path.exists(self.best_path):
                    os.remove(self.best_path)
                self.best_path = self.base_path + '.best'
                torch.save(self.model.state_dict(), self.best_path)
        else:
            if self.model_recover:
                self.model.load_state_dict(torch.load(self.best_path))

class ModelCheckpoint(object):

    def __init__(self,model, checkpoint_dir,

                 # monitor,
                 mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True,
                 model_recover = False
                 ):

        self.model = model
        self.base_path = checkpoint_dir
        self.best_path = self.base_path+'.best'

        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only
        self.model_recover = model_recover
        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

        # if save_best_only:
        #     self.model_name = f"BEST_{arch}_MODEL.pth"

    def epoch_step(self, current):
        '''
        正常模型
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        # 是否保存最好模型
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model

        if self.monitor_op(current, self.best):
            # logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
            self.best = current
             # Only save the model it-self
            torch.save(model_to_save.state_dict(), self.base_path)

            if self.model_recover:
                if os.path.exists(self.best_path):
                    os.remove(self.best_path)
                self.best_path = self.base_path+'.best'
                torch.save(self.model.state_dict(), self.best_path)
        else:
            if self.model_recover:
                self.model.load_state_dict(torch.load(self.best_path))
