'''Class with basic trainer class for NLP models'''
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .utils import LOG_INFO


class NLPTrainer(Trainer):
    '''Class with basic trainer class for NLP models'''

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: dict[str, float] = None,
        callbacks=None,
        optimizers=None,
        **kwargs,
    ):
        '''Class with basic trainer class for NLP models'''
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.args = args

    def compute_metrics(self, p: EvalPrediction) -> dict:
        '''Compute metrics for NLP models'''
        preds = np.argmax(p.predictions, axis=1)
        return {'accuracy': (preds == p.label_ids).mean()}

    def _log(self, logs: dict) -> None:
        '''Log metrics for NLP models'''
        if self.args.local_rank not in [-1, 0]:
            return
        for key, value in logs.items():
            if key in ['eval_loss', 'eval_accuracy']:
                LOG_INFO(f'{key} = {value:.3f}')
            else:
                LOG_INFO(f'{key} = {value}')

    def _save_model(self, output_dir: str) -> None:
        '''Save model for NLP models'''
        if self.args.local_rank in [-1, 0]:
            self.tokenizer.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)

    def _save_metrics(self, output_dir: str) -> None:
        '''Save metrics for NLP models'''
        if self.args.local_rank in [-1, 0]:
            self.log_history.to_csv(f'{output_dir}/log_history.csv')

    def _save(self, output_dir: str) -> None:
        '''Save model and metrics for NLP models'''
        self._save_model(output_dir)
        self._save_metrics(output_dir)

    def _load_model(self, output_dir: str) -> None:
        '''Load model for NLP models'''
        if self.args.local_rank not in [-1, 0]:
            return
        self.tokenizer = self.tokenizer.from_pretrained
        self.model = self.model.from_pretrained(output_dir)

    def _load_metrics(self, output_dir: str) -> None:
        '''Load metrics for NLP models'''
        if self.args.local_rank not in [-1, 0]:
            return
        self.log_history = pd.read_csv(f'{output_dir}/log_history.csv')
