import re
import random
import numpy as np
import pytorch_lightning as pl
from collections import defaultdict
from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from data import MaskedLMDataset, GLUEClassificationDataset


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('@.*?\s+', '', text)
    text = re.sub('#.*?\s+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    text = text.strip()
    return text


class DataModule(pl.LightningDataModule):
    def __init__(self, params, tokenizer, collate_fn):
        super(DataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

        self.train_dataset=None
        self.validation_dataset=None
        self.predict_dataset=None
        
        self.train_samples = []
        self.val_samples = []
        self.inference_samples = []
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument('--train_data_path', type=str, default=None)
        parser.add_argument('--inference_data_path', type=str, default=None)
        return parent_parser

    def prepare_data(self):
        if self.params.train_data_path:
            with open(self.params.train_data_path, 'r') as f:
                lines = f.read().splitlines()
                
            random.seed(42)
            texts = random.choices(lines, k=500000)
            del lines

            texts = [preprocess(t) for t in texts]

            train_set, val_set = train_test_split(texts, test_size=0.1, random_state=42)

            for sample in train_set:
                self.train_samples.append(self._encode(sample))

            for sample in val_set:
                self.val_samples.append(self._encode(sample))

        if self.params.inference_data_path:
            with open(self.params.inference_data_path, 'r') as f:
                inference_samples = f.read().splitlines()
            
            for sample in inference_samples:
                self.inference_samples.append(self._encode(sample, True))

    def _encode(self, sample, inference=False):
        return self.tokenizer(
                    text=sample,
                    max_length=self.params.max_length,
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True,
                    return_tensors='pt' if inference else None
                )

    def setup(self, stage=None):
        if self.params.train_data_path:
            self.train_dataset = MaskedLMDataset(
                self.train_samples
            )

            self.val_dataset = MaskedLMDataset(
                self.val_samples
            )
        
        if self.params.inference_data_path:
            self.predict_dataset = MaskedLMDataset(
                self.inference_samples
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.params.num_workers,
        )
        

class GLUEDataModule(pl.LightningDataModule):
    def __init__(self, params, tokenizer):
        super(GLUEDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer

        self.samples = defaultdict(list)

    def prepare_data(self):
        ds_initial = load_from_disk(f'{self.params.glue_cls_task}.hf')
        ds_train_val = ds_initial['train'].train_test_split(test_size=0.1, stratify_by_column="label")
        
        ds = DatasetDict({
            'train': ds_train_val['train'],
            'validation': ds_train_val['test'],
            'test': ds_initial['validation']
        })
        
        for split in ['train', 'validation', 'test']:
            df = ds[split].to_pandas()
            df['sentence'] = df['sentence'].apply(lambda x: preprocess(x))
            
            for sample, label in df[['sentence', 'label']].values:
                encoded_dict = {key: val.squeeze() for key, val in self._encode(sample).items()}
                encoded_dict['labels'] = torch.Tensor([0, 0])
                encoded_dict['labels'][label] = 1
                self.samples[split].append(encoded_dict)
    
    def _encode(self, sample):
        return self.tokenizer(
                    text=sample,
                    max_length=self.params.max_length,
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True,
                    return_tensors='pt'
                )

    def setup(self, stage=None):
        self.train_dataset = GLUEClassificationDataset(
            self.samples['train']
        )

        self.val_dataset = GLUEClassificationDataset(
            self.samples['validation']
        )
        
        self.test_dataset = GLUEClassificationDataset(
            self.samples['test']
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
        
        