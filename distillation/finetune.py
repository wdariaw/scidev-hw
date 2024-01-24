import argparse
import torch
import pytorch_lightning as pl
from transformers import BertTokenizer

from data_modules import GLUEDataModule
from utils import seed_everything
from models import GLUEClassifier


def main(config):
    seed_everything(42)
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    data_module = GLUEDataModule(config, tokenizer)
    model = GLUEClassifier(config)
    trainer = pl.Trainer(max_epochs=config.epochs, accelerator="gpu", devices=1)
    trainer.fit(model, data_module)
    trainer.test(
        model,
        datamodule=data_module
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune distilled BERT.'
    )

    parser.add_argument('--student_hidden_size', type=int, default=312)
    parser.add_argument('--student_intermediate_size', type=int, default=1200)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--student_num_hidden_layers', type=int, default=4)
    parser.add_argument('--glue_cls_task', type=str, default='cola')
    parser.add_argument('--student_weights_path', type=str, default=None)
    parser.add_argument('--finetune_distilled', action='store_true')

    args = parser.parse_args()
    
    main(args)