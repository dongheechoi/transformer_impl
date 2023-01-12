
import os
import logging
import datasets
import torch
from datasets import load_dataset
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import sys

## ref : https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    dh : default is changed
    """

    source_lang: str = field(default='de', metadata={"help": "Source language id for translation."})
    target_lang: str = field(default='en', metadata={"help": "Target language id for translation."})
    dataset_name: Optional[str] = field(
        default='wmt14', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default='de-en', metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments,Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

raw_datasets = load_dataset(
    data_args.dataset_name,
    data_args.dataset_config_name,
    cache_dir=model_args.cache_dir,
    use_auth_token=True if model_args.use_auth_token else None,
)
dataset = load_dataset("wmt14",'de-en', cache_dir=f'{_current_path}/data/')
print(dataset)
print('>> train dataset sample')
for row in dataset['train']:
    print(row)
    break
