import datasets
import torch
from datasets import load_dataset

def main():
  dataset = load_dataset("wmt14",'de-en', cache_dir=f'{_current_path}/data/')
  print(dataset)
  print('>> train dataset sample')
  for row in dataset['train']:
    print(row)
    break
    
if __name__ == '__main__':
  print('Run!')
  
