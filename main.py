import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
from tqdm import tqdm

def train():
    export_root = setup_train(args)
    print("//////////////////")
  
    # tqdm_dataloader = tqdm(train_loader)
    #
    # for batch_idx, batch in enumerate(tqdm_dataloader):
    #     print(batch_idx, batch[0].size(),batch[1].size())
    #     batch_size = batch[0].size(0)
    # print(train_loader,val_loader,test_loader)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    # test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    # if test_model:
    trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
