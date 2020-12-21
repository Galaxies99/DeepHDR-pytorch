import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.solvers import PolyLR
from utils.loss import HDRLoss
from utils.HDRutils import tonemap
from dataset.HDR import KalantariDataset
from models.DeepHDR import DeepHDR
from utils.configs import Configs


# Get configurations
configs = Configs()

# Load Data & build dataset
train_dataset = KalantariDataset(configs=configs)
train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

# test_dataset = KalantariDataset(configs=configs)
# test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=True)


# Build DeepHDR model from configs
model = DeepHDR(configs)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

# Define Criterion
criterion = HDRLoss()

# Read checkpoints
start_epoch = 0
checkpoint_file = configs.checkpoint_dir + '/checkpoint.tar'
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Load checkpoint %s (epoch %d)", checkpoint_file, start_epoch)
cur_epoch = start_epoch

# Learning rate scheduler
lr_scheduler = PolyLR(optimizer, max_iter=configs.epoch, power=0.9, last_step=start_epoch-1)


def train_one_epoch():
    model.train()
    for idx, data in enumerate(train_dataloader):
        in_LDRs, ref_LDRs, in_HDRs, ref_HDRs, in_exps, ref_exps = data
        # Forward
        result = model(in_LDRs, in_HDRs)
        # Backward
        loss = criterion(tonemap(result), tonemap(ref_HDRs))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('--------------- Train Batch %d ---------------' % (idx + 1))
        print('loss:', loss.detach().numpy())

'''
def eval_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(test_dataloader):
        # Forward
        with torch.no_grad():
            res = model(data)

        # Compute loss
        with torch.no_grad():
            loss = criterion(tonemap(res), tonemap(data['ref_HDR']))

        print('--------------- Eval Batch %d ---------------' % (idx + 1))
        print('loss:', loss)
        mean_loss += loss
        count += 1

    mean_loss = mean_loss / count
    return mean_loss
'''

def train(start_epoch):
    global cur_epoch
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, configs.epoch):
        cur_epoch = epoch
        print('**************** Epoch %d ****************' % (epoch + 1))
        print('learning rate: %f' % (lr_scheduler.get_last_lr()[0]))
        train_one_epoch()
        # loss = eval_one_epoch()
        lr_scheduler.step()
        save_dict = {'epoch': epoch + 1,
                     'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss,
                     'model_state_dict': model.state_dict()
                     }
        torch.save(save_dict, os.path.join(configs.checkpoint_dir, 'checkpoint.tar'))
        # print('mean eval loss: %.10f' % loss.detach().numpy())


if __name__ == '__main__':
    train(start_epoch)
