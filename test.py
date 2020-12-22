import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.loss import HDRLoss
from utils.HDRutils import tonemap
from dataset.HDR import KalantariTestDataset
from models.DeepHDR import DeepHDR
from utils.configs import Configs


# Get configurations
configs = Configs()

# Load dataset
test_dataset = KalantariTestDataset(configs=configs)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Build DeepHDR model from configs
model = DeepHDR(configs)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), betas=(configs.beta, 0.999), lr=configs.learning_rate)

# Define Criterion
criterion = HDRLoss()

# Read checkpoints
checkpoint_file = configs.checkpoint_dir + '/checkpoint.tar'
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Load checkpoint %s", checkpoint_file)
else:
    raise ModuleNotFoundError('No checkpoint files.')


def eval_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(test_dataloader):
        in_LDRs, in_HDRs, in_exps, ref_HDRs = data
        # Forward
        with torch.no_grad():
            res = model(in_LDRs, in_HDRs)

        # Compute loss
        with torch.no_grad():
            loss = criterion(tonemap(res), tonemap(ref_HDRs))

        print('--------------- Test Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.detach().numpy())
        mean_loss += loss.detach().numpy()
        count += 1

    mean_loss = mean_loss / count
    return mean_loss


def eval():
    loss = eval_one_epoch()
    print('mean eval loss: %.12f' % loss)


if __name__ == '__main__':
    eval()
