import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.loss import HDRLoss
from utils.HDRutils import tonemap
from utils.dataprocessor import dump_sample
from dataset.HDR import KalantariTestDataset
from models.DeepHDR import DeepHDR
from utils.configs import Configs


# Get configurations
configs = Configs()
# configs = Configs(data_path='/Users/galaxies/Documents/Benchmark/kalantari_dataset')


# Load dataset
test_dataset = KalantariTestDataset(configs=configs)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Build DeepHDR model from configs
model = DeepHDR(configs)
if configs.multigpu is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)
    model = torch.nn.DataParallel(model)

# Define optimizer
optimizer = optim.Adam(model.parameters(), betas=(configs.beta1, configs.beta2), lr=configs.learning_rate)

# Define Criterion
criterion = HDRLoss()

# Read checkpoints
checkpoint_file = configs.checkpoint_dir + '/checkpoint.tar'
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Load checkpoint %s (epoch %d)", checkpoint_file, start_epoch)
else:
    raise ModuleNotFoundError('No checkpoint files.')


def test_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(test_dataloader):
        sample_path, in_LDRs, in_HDRs, in_exps, ref_HDRs = data
        sample_path = sample_path[0]
        in_LDRs = in_LDRs.to(device)
        in_HDRs = in_HDRs.to(device)
        ref_HDRs = ref_HDRs.to(device)
        # Forward
        with torch.no_grad():
            res = model(in_LDRs, in_HDRs)

        # Compute loss
        with torch.no_grad():
            loss = criterion(tonemap(res), tonemap(ref_HDRs))

        dump_sample(sample_path, res.cpu().detach().numpy())

        print('--------------- Test Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1

    mean_loss = mean_loss / count
    return mean_loss


def test():
    loss = test_one_epoch()
    print('mean test loss: %.12f' % loss)


if __name__ == '__main__':
    test()
