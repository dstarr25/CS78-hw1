from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE

# Specify the lead_data arguments
data_path = "xor_dataset.pt"
mean_subtraction = False
normalization = False

xor_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
in_features = 2
out_size = 2
hidden_units = [3]
non_linearity = ['tanH']

# create a network base on the architecture
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
train_opts = {
    "num_epochs": 20, # used 20 because it took around 18 epochs to get to 100% accuracy
    "lr": 0.5,
    "momentum": 0.9,
    "batch_size": 4,
    "weight_decay": 0,
    "step_size": 15,
    "gamma": 1,
}

# train and save the model with base options
train(net, xor_dataset, train_opts)
save(net, 'xor_solution.pt')
