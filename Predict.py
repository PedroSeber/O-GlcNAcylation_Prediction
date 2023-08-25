import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from os.path import join as osjoin
from os.path import exists as osexists
from ANN_train import SequenceMLP
from one_hot_encode_csv import OHE_for_LSTM

def predict_OGlcNAcylation(sequence, threshold = 0.5, batch_size = 2048):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Sequence data preparation
    if osexists(sequence): # User passed a .csv with sequences
        sequence_path = sequence
        sequence = pd.read_csv(sequence).values.squeeze()
    # One-hot encoding the sequences
    OHE_seq = torch.Tensor(OHE_for_LSTM(np.atleast_1d(np.array(sequence)), 20))
    seq_dataloader = DataLoader(OHE_seq, batch_size, shuffle = False)

    # Model preparation
    mydict = torch.load(osjoin('RNN-225_20-window_dict.pt'), map_location = torch.device(device))
    # Getting the size of the model from mydict
    layers = []
    for array_name, array in mydict.items():
        if 'weight' in array_name:
            layers.append(tuple(array.T.shape))
    # Building the model
    model = SequenceMLP(layers[4:], 'relu', layers[1][0])
    model.load_state_dict(mydict)
    model.to(device)
    model.eval()
    # Making predictions
    pred = torch.empty((len(seq_dataloader.dataset), 2))
    for idx, data in enumerate(seq_dataloader):
        data = data.to(device)
        temp_pred = model(torch.empty(batch_size, 0), data).cpu().detach()
        pred[idx*batch_size:(idx*batch_size)+len(temp_pred), :] = temp_pred
    # Renormalizing the pred
    pred = (pred.T / pred.sum(axis=1)).T
    pred_bool = pred[:, 1] >= threshold
    if 'sequence_path' in locals():
        output = np.concatenate((sequence, pred[:, 1], pred_bool)).reshape(-1, len(sequence)).T
        output = pd.DataFrame(output, columns = ['Sequence', 'Predicted O-GlcNAcylation Chance', f'Chance >= {threshold}'])
        output.to_csv(''.join(sequence_path.split('.')[:-1]) + f'_predictions.csv', index = False)
    else:
        print(f'Sequence {sequence} | Predicted O-GlcNAcylation Chance = {pred[0, 1].item():.4e} | Chance >= {threshold}: {pred_bool.item()}')

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads a trained RNN model and predicts the presence of O-GlcNAcylation sites based on protein sequence.')
    parser.add_argument('sequence', type = str, nargs = 1, help = 'The protein sequence of the site (15 AAs on each side + the central S/T) OR a .csv file with a header and the site sequences.')
    parser.add_argument('-t', '--threshold', metavar = '0.5', nargs = 1, default = [0.5], help='The minimum prediction threshold for a site to be considered O-GlcNAcylated. Optional, default = 0.5.')
    parser.add_argument('-bs', '--batch_size', metavar = '2048', nargs = 1, default = [2048], help='The number of predictions done at a time. Lower only if getting out of memory errors. Optional, default = 2048.')
    args = parser.parse_args()
    sequence = args.sequence[0] # [0] to convert from list to string
    threshold = args.threshold[0]
    batch_size = args.batch_size[0]
    predict_OGlcNAcylation(sequence, threshold, batch_size)
