import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from os.path import join as osjoin
from os.path import exists as osexists
from os import listdir
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.preprocessing import LabelBinarizer

def predict_OGlcNAcylation(sequence, threshold = 0.5, batch_size = 2048, shapley = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Sequence data preparation
    if osexists(sequence): # User passed a .csv with sequences
        sequence_path = sequence
        sequence = pd.read_csv(sequence).values.squeeze()
    # One-hot encoding the sequences
    OHE_seq = torch.Tensor(OHE_for_LSTM(np.atleast_1d(np.array(sequence)), 20))
    seq_dataloader = DataLoader(OHE_seq, batch_size, shuffle = False)

    # Model preparation
    if 'RNN-[600,75]_20-window_dict.pt' in listdir():
        mydict = torch.load(osjoin('RNN-[600,75]_20-window_dict.pt'), map_location = torch.device(device)) # New model with higher performance from "Predicting O-GlcNAcylation sites in mammalian proteins with transformers and RNNs trained with a new loss function" work
    else:
        mydict = torch.load(osjoin('RNN-225_20-window_dict.pt'), map_location = torch.device(device)) # Old model from "Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins" work; kept for compatibility
    layers = []
    for array_name, array in mydict.items(): # Getting the size of the model from mydict
        if 'weight' in array_name:
            layers.append(tuple(array.T.shape))
    # Building the model
    if len(layers) == 10: # Model with two RNN cells
        model = SequenceMLP(layers[8:], 'relu', [layers[1][0], layers[5][0]])
    else: # Model with one RNN cell
        model = SequenceMLP(layers[4:], 'relu', layers[1][0])
    model.load_state_dict(mydict)
    model.to(device)
    model.eval()
    # Making predictions
    pred = torch.empty((len(seq_dataloader.dataset), 2))
    for idx, data in enumerate(seq_dataloader):
        data = data.to(device)
        temp_pred = model(data).cpu().detach()
        pred[idx*batch_size:(idx*batch_size)+len(temp_pred), :] = temp_pred
    pred_bool = pred[:, 1] >= threshold
    if shapley: # Also do predictions using Shapley values
        import shap
        background = _make_background_shap().to(device)
        explainer = shap.DeepExplainer(model, background)
        shap_values = np.array(explainer.shap_values(OHE_seq.to(device), check_additivity = False))
        shap_unsummed = np.empty(OHE_seq.shape[:2]) # n_samples x 2*window_size+1
        shap_chance = np.empty(OHE_seq.shape[0])
        for idx in range(shap_chance.shape[0]):
            temp_sum = shap_values[0, idx, OHE_seq[idx].numpy().astype(bool)] # Real data, but may have fewer AAs than 2*window_size + 1
            shap_unsummed[idx, :] = np.concatenate(( temp_sum, np.zeros(OHE_seq.shape[1] - temp_sum.shape[0]) ))
            shap_chance[idx] = shap_unsummed[idx, :].sum()
        shap_unsummed = pd.DataFrame(shap_unsummed, columns = list(range(-20, 21)))
    # Saving or outputting the predictions
    if 'sequence_path' in locals(): # Multiple sequences from a .csv file
        if 'shap_chance' not in locals(): # User didn't ask for Shapley value predictions
            shap_chance = []
        else:
            shap_unsummed.to_csv(''.join(sequence_path.split('.')[:-1]) + '_Shapley_values.csv', index = False)
        output = np.concatenate((sequence, pred[:, 1], pred_bool, shap_chance)).reshape(-1, len(sequence)).T
        output = pd.DataFrame(output, columns = ['Sequence', 'Predicted O-GlcNAcylation Chance', f'Chance >= {threshold}', 'Shapley value Chance'][:output.shape[1]])
        output.to_csv(''.join(sequence_path.split('.')[:-1]) + '_predictions.csv', index = False)
    else: # One prediction
        if 'shap_chance' not in locals(): # User didn't ask for Shapley value predictions
            shap_chance = 'N/A'
        else:
            shap_chance = f'{shap_chance[0]:.4e}'
        print(f'Sequence {sequence} | Predicted O-GlcNAcylation Chance = {pred[0, 1].item():.4e} | Chance >= {threshold}: {pred_bool.item()} | Shapley value Chance = {shap_chance}')
        if 'shap_unsummed' in locals():
            print(f'Unsummed shap coefficients: {shap_unsummed.iloc[0].values}')

def _make_background_shap(bg_size = 4000):
    """
    A helper function called automatically when making Shapley value predictions. It shouldn't be called by the user
    """
    X_data = torch.Tensor(np.load('OH_LSTM_data_v5_20-window.npy'))
    y_data = pd.read_csv('OH_data_v5_5-window.csv').squeeze().values
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 123)
    rng = np.random.default_rng(123)
    bg_idx = rng.choice(X_data.shape[0], bg_size, replace = False)
    bg_idx = list(bg_idx[y_data[bg_idx] == 0]) # Select only negative entries for the background. Converting to list to do some appends below
    while len(bg_idx) < bg_size:
        temp_idx = rng.choice(X_data.shape[0])
        if y_data[temp_idx] == 0 and temp_idx not in bg_idx:
            bg_idx.append(temp_idx)
    bg_idx = np.array(bg_idx)
    background = X_data[bg_idx]
    return background

# MLP or LSTM+MLP model
class SequenceMLP(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu', lstm_hidden_size = 0):
        super(SequenceMLP, self).__init__()
        # Setup to convert string to activation function
        if activ_fun == 'relu':
            torch_activ_fun = torch.nn.ReLU()
        elif activ_fun == 'tanh':
            torch_activ_fun = torch.nn.Tanh()
        elif activ_fun == 'sigmoid':
            torch_activ_fun = torch.nn.Sigmoid()
        elif activ_fun == 'tanhshrink':
            torch_activ_fun = torch.nn.Tanhshrink()
        elif activ_fun == 'selu':
            torch_activ_fun = torch.nn.SELU()
        #elif activ_fun == 'attention':
        #    torch_activ_fun = torch.nn.MultiheadAttention(myshape_X, 4)
        else:
            raise ValueError(f'Invalid activ_fun. You passed {activ_fun}')

        # LSTM cell
        if isinstance(lstm_hidden_size, int) and lstm_hidden_size:
            self.lstm = [torch.nn.LSTM(20, lstm_hidden_size, batch_first = True, bidirectional = True).cuda()] # Need to send to device because the cells are in a list
        elif isinstance(lstm_hidden_size, (list, tuple)):
            self.lstm = []
            for idx, size in enumerate(lstm_hidden_size):
                if idx == 0:
                    self.lstm.append(torch.nn.LSTM(20, size, batch_first = True, bidirectional = True).cuda()) # Need to send to device because the cells are in a list
                else:
                    self.lstm.append(torch.nn.LSTM(lstm_hidden_size[idx-1], size, batch_first = True, bidirectional = True).cuda()) # Need to send to device because the cells are in a list
        self.lstm = torch.nn.ModuleList(self.lstm) # Need to transform list into a ModuleList so PyTorch updates and interacts with the weights properly
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if 'lstm' in dir(self):
            for cell in self.lstm:
                x, (ht, _) = cell(x)
                x = (x[:, :, :x.shape[2]//2] + x[:, :, x.shape[2]//2:]) / 2 # Average between forward and backward
            to_MLP = (ht[0] + ht[1]) / 2 # Average between forward and backward
            out = self.model(to_MLP)
        else:
            out = self.model(x)
        probs = self.sigmoid(out)
        probs = (probs.T / probs.sum(axis=1)).T # Normalizing the probs to 1
        return probs

def OHE_for_LSTM(sequences, window = 10):
    lb = LabelBinarizer()
    _ = lb.fit(('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'))
    sequence_size = 2*window + 1 # window AAs before + central AA + window AAs after
    X = np.empty(( sequences.shape[0], sequence_size, 20 ), dtype = np.uint8) # sequences.shape[0] AA sequences, 2*window + 1 AAs per protein sequence, 20 nucleotides in one-hot enconding per site

    for idx, elem in enumerate(sequences):
        if not(idx % 5000):
            print(f'OHE current idx: {idx:6,}/{sequences.shape[0]-1:,} (Updated every 5000 idx)', end = '\r')
        # list(elem) separates the AAs into one list with 2*window + 1 elements [instead of a single string]
        # The lb.transform() then creates a (2*window + 1)x20 array
        temp = lb.transform(list(elem))
        # Some sequences have less than 2*window + 1 elements because they are close to the beginning or end of the protein
        if temp.shape[0] < sequence_size:
            temp = np.concatenate([temp, np.zeros([sequence_size - temp.shape[0], 20]) ])
        X[idx, :, :] = temp

    return X

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads a trained RNN model and predicts the presence of O-GlcNAcylation sites based on protein sequence.')
    parser.add_argument('sequence', type = str, nargs = 1, help = 'The protein sequence of the site (15 AAs on each side + the central S/T) OR a .csv file with a header and the site sequences.')
    parser.add_argument('-t', '--threshold', metavar = '0.5', nargs = 1, type = float, default = [0.5], help = 'The minimum prediction threshold for a site to be considered O-GlcNAcylated. Optional, default = 0.5.')
    parser.add_argument('-bs', '--batch_size', metavar = '2048', nargs = 1, type = int, default = [2048], help = 'The number of predictions done at a time. Lower only if getting out of memory errors. Optional, default = 2048.')
    parser.add_argument('-shap', '--shapley_values', metavar='True | [False]', type = bool, nargs = '?', default = False, const = True, help = 'Whether to also make predictions using Shapley values, which are interpretable.' +
                        'This is slower than the default of not using Shapley. Optional, default = False')
    args = parser.parse_args()
    sequence = args.sequence[0] # [0] to convert from list to string
    threshold = args.threshold[0]
    batch_size = args.batch_size[0]
    shap = args.shapley_values
    predict_OGlcNAcylation(sequence, threshold, batch_size, shap)
