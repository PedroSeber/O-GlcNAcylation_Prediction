# General & data manipulation imports
import pandas as pd
import numpy as np
from os import mkdir
from os.path import isdir, isfile
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
# Torch & model creation imports
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
# Training & validation imports
from itertools import product
# Results & visualization imports
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Convenience imports
import pdb
from time import time

def train_CV_complete(weight_hyperparam, activ_fun_list = ['relu'], lstm_size = 75, data_version = 'v5', window_size = 10, batch_size = 32):
    """
    Cross-validates and tests an MLP (lstm_size = 0) or RNN (lstm_size > 0) model on O-GlcNAcylation data. The data used depend on the data_version variable

    Parameters
    ----------
    weight_hyperparam : list with two elements of the form [1, number > 1]
        The weights to use in the loss function. The second number refers to the weight of the positive class, which should be > 1 because there are fewer positive samples
    activ_fun_list : list of strings, optional, default = ['relu']
        A list of strings representing the activation functions used during cross-validation. Check the class SequenceMLP to see what functions are available
    lstm_size : int, optional, default = 0
        The size of the LSTM layer. Set to 0 to use only an MLP
    data_version : string in the form 'v#', optional, default = 'v5'
        The version of the data to be used. Should be left as 'v5'
    window_size : int, optional, default = 10
        The number of AAs before and after the central S/T. Used only when data_version in {'v3', 'v5'}
    batch_size : int, optional, default = 32
        The batch size used during cross-validation
    """
    ### DATA SETUP ###
    if data_version in {'v1', 'v2'}:
        window_size_path = ''
        myshape_X = data.shape[1] - 1 # For convenience when declaring ANNs
    elif data_version in {'v3', 'v4'}:
        window_size_path = f'_{window_size}-window'
        myshape_X = 76 # Manually declaring, 76 because the v1 dataset had 76 features
    else:
        window_size_path = f'_{window_size}-window'
        myshape_X = 75 # Rounded the 76 to 75
    # Loading and transforming the data if using an LSTM
    if lstm_size:
        lstm_data = torch.Tensor(np.load(f'OH_LSTM_data_{data_version}{window_size_path}.npy'))
    data = torch.Tensor(pd.read_csv(f'OH_data_{data_version}{window_size_path}.csv').values) # This is used for all ANN types

    # Pre-declaring paths for convenience (to save / load results)
    if lstm_size:
        working_dir = f'RNN_{lstm_size}_results_{data_version}-data{window_size_path}'
    else:
        working_dir = f'ANN_results_{data_version}-data'
    if not isdir(working_dir):
        mkdir(working_dir)

    # Setting each activ_fun to lowercase for consistency
    activ_fun_list = [activ_fun.casefold() for activ_fun in activ_fun_list]

    # Data splitting - 80% Cross Validation, 20% Test
    if lstm_size:
        cv_data, test_data, cv_lstm_data, test_lstm_data = train_test_split(data, lstm_data, test_size = 0.2, random_state = 123)
    else:
        cv_data, test_data = train_test_split(data, test_size = 0.2, random_state = 123)

    ### MODEL AND RUN SETUP ###
    # Setting up the hyperparameters
    layers = [
        # 1 hidden layer
        #[(myshape_X, myshape_X*12), (myshape_X*12, 2)], ## Not in the MLP-only
        #[(myshape_X, myshape_X*11), (myshape_X*11, 2)], ## Not in the MLP-only
        #[(myshape_X, myshape_X*10), (myshape_X*10, 2)], ## Not in the MLP-only
        #[(myshape_X, myshape_X*9), (myshape_X*9, 2)], ## Not in the MLP-only CV
        #[(myshape_X, myshape_X*8), (myshape_X*8, 2)], ##
        [(myshape_X, myshape_X*7), (myshape_X*7, 2)], ##
        [(myshape_X, myshape_X*6), (myshape_X*6, 2)], ##
        [(myshape_X, myshape_X*5), (myshape_X*5, 2)], ## Not in the LSTM w/ Mauri CV
        #[(myshape_X, myshape_X*4), (myshape_X*4, 2)],
        #[(myshape_X, myshape_X*3), (myshape_X*3, 2)],
        #[(myshape_X, myshape_X*2), (myshape_X*2, 2)],
        #[(myshape_X, myshape_X), (myshape_X, 2)],
    ]
    #lr_vals = [1e-2, 5e-3, 1e-3, 5e-4]
    lr_vals = [1e-2, 5e-3]
    hyperparam_list = list(product(layers, lr_vals))
    # v1: There are 42'981 total points / 570 positive (1.33%) -> "natural" my_weight[1] = (42981-570)/570 = 74.4
    # v3: There are 41'600 total points / 535 positive (1.29%) -> (41600-535)/535 = 76.8
    my_weight = torch.Tensor(weight_hyperparam)
    my_loss = torch.nn.CrossEntropyLoss(weight = my_weight).cuda()

    ### TRAINING AND VALIDATING THE MODEL ###
    def CV_model(activ_fun, working_dir, F1_score_file, val_loss_file):
        """
        This function runs a cross-validation procedure for each combination of layers + learning rates
        Results are saved in a .csv file inside {working_dir}
        """
        # LSTM changes the configuration of the first layer. Thus, need to increase the ...
        # size of the 1st MLP layer to lstm_size
        if lstm_size:
            for cur_hp in hyperparam_list:
                cur_hp[0][0] = (lstm_size, cur_hp[0][0][1])
        # Recording the validation F1 scores and losses
        try:
            final_val_F1 = pd.read_csv(f'{working_dir}/{F1_score_file}', index_col = 0)
        except FileNotFoundError:
            final_val_F1 = pd.DataFrame(np.nan, index = lr_vals, columns = [str(elem) for elem in layers])
        try: # Separate try-except to ensure F1 records aren't overwritten if they exist w/o val-loss records
            final_val_loss = pd.read_csv(f'{working_dir}/{val_loss_file}', index_col = 0)
        except FileNotFoundError:
            final_val_loss = pd.DataFrame(np.nan, index = lr_vals, columns = [str(elem) for elem in layers])

        # Train and validate
        print(f'Beginning CV on activation function {activ_fun} (weight = {weight_hyperparam[1]})')
        for cur_idx, cur_hp in enumerate(hyperparam_list):
            # We added a new layer configuration to the hyperparameters
            if not str(cur_hp[0]) in list(final_val_F1.columns):
                final_val_F1.insert(layers.index(cur_hp[0]), str(cur_hp[0]), np.nan) # layers.index to ensure consistent order
            if not str(cur_hp[0]) in list(final_val_loss.columns):
                final_val_loss.insert(layers.index(cur_hp[0]), str(cur_hp[0]), np.nan)
            # We added a new learning rate to the hyperparameters
            if not cur_hp[1] in final_val_F1.index.to_list():
                final_val_F1.loc[cur_hp[1], :] = np.nan
            if not cur_hp[1] in final_val_loss.index.to_list():
                final_val_loss.loc[cur_hp[1], :] = np.nan

            # Run CV only if we do not have validation losses for this set of parameters
            if np.isnan( final_val_F1.at[cur_hp[1], str(cur_hp[0])] ):
                print(f'Beginning hyperparameters {cur_idx+1:2}/{len(hyperparam_list)} for {activ_fun}; layers = {cur_hp[0]}, lr = {cur_hp[1]}')
                temp_val_F1 = 0
                temp_val_loss = 0
                my_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)
                for fold_idx, (train_idx, val_idx) in enumerate(my_kfold.split(cv_data[:, :-1], cv_data[:, -1])):
                    print(f'Current fold: {fold_idx+1}/{my_kfold.n_splits}', end = '\r')
                    # Creating the Datasets
                    if lstm_size:
                        train_dataset_fold = MyDataset(cv_data[train_idx], cv_lstm_data[train_idx])
                        train_neg = torch.where(cv_data[train_idx] == 0)[0] # Idxs of negative training samples
                        train_pos = torch.where(cv_data[train_idx] == 1)[0]
                        val_dataset_fold = MyDataset(cv_data[val_idx], cv_lstm_data[val_idx])
                    else:
                        train_dataset_fold = MyDataset(cv_data[train_idx])
                        val_dataset_fold = MyDataset(cv_data[val_idx])

                    num_repeats = 0 # Count how many times the while loop repeated
                    while not num_repeats or CM[1,1]+CM[0,1] == 0:# or CM[0,0]+CM[1,0] == 0: # 1st run or precision was = 0 (all sites negative) or all sites positive --> fold had a bad initialization; rerun
                        if 'delta_t' in locals(): # To ensure some print messages actually print
                            del delta_t
                        # Creating the DataLoaders
                        train_loader_fold = DataLoader(train_dataset_fold, batch_size, shuffle = True)
                        val_loader_fold = DataLoader(val_dataset_fold, batch_size, shuffle = True)

                        # Declaring the model and optimizer
                        model = SequenceMLP(cur_hp[0], activ_fun, lstm_size).cuda()
                        optimizer = torch.optim.Adam(model.parameters(), lr = cur_hp[1])
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 7, verbose = True, min_lr = 1e-5)
                        # Convenience variables
                        increase_set_interval = 7
                        n_divisions_set = 6
                        n_epochs = increase_set_interval * (n_divisions_set+3)
                        n_epochs = 65
                        # Train and validate
                        for epoch in range(n_epochs):
                            # Begin with a less-imbalanced dataset, and gradually add more negative entries
                            """if data_version == 'v5' and epoch in set(increase_set_interval*np.arange(0, n_divisions_set+1)): # Make a loader change every increase_set_interval epochs
                                negative_cutoff = (len(train_idx) // n_divisions_set) * (epoch//increase_set_interval + 1) # Discards 0 to n_divisions_set samples on the last dataset expansion, but this is acceptable
                                train_idx_reduced = torch.cat((train_pos, train_neg[:negative_cutoff]))
                                #print(f'Epoch {epoch:3}, using {len(train_idx_reduced):7,} / {len(train_idx):,}' + ' '*5)
                                temp = train_dataset_fold[train_idx_reduced] # This is a tuple of len = 3, each entry containing "many" entries
                                temp = tuple(zip(temp[0], temp[1], temp[2])) # This is a tuple of len = "many", each entry containing 3 entries
                                train_loader_fold = DataLoader(temp, batch_size, shuffle = True)"""
                            t1 = time()
                            if 'delta_t' in locals():
                                print(f'Current fold: {fold_idx+1}/{my_kfold.n_splits}; epoch: {epoch+1:2}/{n_epochs}; number of repeats: {num_repeats:2}; Epoch time = {delta_t:.2f}  ', end = '\r')
                            else:
                                print(f'Current fold: {fold_idx+1}/{my_kfold.n_splits}; epoch: {epoch+1:2}/{n_epochs}; number of repeats: {num_repeats:2}')
                            train_loss = loop_model(model, optimizer, train_loader_fold, my_loss, epoch, lstm_size)
                            val_loss = loop_model(model, optimizer, val_loader_fold, my_loss, epoch, lstm_size, evaluation = True)
                            ###if data_version == 'v5' and epoch >= increase_set_interval*n_divisions_set: # Scheduler activates only on higher epochs, when the training dataset is complete
                            scheduler.step(val_loss)
                            t2 = time()
                            delta_t = t2 - t1
                            if cur_hp[1]/optimizer.param_groups[0]['lr'] >= 15.9: # 4 reductions in LR. Should be 16, but doing 15.9 due to floats
                                print(f'Early stopping at epoch {epoch+1:2}' + ' '*5)
                                break
                        # Calculating and recording the validation F1 score for this fold
                        val_pred = torch.empty((len(val_loader_fold.dataset), 2))
                        val_y = torch.empty((len(val_loader_fold.dataset)), dtype = torch.long)
                        for idx, data in enumerate(val_loader_fold):
                            if lstm_size:
                                X, y, lstm = data
                                lstm = lstm.cuda()
                            else:
                                X, y = data
                                lstm = None
                            X = X.cuda()
                            pred = model(X, lstm).cpu().detach()
                            val_pred[idx*batch_size:(idx*batch_size)+len(pred), :] = pred
                            val_y[idx*batch_size:(idx*batch_size)+len(y)] = y
                        val_pred_CM = val_pred.argmax(axis=1)
                        CM = confusion_matrix(val_y, val_pred_CM) # Confusion matrix to make F1 calcs easier
                        num_repeats += 1
                    rec = CM[1,1]/(CM[1,1]+CM[1,0])
                    pre = CM[1,1]/(CM[1,1]+CM[0,1])
                    if rec and pre: # Avoids dividing by 0 when calculating F1
                        F1 = 2/(1/rec + 1/pre)
                    else:
                        F1 = 0
                    print(CM)
                    print(F1)
                    temp_val_F1 += F1 / my_kfold.n_splits
                    temp_val_loss += my_loss(val_pred.cuda(), val_y.cuda()) / my_kfold.n_splits

                # Saving the average validation F1 after CV
                temp_val_loss = temp_val_loss.cpu().detach().item()
                final_val_F1.at[cur_hp[1], str(cur_hp[0])] = temp_val_F1
                final_val_loss.at[cur_hp[1], str(cur_hp[0])] = temp_val_loss
                final_val_F1.to_csv(f'{working_dir}/{F1_score_file}')
                final_val_loss.to_csv(f'{working_dir}/{val_loss_file}')
        return final_val_F1

    final_val_F1_list = np.empty_like(activ_fun_list, dtype = object) # This will hold multiple DataFrames, one for each activation fun type
    for idx, activ_fun in enumerate(activ_fun_list):
        # Setup for the results files
        F1_score_file = f'ANN_F1_{activ_fun}_{weight_hyperparam[1]}weight.csv'
        val_loss_file = f'ANN_val-loss_{activ_fun}_{weight_hyperparam[1]}weight.csv'
        # Running the CV
        final_val_F1_list[idx] = CV_model(activ_fun, working_dir, F1_score_file, val_loss_file)

    ### FINAL EVALUATION - TESTING THE BEST MODEL ###
    def run_final_evaluation(model, activ_fun, threshold = 0.5):
        model.eval()

        # Train loss
        train_pred = torch.empty((len(train_loader.dataset), 2))
        train_y = torch.empty((len(train_loader.dataset)), dtype = torch.long)
        for idx, data in enumerate(train_loader):
            if lstm_size:
                X, y, lstm = data
                lstm = lstm.cuda()
            else:
                X, y = data
                lstm = None
            X = X.cuda()
            pred = model(X, lstm).cpu().detach()
            train_pred[idx*batch_size:(idx*batch_size)+len(pred), :] = pred
            train_y[idx*batch_size:(idx*batch_size)+len(y)] = y
        #train_loss = my_loss(train_pred, train_y)
        #print(f'The train loss was {train_loss.item():.3f}')
        # Renormalizing the train_pred
        train_pred = (train_pred.T / train_pred.sum(axis=1)).T
        # Train confusion matrix
        train_pred_CM = train_pred[:, 1] >= threshold
        CM = confusion_matrix(train_y, train_pred_CM)
        if CM[1,1]+CM[0,1]:
            rec = CM[1,1]/(CM[1,1]+CM[1,0])
            pre = CM[1,1]/(CM[1,1]+CM[0,1])
            f1 = 2/(1/rec + 1/pre)
        else:
            rec, pre, f1 = 0, 0, 0
        print(f'The train recall was {rec*100:.2f}%')
        print(f'The train precision was {pre*100:.2f}%')
        print(f'The train F1 score was {f1*100:.2f}%')
        #fig, ax = plt.subplots(figsize = (8,8))
        #ax.set_title(f'{activ_fun} - Train Confusion Matrix')
        #_ = ConfusionMatrixDisplay(CM).plot(ax = ax)

        # Test loss
        test_pred = torch.empty((len(test_loader.dataset), 2))
        test_y = torch.empty((len(test_loader.dataset)), dtype = torch.long)
        for idx, data in enumerate(test_loader):
            if lstm_size:
                X, y, lstm = data
                lstm = lstm.cuda()
            else:
                X, y = data
                lstm = None
            X = X.cuda()
            pred = model(X, lstm).cpu().detach()
            test_pred[idx*batch_size:(idx*batch_size)+len(pred), :] = pred
            test_y[idx*batch_size:(idx*batch_size)+len(y)] = y
        test_loss = my_loss(test_pred.cuda(), test_y.cuda())
        print(f'The test loss was {test_loss:.3f}')
        # Renormalizing the test_pred
        test_pred = (test_pred.T / test_pred.sum(axis=1)).T
        # Test confusion matrix
        test_pred_CM = test_pred[:, 1] >= threshold
        CM = confusion_matrix(test_y, test_pred_CM)
        if CM[1,1]+CM[0,1]:
            rec = CM[1,1]/(CM[1,1]+CM[1,0])
            pre = CM[1,1]/(CM[1,1]+CM[0,1])
            f1 = 2/(1/rec + 1/pre)
        else:
            rec, pre, f1 = 0, 0, 0
        print(f'The test recall was {rec*100:.2f}%')
        print(f'The test precision was {pre*100:.2f}%')
        print(f'The test F1 score was {f1*100:.2f}%')
        print(CM)
        #fig, ax = plt.subplots(figsize = (8,8))
        #ax.set_title(f'{activ_fun} - Test Confusion Matrix')
        #_ = ConfusionMatrixDisplay(CM).plot(ax = ax)

    # Creating the full training Dataset / DataLoader
    if lstm_size:
        train_dataset = MyDataset(cv_data, cv_lstm_data)
        test_dataset = MyDataset(test_data, test_lstm_data)
    else:
        train_dataset = MyDataset(cv_data)
        test_dataset = MyDataset(test_data)
    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

    for final_val_F1, activ_fun in zip(final_val_F1_list, activ_fun_list):
        best_model_file = f'ANN_{activ_fun}_{weight_hyperparam[1]}weight_dict.pt'
        # Finding the best hyperparameters
        best_idx = np.unravel_index(np.nanargmax(final_val_F1.values), final_val_F1.shape)
        best_LR = final_val_F1.index[best_idx[0]]
        best_neurons_str = final_val_F1.columns[best_idx[1]]
        # Converting the best number of neurons from str to list
        best_neurons = []
        temp_number = []
        temp_tuple = []
        for elem in best_neurons_str:
            if elem in '0123456789':
                temp_number.append(elem)
            elif elem in {',', ')'} and temp_number: # Finished a number. 2nd check because there is a comma right after )
                converted_number = ''.join(temp_number)
                temp_tuple.append( int(converted_number) )
                temp_number = []
            if elem in {')'}: # Also finished a tuple
                best_neurons.append(tuple(temp_tuple))
                temp_tuple = []
        # Re-declaring the model
        model = SequenceMLP(best_neurons, activ_fun, lstm_size).cuda()

        # Checking if we already retrained this model
        try:
            mydict = torch.load(f'{working_dir}/{best_model_file}')
            model.load_state_dict(mydict)
        except FileNotFoundError: # Retraining the model with the full training set
            optimizer = torch.optim.Adam(model.parameters(), lr = best_LR)
            # Retrain
            for epoch in range(n_epochs):
                if lstm_size:
                    print(f'For {activ_fun}: epoch {epoch+1:3}/{n_epochs}' + ' '*20, end = '\r')
                train_loss = loop_model(model, optimizer, train_loader, my_loss, epoch, lstm_size)
            # Save the retrained model
            torch.save(model.state_dict(), f'{working_dir}/{best_model_file}')

        # CV Data
        print(f'Final results for {activ_fun} & weight {weight_hyperparam[1]}')
        print(f'Best hyperparameters: {best_neurons}, {best_LR}')
        print(f'CV F1 score: {final_val_F1.iat[best_idx]:.4f}')
        run_final_evaluation(model, activ_fun, 0.5)
        print()

### Other functions and classes
class MyDataset(Dataset):
    def __init__(self, data, lstm_data = None):
        self.Xdata = data[:, :-1]
        self.ydata = data[:, -1].type(torch.LongTensor)
        self.lstm_data = lstm_data

    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, idx):
        if isinstance(self.lstm_data, torch.Tensor):
            return self.Xdata[idx], self.ydata[idx], self.lstm_data[idx]
        else:
            return self.Xdata[idx], self.ydata[idx]

# MLP or LSTM+MLP model
class SequenceMLP(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu', lstm_size = 0):
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
        if lstm_size:
            self.lstm = torch.nn.LSTM(20, lstm_size, num_layers=1, batch_first=True, bidirectional=True)
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lstm_data = None):
        if 'lstm' in dir(self):
            _, (ht, _) = self.lstm(lstm_data) # Passing only the seq data through the LSTM
            to_MLP = ht[-1]
            out = self.model(to_MLP)
        else:
            out = self.model(x)
        probs = self.sigmoid(out)
        return probs

# A helper function that is called every epoch of training or validation
def loop_model(model, optimizer, loader, loss_function, epoch, lstm_size = None, evaluation = False):
    if evaluation:
        model.eval()
    else:
        model.train()
    batch_losses = []
    for data in loader:
        if lstm_size:
            X, y, lstm = data
            lstm = lstm.cuda()
        else:
            X, y = data
            lstm = None
        X = X.cuda()
        y = y.cuda()
        pred = model(X, lstm)
        loss = loss_function(pred, y)
        # Backpropagation
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_losses.append(loss.item()) # Saving losses
    return np.array(batch_losses).mean()

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Trains and cross-validates an MLP or RNN on O-glycosylation data')
    parser.add_argument('weight', type = int, nargs = '+', help = 'The weight(s) used in the loss function for positive-class predictions')
    parser.add_argument('-act', '--activ_fun_list', type = str, nargs = '+', metavar = 'relu', default = ['relu'],
        help = 'The activation functions tested. Must be in {"relu", "tanh", "sigmoid", "tanhshrink", "selu"}. Separate the names with a space')
    parser.add_argument('-ls', '--lstm_size', type = int, nargs = 1, metavar = 75, default = [75], help = 'Size of the LSTM unit. Set to 0 to use only an MLP')
    parser.add_argument('-dv', '--data_version', type = str, nargs = 1, metavar = 'v5', default = ['v5'],
        help = 'The version of the data to be used. Should be of the form "v#". Should be left as "v5"')
    parser.add_argument('-ws', '--window_size', type = int, nargs = 1, metavar = 10, default = [10],
        help='The number of AAs before and after the central S/T. Used only when data_version == "v5"')
    parser.add_argument('-bs', '--batch_size', type = int, nargs = 1, metavar = 32, default = [32], help='The batch size used in each epoch')
    myargs = parser.parse_args()
    for this_weight in myargs.weight:
        train_CV_complete([1, this_weight], myargs.activ_fun_list, myargs.lstm_size[0], myargs.data_version[0], myargs.window_size[0])
