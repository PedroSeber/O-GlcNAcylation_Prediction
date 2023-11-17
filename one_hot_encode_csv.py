"""
One-hot encodes the original data from Mauri et al. ('file_for_ML.csv'). The output file is 'OH_data.csv'.
Also prepares sequence data for LSTM as needed
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def preprocess_data(filename, window_size = 10):
    """
    Transforms the data for model training with ANN_train.py

    Parameters
    ----------
    filename : string
        The name of the raw data file.
        Should be one of "file_for_ML.csv", "file_for_ML_v2.csv", "all_sites_fixed.csv", "all_sites_PS.csv", "all_sites_PS_Uniprot.csv", or "OVSlab_allSpecies_O-GlcNAcome_PS.csv".
    window_size : int, optional, default = 10
        If filename == "OVSlab_allSpecies_O-GlcNAcome_PS.csv" (v5 data), the size of each side of the window taken for each protein sequence.
            Each final sequence has 2*window_size + 1 amino acids.
        Else, the window_size is fixed at 10 and each final sequence has 21 amino acids (for comparison with YinOYang).
    """
    if filename == 'file_for_ML.csv':
        X_data_range = list(range(1, 15))
        already_OHE = {4, 11, 12} # "pro" and "is_threonine" are already OHEed; "rigidness" is a float
        to_skip = {} # Do not skip any of the intermediate columns
    elif filename == 'file_for_ML_v2.csv':
        X_data_range = list(range(3, 17)) + [18, 19] # nSer to pro, then SS_angle to Phi (Psi gets processed when Phi is processed)
        already_OHE = {7, 14, 15} # "pro" and "is_threonine" are already OHEed; "rigidness" is a float. Note that "ASA" is handled separately
    elif filename not in {'all_sites_fixed.csv', 'all_sites_PS.csv', 'all_sites_PS_Uniprot.csv', 'OVSlab_allSpecies_O-GlcNAcome_PS.csv'}: # These data files do not require any one-hot enconding, just processing
        raise ValueError('filename should be "file_for_ML.csv", "file_for_ML_v2.csv", "all_sites_fixed.csv", "all_sites_PS.csv", "all_sites_PS_Uniprot.csv", or "OVSlab_allSpecies_O-GlcNAcome_PS.csv"')
    data = pd.read_csv(filename)

    # One-hot encoding or normalizing the data as needed
    if filename in {'file_for_ML.csv', 'file_for_ML_v2.csv'}:
        for idx in X_data_range:
            # Checking what to do for each column
            if idx in already_OHE:
                one_hot_data = pd.concat((one_hot_data, data.iloc[:, idx]), axis = 1)
            elif idx == 3 and filename == 'file_for_ML_v2.csv': # Sum of S and T for the v2 dataset
                summed_ser_thr = data.iloc[:, 3] + data.iloc[:, 4] + 1 # The original dataset summed Ser and Thr. +1 because this file ignores the central S/T
                enc = OneHotEncoder()
                temp_data = np.atleast_2d(summed_ser_thr.values).T # The OneHotEncoder demands 2D data
                enc.fit(temp_data)
                # Creating a new DataFrame
                one_hot_data = pd.DataFrame(enc.transform(temp_data).toarray()[:, 1:].astype(np.uint8), columns = enc.get_feature_names_out(['n_ser_thr'])[1:]) # Using [1:] to avoid multicollinearity, since we have a bias term
            elif idx == 4 and filename == 'file_for_ML_v2.csv': # We already dealt with nThr in the elif-statement above
                continue
            elif idx == 16 and filename == 'file_for_ML_v2.csv': # ASA is a float, but we want to normalize it to 0-1
                normalized = data.iloc[:, idx] / data.iloc[:, idx].max().max()
                one_hot_data = pd.concat((one_hot_data, normalized), axis = 1)
            elif idx == 19 and filename == 'file_for_ML_v2.csv': # Secondary structury based on Phi-Psi angles
                phi_psi_struct = np.zeros_like(data.iloc[:, idx])
                alpha_mask = (data.iloc[:, idx] > -160) & (data.iloc[:, idx] < -50) & (data.iloc[:, idx+1] > -60) & (data.iloc[:, idx+1] < 20)
                beta_mask = (data.iloc[:, idx] > -160) & (data.iloc[:, idx] < -50) & (data.iloc[:, idx+1] > 100) & (data.iloc[:, idx+1] < 180)
                phi_psi_struct[alpha_mask] = 1
                phi_psi_struct[beta_mask] = 2
                enc = OneHotEncoder()
                temp_data = np.atleast_2d(phi_psi_struct).T # The OneHotEncoder demands 2D data
                enc.fit(temp_data)
                temp_DF = pd.DataFrame(enc.transform(temp_data).toarray()[:, 1:].astype(np.uint8), columns = enc.get_feature_names_out(['phi_psi_struct'])[1:]) # Using [1:] to avoid multicollinearity, since we have a bias term
                one_hot_data = pd.concat((one_hot_data, temp_DF), axis = 1)
            else:
                enc = OneHotEncoder()
                temp_data = np.atleast_2d(data.iloc[:, idx].values).T # The OneHotEncoder demands 2D data
                enc.fit(temp_data)
                # Creating a new DataFrame or appending to it
                if idx == 1:
                    one_hot_data = pd.DataFrame(enc.transform(temp_data).toarray()[:, 1:].astype(np.uint8), columns = enc.get_feature_names_out([data.columns[idx]])[1:]) # Using [1:] to avoid multicollinearity, since we have a bias term
                else:
                    temp_DF = pd.DataFrame(enc.transform(temp_data).toarray()[:, 1:].astype(np.uint8), columns = enc.get_feature_names_out([data.columns[idx]])[1:]) # Using [1:] to avoid multicollinearity, since we have a bias term
                    one_hot_data = pd.concat((one_hot_data, temp_DF), axis = 1)

    # Wrapping up and generating sequence files for the LSTM models
    if filename == 'file_for_ML.csv':
        # Putting the y values in the end
        y_vals = data.iloc[:, 0].copy()
        y_vals[y_vals == 2] = 0 # Mauri et al. label non-acylated proteins as 2 instead of 0
        one_hot_data = pd.concat((one_hot_data, y_vals), axis = 1)
        # Saving the output
        one_hot_data.to_csv('OH_data_v1.csv', index = False)
    elif filename == 'file_for_ML.csv':
        # Putting the y values in the end
        y_vals = data.iloc[:, -1].copy()
        one_hot_data = pd.concat((one_hot_data, y_vals), axis = 1)
        # Saving the output
        one_hot_data.to_csv('OH_data_v2.csv', index = False)
        # Generating the LSTM data
        seq_data = data.iloc[:, 2].copy()
        for idx in range(seq_data.shape[0]):
            seq_data[idx] = seq_data[idx].translate({ord(i): None for i in '"'}) # str.translate replaces the characters on the right of the for (in this case, a quote character) with None
        OHE_seq_data = OHE_for_LSTM(seq_data.values.squeeze())
        np.save('OH_LSTM_data_v2.npy', OHE_seq_data) # np.save instead of csv because this is a 3D array
    elif filename in {'all_sites_fixed.csv', 'all_sites_PS.csv', 'all_sites_PS_Uniprot.csv'}:
        positive_sites = set()
        for idx in range(data.shape[0]):
            positive_sites.add( (data.iat[idx, 0], int(data.iat[idx, 1])-1) )
        seq_data = []
        y_boolean = []
        has_been_sweeped = set()
        for idx in range(data.shape[0]):
            if data.iat[idx, 0] not in has_been_sweeped: # Run through the entire sequence only once
                for letter_idx, letter in enumerate(data.iat[idx, 2]):
                    if letter in {'T', 'S'} or letter_idx == int(data.iat[idx, 1])-1: # Take all T/S or the first positive site
                        seq_data.append( get_nearby_AA(data.iat[idx, 2], letter_idx, 10) ) # Technically could use any window_size, but other values would prevent comparison with YinOYang
                        y_boolean.append( int((data.iat[idx, 0], letter_idx) in positive_sites) )
                has_been_sweeped.add(data.iat[idx, 0]) # Add the protein ID to a set, to avoid running through this entire protein again
            elif data.iat[idx, 2][data.iat[idx, 1]-1] not in {'T', 'S'}: # This protein has more than one positive site, but we have taken it already if that additional site is an S or T
                # NOTE: this section never gets called because all sites that are not the first coincidentally are all T or S.
                # For all_sites_fixed.csv, there are 4 first sites that are non-S and non-T, one in each of F6T0L5, P52285, Q8WWM7, and Q557E4
                seq_data.append( get_nearby_AA(data.iat[idx, 2], data.iat[idx, 1]-1) )
                y_boolean.append(1) # We know this is a positive site
        OHE_seq_data = OHE_for_LSTM(np.array(seq_data), 10) # Technically could use any window_size, but other values would prevent comparison with YinOYang
        # Modifying the name of the saved file
        if filename == 'all_sites_fixed.csv':
            data_version = 'v3'
        else:
            data_version = f'v4{"-Uniprot"*("Uniprot" in filename)}' # v4 or v4-Uniprot
        np.save(f'OH_LSTM_data_{data_version}.npy', OHE_seq_data) # np.save instead of csv because this is a 3D array
        y_boolean = pd.Series(y_boolean, name = 'is_O-GlcNAcylated')
        y_boolean.to_csv(f'OH_data_{data_version}.csv', index = False)
    elif filename == 'OVSlab_allSpecies_O-GlcNAcome_PS.csv':
        #mammalian_species = {'Camelus dromedarius', 'Rattus norvegicus', 'Mus musculus', 'Chlorocebus aethiops', 'Oryctolagus cuniculus', 'Macaca fascicularis', 'Ovis aries', 'Rattus sp', 'Chlorocebus sabaeus',
        #                     'Canis lupus familiaris', 'Bos taurus', 'Homo sapiens', 'Capra hircus', 'Macaca mulatta', 'Sus scrofa'} # Manually generated from the original dataset
        positive_sites = set()
        for idx in range(data.shape[0]):
            temp_sites = data.iat[idx, 2].split(';') # A single protein may have multiple positive sites, which are separated with a ;
            for site in temp_sites:
                positive_sites.add( (data.iat[idx, 0], int(site[1:])) ) # site[1:] to not add the letter
        seq_data = []
        if window_size == 5:
            seq_data_10 = [] # Comment out this line to not generate data for larger window sizes
            seq_data_15 = []
            seq_data_20 = []
            seq_data_25 = []
        y_boolean = []
        already_included = set()
        conflict_seqs = set()
        for idx in range(data.shape[0]):
            if not(idx % 100):
                print(f'Current idx: {idx:4}/{data.shape[0]-1} (Updated every 100 idx)', end = '\r')
            for letter_idx, letter in enumerate(data.iat[idx, 3]):
                if letter in {'T', 'S'}: # All sites are S or T
                    sequence = get_nearby_AA(data.iat[idx, 3], letter_idx, window_size)
                    positive_bool = (data.iat[idx, 0], letter_idx+1) in positive_sites
                    if (sequence, positive_bool) not in already_included and (sequence, not(positive_bool)) not in already_included: # Equivalent to just checking whether the sequence has been included - proceed only if the sequence is not there as a positive entry and not there as a negative entry
                        seq_data.append(sequence)
                        y_boolean.append(int(positive_bool)) # Ints allow one to easily obtain the sum and mean of the data
                        if window_size == 5 and 'seq_data_10' in locals(): # Also preparing seq data for larger window sizes
                            seq_data_10.append(get_nearby_AA(data.iat[idx, 3], letter_idx, 10))
                            seq_data_15.append(get_nearby_AA(data.iat[idx, 3], letter_idx, 15))
                            seq_data_20.append(get_nearby_AA(data.iat[idx, 3], letter_idx, 20))
                            seq_data_25.append(get_nearby_AA(data.iat[idx, 3], letter_idx, 25))
                        already_included.add((sequence, positive_bool))
                    elif (sequence, not(positive_bool)) in already_included: # Conflict: the sequence has already been included, but with the opposite O-GlcNAcylation status
                        conflict_seqs.add( (sequence, int(not(positive_bool))) )
        print('One-hot encoding the sequence data' + ' '*15)
        OHE_seq_data = OHE_for_LSTM(np.array(seq_data), window_size)
        if window_size == 5 and 'OHE_seq_data' in locals(): # 'OHE_seq_data' won't be in locals if you comment the above line to make the motif testing go faster
            np.save('OH_LSTM_data_v5_5-window.npy', OHE_seq_data) # np.save instead of csv because this is a 3D array
        elif 'OHE_seq_data' in locals(): # Directly using a larger window_size adds entries to the dataset. These entries have identical amino acids between -5 and 5, but are unique (at least one different amino acid between -X and X)
            np.save(f'OH_LSTM_data_v5_{window_size}-window_expanded.npy', OHE_seq_data) # np.save instead of csv because this is a 3D array
        if window_size == 5 and 'seq_data_10' in locals():
            print('One-hot encoding sequence data for larger windows' + ' '*10)
            OHE_seq_data_10 = OHE_for_LSTM(np.array(seq_data_10), 10)
            np.save('OH_LSTM_data_v5_10-window.npy', OHE_seq_data_10) # np.save instead of csv because this is a 3D array
            OHE_seq_data_15 = OHE_for_LSTM(np.array(seq_data_15), 15)
            np.save('OH_LSTM_data_v5_15-window.npy', OHE_seq_data_15) # np.save instead of csv because this is a 3D array
            OHE_seq_data_20 = OHE_for_LSTM(np.array(seq_data_20), 20)
            np.save('OH_LSTM_data_v5_20-window.npy', OHE_seq_data_20) # np.save instead of csv because this is a 3D array
            OHE_seq_data_25 = OHE_for_LSTM(np.array(seq_data_20), 25)
            np.save('OH_LSTM_data_v5_25-window.npy', OHE_seq_data_25) # np.save instead of csv because this is a 3D array
        # Setting conflict sequences as positive, since they're O-GlcNAcylated in at least one site
        for elem in conflict_seqs:
            if elem[1] == 0:
                temp_idx = seq_data.index(elem[0])
                y_boolean[temp_idx] = 1
        y_boolean = pd.Series(y_boolean, name = 'is_O-GlcNAcylated')
        y_boolean.to_csv(f'OH_data_v5_{window_size}-window.csv', index = False)
        # W-F et al. motif analysis (page 3 of their paper) - should be done once by running this file with window_size == 3
        if window_size == 3:
            follows_motif = np.zeros_like(y_boolean, dtype = np.uint8) # A score that shows how many sites in a sequence follow the motif
            for idx, seq in enumerate(seq_data):
                if len(seq) < 6:
                    n_follow = 0
                elif seq[window_size] == 'S':
                    n_follow = (seq[window_size-3]=='P') + (seq[window_size-2]=='P') + ((seq[window_size-1]=='V')|(seq[window_size-1]=='T')) + ((seq[window_size+1]=='S')|(seq[window_size+1]=='T')) + (seq[window_size+2]=='A')
                elif seq[window_size] == 'T':
                    n_follow = ((seq[window_size-3]=='P')|(seq[window_size-3]=='T')) + (seq[window_size-2]=='P') + ((seq[window_size-1]=='V')|(seq[window_size-1]=='T')) + ((seq[window_size+1]=='S')|(seq[window_size+1]=='T')) + \
                               ((seq[window_size+2]=='A')|(seq[window_size+2]=='T'))
                follows_motif[idx] = n_follow
        if 'follows_motif' in locals():
            min_cutoff = 0 if follows_motif.min() < 0 else 1
            cutoffs = range(min_cutoff, follows_motif.max() + 1)
            y_boolean = pd.Series(y_boolean, dtype = bool) # Converting to bool to avoid weirdness with negation. Original is int for convenience
            motif_TP = np.zeros(cutoffs.stop - cutoffs.start, dtype = np.uint32)
            motif_TN = np.zeros_like(motif_TP)
            motif_FP = np.zeros_like(motif_TP)
            motif_FN = np.zeros_like(motif_TP)
            for idx, cutoff in enumerate(cutoffs):
                motif_TP[idx] = (y_boolean&(follows_motif >= cutoff)).sum()
                motif_TN[idx] = (~y_boolean&~(follows_motif >= cutoff)).sum()
                motif_FP[idx] = (~y_boolean&(follows_motif >= cutoff)).sum()
                motif_FN[idx] = (y_boolean&~(follows_motif >= cutoff)).sum()
            rec = motif_TP / (motif_TP + motif_FN)
            pre = motif_TP / (motif_TP + motif_FP)
            f1 = 2/(1/rec + 1/pre)
            TP, TN, FP, FN = motif_TP.astype(float), motif_TN.astype(float), motif_FP.astype(float), motif_FN.astype(float)
            MCC = (TP*TN - FP*FN) / np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
            print(f'Motif analysis: Cutoffs = {list(cutoffs)}, TP = {motif_TP}, TP + FP (what is marked as positive) = {motif_TP + motif_FP}' + ' '*10)
            print(f'Rec = {np.round(rec*100, 2)}')
            print(f'Pre = {np.round(pre*100, 2)}')
            print(f'F1 = {np.round(f1*100, 2)}')
            print(f'MCC = {np.round(MCC*100, 2)}')
        print(f'Total sites: {len(seq_data):,}' + ' '*30)
        print(f'Positive sites: {sum(y_boolean):,} ({np.mean(y_boolean)*100:.2f}%)')

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

def get_nearby_AA(sequence, letter_idx, window = 10):
    # Get all aminoacids in *sequence* within *window* distance of *sequence[letter_idx]*, leading to an output with len = window*2+1 in the general case
    if letter_idx <= window: # Relevant AA is near the beginning of the sequence
        nearby = sequence[:letter_idx+1+window]
    elif letter_idx + window > len(sequence): # Relevant AA is near the end of the sequence
        nearby = sequence[letter_idx-window:]
    else: # Relevant AA is at the middle of the sequence (general case)
        nearby = sequence[letter_idx-window:letter_idx+1+window]
    return nearby

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Preprocess the original or modified dataset for O-GlcNAcylation prediction')
    parser.add_argument('filename', type = str, nargs = 1,
        help = 'The name of the raw dataset. Must be "file_for_ML.csv", "file_for_ML_v2.csv", "all_sites_fixed.csv", "all_sites_PS.csv", "all_sites_PS_Uniprot.csv", or "OVSlab_allSpecies_O-GlcNAcome_PS.csv"')
    parser.add_argument('-ws', '--window_size', type = int, nargs = 1, metavar = 10, default = [10],
        help = 'The size of each side of the window. The total window size is 2*window_size + 1. Optional, default = 10.')
    args = parser.parse_args()
    preprocess_data(args.filename[0], args.window_size[0]) # [0] to convert from list to string

