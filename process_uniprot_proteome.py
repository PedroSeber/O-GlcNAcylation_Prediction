from one_hot_encode_csv import get_nearby_AA
import numpy as np
import pandas as pd

def preprocess_uniprot_proteome(filename):
    # Reading the file and extracting the whole-protein sequences from it
    if filename.endswith('.gz'):
        import gzip
        with gzip.open(filename, 'rb') as f:
            data = f.read().decode()
    else: # Assuming regular .fasta file
        with open(filename) as f:
            data = f.read()
    data = data.replace('\n', '').split('>sp|')[1:] # Removing linebreaks (because the same sequence is split every once in a while) and splitting by '>sp|', the beginning of a sequence header. [1:] because the first element is empty
    data = [elem.split('SV=') for elem in data]
    headers, seqs = zip(*data)
    headers = [elem.split(' OS=')[0].split('|') for elem in headers] # Keeping only the Uniprot ID and protein name in the headers
    uniprotID, protein_name = zip(*headers)
    seqs = [elem[1:] for elem in seqs] # Removing the number from SV = that was tacked into the sequences due to the split 2 lines above
    # Obtaining 41-AA-long sequences (window size = 20) from the proteins above
    uniprotID_processed = []
    protein_name_processed = []
    seq_data_20 = []
    position_list = []
    for protein_idx, this_sequence in enumerate(seqs):
        if not(protein_idx % 100):
            print(f'Current protein idx: {protein_idx:5}/{len(seqs)-1} (Updated every 100 idx)', end = '\r')
        for letter_idx, letter in enumerate(this_sequence):
            if letter in {'T', 'S'}: # All sites are S or T
                seq_data_20.append( get_nearby_AA(this_sequence, letter_idx, 20) )
                uniprotID_processed.append( uniprotID[protein_idx] )
                protein_name_processed.append( protein_name[protein_idx] )
                position_list.append(f'{letter}{letter_idx+1}')
    output = np.concatenate((uniprotID_processed, protein_name_processed, seq_data_20, position_list)).reshape(4, -1).T
    output = pd.DataFrame(output, columns = ['Uniprot ID', 'Protein name', 'Local sequence', 'Central amino acid position'])
    output.to_csv(f'{filename.split(".fasta")[0]}_dataset.csv', index = False)

if __name__ == '__main__':
    # Input setup
    import argparse
    from os.path import exists as osexists
    from Predict import predict_OGlcNAcylation
    parser = argparse.ArgumentParser(description = 'Reads a .fasta.gz or .fasta file from Uniprot containing the proteome of a species and builds a dataset formatted in the way expected by Predict.py.')
    parser.add_argument('filename', type = str, nargs = 1, help = 'The path to the .fasta.gz or .fasta file.')
    parser.add_argument('-t', '--threshold', metavar = '0.6', nargs = 1, type = float, default = [0.6], help = 'The minimum prediction threshold for a site to be considered O-GlcNAcylated. Optional, default = 0.5.')
    parser.add_argument('-bs', '--batch_size', metavar = '2048', nargs = 1, type = int, default = [2048], help = 'The number of predictions done at a time. Lower only if getting out of memory errors. Optional, default = 2048.')
    args = parser.parse_args()
    filename = args.filename[0] # [0] to convert from list to string
    threshold = args.threshold[0]
    batch_size = args.batch_size[0]
    # Building the .csv dataset if needed
    if not(osexists(f'{filename.split(".fasta")[0]}_dataset.csv')):
        preprocess_uniprot_proteome(filename)
        print(f'Dataset {filename.split(".fasta")[0]}_dataset.csv built.' + ' '*15)
    else:
        print(f'Dataset {filename.split(".fasta")[0]}_dataset.csv was already built; loading that file.')
    # Predicting O-GlcNAcylation sites
    predict_OGlcNAcylation(f'{filename.split(".fasta")[0]}_dataset.csv', threshold, batch_size, [2])
    print('RNN predictions done.' + ' '*10)
    # Merging the dataset and prediction files into one for more human-readable output
    dataset = pd.read_csv(f'{filename.split(".fasta")[0]}_dataset.csv')
    preds = pd.read_csv(f'{filename.split(".fasta")[0]}_dataset_predictions.csv')
    merge = pd.concat((dataset, preds.iloc[:, 1:]), axis = 1)
    merge.to_csv(f'{filename.split(".fasta")[0]}_dataset_predictions.csv', index = False)
    print(f'Run complete. Results are available in the {filename.split(".fasta")[0]}_dataset_predictions.csv file')
