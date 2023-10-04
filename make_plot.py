import numpy as np
import matplotlib.pyplot as plt
# Plotting settings
plt.rcParams.update({'font.size': 24, 'lines.markersize': 10})
annotation_fontsize = 6
bbox_dict = dict(facecolor='white', alpha=1, edgecolor='white', linewidth = 0, pad = 0.15)

def main(data_version):
    """
    Generates Precision-Recall curves for the models trained on different datasets

    Parameters
    ----------
    data_version : string
        The version of the dataset you are using. Should be "v#", where # is an integer. The meaning of each version is:
        v1 = Mauri et al.'s original data [issues: duplicated entries, mismatch between raw and processed data, non-mammalian entries]
        v2 = Seokyoung's modified data based on Mauri et al.'s original data [issues: duplicated data, non-mammalian entries. Tested but used in the paper]
        v3 = Mauri et al.'s original data fixed by me; sequence only [issues: non-mammalian entries]
        v4 = Uniprot-only data [Never used]
        v5 = Wulff-Fuentes et al.'s data filtered to contain only mammalian sequences with site information
    """
    fig, ax = plt.subplots(figsize = (16, 9), dpi = 500)
    if data_version == 'v1':
        # Plotting - Prepublished models
        plt.plot([48.55, 31.27, 14.36, 3.82], [3.17, 5.05, 6.89, 8.68], '-+', label = 'YinOYang') # Labeled for use in the plot legend
        plt.plot(65.09, 3.90, 'o', label = 'O-GlcNAc-PRED-II')
        plt.plot(49.09, 6.20, '1', label = 'OGTSite')

        # Plotting - Mauri models
        plt.plot(98.58, 1.37, '^', label = 'Mauri')
        plt.plot(39.82, 3.10, '^')
        plt.plot(38.94, 0.90, '^')
        plt.plot(10.93, 0.92, '^')
        plt.plot(39.38, 0.78, '^')
        plt.plot(38.94, 0.85, '^')

        # Plotting - Our MLP v1 data
        x_values = [79.25, 78.30,   75.47,   68.87,   58.49, 54.72,   52.83,  43.40, 42.45,  36.79,  33.96,  31.13,  27.36,  24.53,  23.58,  22.64, 22.64, 22.64, 22.64, 22.64, 22.64, 22.64, 22.64, 22.64, 21.70,  19.81,  18.87,   18.87, 16.04,     14.15] # Recall
        y_values = [01.88, 02.05,   02.24,   02.35,   02.36, 02.62,   03.03,  03.08, 03.91,  04.38,  05.57,  06.93,  09.27,  12.81,  18.25,  22.86, 24.64, 28.57, 30.00, 31.58, 32.88, 33.80, 34.29, 35.29, 35.94,  36.21,  41.67,   44.44, 44.74,     45.45] # Precision
        labels = ['1e-16',    '', '1e-14', '1e-13', '1e-12',    '', '1e-10', '1e-9',    '', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5',    '',    '', '0.8', '0.9', '0.95', '0.99', '0.999',    '', '0.99995']
        #labels = ['1e-16', '1e-15', '1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '0.99995']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our MLP')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 1.33, 's', label = 'No model')

    elif data_version == 'v3':
        # Plotting - YoY
        x_values = [ 100, 97.91, 81.21, 57.69, 37.76, 21.82, 06.45, 00.57]
        y_values = [1.35, 01.39, 01.75, 02.29, 03.18, 04.93, 05.74, 10.00]
        labels = ['0.15', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
        yoy_line = plt.plot(x_values, y_values, '-', label = 'YinOYang')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = yoy_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-76
        x_values = [34.69, 34.69,   31.63, 30.61,   30.61, 28.57,   28.57, 27.55,  27.55,  23.47,  22.45,  21.43,  21.43,  20.41,  20.41, 19.39, 18.37, 18.37, 18.37, 18.37, 17.35, 17.35, 17.35, 16.33, 16.33,  15.31,  15.31, 13.27,    12.24,     11.22]
        y_values = [03.39, 03.67,   03.80, 04.16,   04.68, 04.99,   05.65, 06.21,  07.38,  07.54,  08.66,  09.86,  11.29,  12.74,  15.27, 15.70, 16.22, 17.14, 17.35, 18.75, 18.48, 18.89, 19.10, 19.05, 21.05,  20.27,  24.19, 26.53,    26.67,     30.56]
        labels = ['1e-16',    '', '1e-14',    '', '1e-12',    '', '1e-10',    '', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01',    '', '0.1',    '',    '', '0.4',    '',    '', '0.7', '0.8', '0.9', '0.95', '0.99',    '', '0.9999', '0.99999']
        #labels = ['1e-16', '1e-15', '1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '0.99999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-76', color = '#FF0000')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-152
        x_values = [45.92,   43.88, 39.90, 35.71,   34.69, 30.61, 27.55,  23.47,  23.47,  19.39, 18.37,  17.35, 17.35,  17.35, 16.33,  15.31, 15.31, 15.31, 15.31, 15.31,     15.31, 14.29, 14.29, 14.29,  13.27, 13.27,   13.27,    12.24, 10.20,       08.16]
        y_values = [02.65,   02.85, 02.96, 03.14,   03.63, 03.91, 04.18,  04.39,  05.42,  05.86, 06.87,  08.02, 09.66,  12.23, 14.04,  14.42, 14.85, 15.79, 16.13, 16.48,     17.05, 16.07, 17.50, 19.44,  18.57, 21.31,   24.53,    26.09, 27.03,       28.57]
        labels = ['1e-16', '1e-15',    '',    '', '1e-12',    '',    '', '1e-9', '1e-8', '1e-7',    '', '1e-5',    '', '1e-3',    '', '0.05',    '',    '',    '',    '', '0.5-0.6', '0.7',    '', '0.9', '0.95',    '', '0.999', '0.9999',    '', '0.9999999']
        #labels = ['1e-16', '1e-15', '1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '0.99999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-152', color = '#56B029')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 1.29, 's', label = 'No model')

    elif data_version == 'v5':
        # Plotting - YoY
        x_values = [100.0, 96.21, 72.74, 48.30, 28.19, 13.59,  4.22,  0.26]
        y_values = [ 2.43,  2.50,  3.12,  4.18,  5.69,  8.08, 11.98, 18.56]
        labels = [ '0.12', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
        yoy_line = plt.plot(x_values, y_values, '-', label = 'YinOYang')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = yoy_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-75; window size = 5
        x_values = [85.08, 75.7 , 66.93, 59.09, 51.72, 45.31, 38.69, 34.43, 32.46, 30.24, 29.13, 28.02, 26.91, 26.02, 24.55, 23.3 , 21.44, 19.43, 15.46,  9.16,  4.15]
        y_values = [ 3.21,  3.73,  4.45,  5.35,  6.46,  7.8 ,  9.35, 10.79, 11.54, 12.18, 12.86, 13.34, 13.71, 14.25, 14.52, 15.14, 16.2 , 17.11, 19.62, 22.7 , 26.79]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-75;   5 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-75; window size = 10
        x_values = [70.79, 64.24, 57.84, 52.97, 48.64, 44.88, 41.16, 38.8 , 37.8 , 36.65, 35.65, 34.79, 34.25, 33.5 , 32.75, 31.68, 30.42, 29.35, 26.84, 22.83, 18.54,  8.77]
        y_values = [ 4.8 ,  5.84,  7.06,  8.74, 10.74, 13.07, 15.65, 17.48, 18.39, 19.44, 20.01, 20.55, 21.13, 21.54, 22.06, 22.46, 23.33, 24.22, 26.09, 27.91, 29.07, 34.36]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-75; 10 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-75; window size = 15
        x_values = [68.4 , 61.49, 55.91, 51.22, 47.6 , 44.6 , 41.41, 39.62, 38.73, 37.69, 37.33, 36.97, 36.51, 35.9 , 35.4 , 34.61, 33.61, 33.  , 30.6 , 27.59, 23.73, 12.13]
        y_values = [ 5.12,  6.51,  8.32, 10.45, 13.23, 16.23, 19.2 , 21.65, 22.73, 23.74, 24.54, 25.25, 25.84, 26.13, 26.69, 27.31, 28.42, 29.38, 30.91, 33.58, 35.01, 39.15]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-75; 15 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-75; window size = 20
        x_values = [69.69, 63.39, 57.91, 53.58, 49.53, 46.46, 43.27, 41.23, 40.37, 39.12, 38.01, 37.47, 36.9 , 36.15, 35.72, 34.82, 33.57, 31.82, 29.49, 24.48, 20.08,  5.48]
        y_values = [ 5.21,  6.48,  8.15, 10.49, 13.28, 16.77, 20.58, 23.38, 24.68, 25.83, 26.42, 27.22, 27.97, 28.42, 29.16, 29.87, 30.79, 31.38, 33.73, 35.08, 36.84, 40.16]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-75; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-150; window size = 20
        x_values = [88.83, 76.77, 64.57, 54.08, 47.21, 43.41, 41.02, 39.51, 38.87, 38.01, 37.58, 36.9 , 36.36, 35.86, 35.36, 34.54, 33.75, 32.86, 30.67, 27.49, 23.59,  7.8 ]
        y_values = [ 3.08,  4.07,  6.21, 10.38, 17.42, 23.52, 27.58, 30.28, 31.58, 32.71, 33.5 , 34.04, 34.58, 34.96, 35.51, 35.87, 36.86, 37.7 , 39.04, 41.18, 42.79, 52.66]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-150; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-225; window size = 20
        x_values = [62.92, 56.41, 51.04, 47.14, 44.42, 42.13, 40.3 , 38.55, 37.83, 37.08, 36.58, 36.22, 35.83, 35.47, 34.9 , 34.29, 33.39, 32.82, 31.1 , 28.31, 25.41, 14.03]
        y_values = [ 5.87,  8.57, 11.96, 15.89, 20.43, 24.81, 29.4 , 32.08, 33.16, 34.3 , 35.11, 35.86, 36.37, 36.9 , 37.43, 37.84, 38.81, 39.75, 41.56, 43.39, 45.81, 53.04]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-225; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Wulff-Fuentes et al. motif (page 3 of their paper)
        x_values = [0.08, 1.01, 6.84, 23.94, 58.50]
        y_values = [39.29, 20.33, 11.85, 5.89, 3.38]
        labels = [5, 4, 3, 2, 1]
        WF_dots = plt.plot(x_values, y_values, 'o', label = 'W-F motif')
        bbox_dict_WF = dict(facecolor='white', alpha=1, edgecolor='white', linewidth = 0, pad = 0.25) # Additional padding
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = WF_dots[0].get_color())
            my_label.set_bbox(bbox_dict_WF)
        # No model / testing everything
        plt.plot(100, 2.44, 's', label = 'No model')

    # F curves
    rec = np.linspace(0.030, 1, 195)
    beta = 1
    x_pos = 100.3
    # These y_pos and extra rec points are manually set up
    if beta == 1:
        y_pos = [1.8, 4.4, 7.2, 10.2, 13.4, 16.7, 20.3, 24.1, 28.0, 32.2, 36.9]
        x_pos_top = [1.9, 4.3, 7.3, 10.3, 13.5, 16.9, 20.4, 24.0, 27.9]
        rec = np.sort( np.concatenate(([0.0255, 0.026, 0.027, 0.028, 0.029, 0.052, 0.053, 0.054], rec)) ) # The F = 0.05 and F = 0.10 curves need a few additional points
    elif beta == 2:
        y_pos = [0, 1.3, 0, 3.8, 0, 6.9, 0, 11.0]
        x_pos_top = [3.3, 7.3, 11.2, 15.5, 20.1, 24.7, 29, 33.9]
        rec = np.sort( np.concatenate((np.linspace(0.0402, 0.0408, 4), np.linspace(0.041, 0.044, 4), [0.046, 0.047], np.linspace(0.081, 0.084, 4), [0.123, 0.124], rec)) ) # The F = 0.05, 0.10, and 0.15 curves need a few additional points
    # Calculating the precision to get a given F score at each recall point
    for idx, F_score in enumerate([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]):
        pre = F_score*rec / (rec + (beta**2*rec) - F_score*beta**2)
        plt.plot(rec[pre>0]*100, pre[pre>0]*100, ':k') # Plotting >0 to avoid impossible numbers (because recall is too low)
        if 'y_pos' in locals() and idx < len(y_pos) and y_pos[idx] > 0: # Right-side labels for F1 isolines
            plt.text(x_pos, y_pos[idx], f'{round(F_score*100)}%', fontsize = 16)
        if 'x_pos_top' in locals() and idx < len(x_pos_top): # Upper-side labels for F1 isolines
            plt.text(x_pos_top[idx], 100.3, f'{round(F_score*100)}%', fontsize = 13)

    # Plot housekeeping
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 110, 10))
    ax.set_ylabel('Precision %')
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 110, 10))
    ax.set_xlabel('Recall %')
    if data_version not in {'BLANK', 'v0'}:
        ax.legend(fontsize = 22)
    plt.tight_layout()
    plt.savefig(f'O-Gly_model_eval_F{beta}_{data_version}-data.png')

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Generates Precision-Recall curves for the models trained on different datasets')
    parser.add_argument('data_version', type = str, nargs = 1, help = 'The version of the dataset you are using. Should be "v#", where # is an integer')
    my_args = parser.parse_args()
    main(my_args.data_version[0])
