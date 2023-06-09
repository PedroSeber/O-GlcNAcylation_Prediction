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
    if data_version in {'v1', 'v2'}:
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

    if data_version == 'v1':
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

    elif data_version == 'v2':
        # Plotting - Our MLP v2 data
        x_values = [73.68,  67.54,  57.89,  45.61,  38.60,  31.58,  21.93,  14.91, 14.91, 14.91,  14.91, 12.28, 12.28, 12.28, 12.28, 12.28,     12.28, 11.40, 11.40,  09.65, 09.65, 09.65, 09.65,    09.65,     08.77] # Recall
        y_values = [01.59,  01.80,  02.07,  02.29,  03.02,  04.06,  05.49,  07.17, 13.18, 17.71,  20.73, 21.54, 29.79, 35.00, 36.84, 37.84,     38.89, 39.39, 40.62,  44.00, 47.83, 57.89, 73.33,    78.57,     83.33] # Precision
        labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3',    '',    '', '0.05', '0.1',    '',    '',    '',    '', '0.7-0.8',    '', '0.9', '0.95',    '',    '',    '', '0.9999', '0.99995']
        #labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.03', '0.05', '0.1', '0.2', '0.3', '0.35-0.45', '0.5-0.65', '0.7-0.8', '0.85', '0.9', '0.95', '0.97', '0.99', '0.999', '0.9999', '0.99995']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our MLP')
        for idx in range(len(labels)):
            #my_label = ax.annotate(labels[idx], (x_values[idx]+X_annotation_space[idx], y_values[idx]+y_annotation_space[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'right', color = ANN_line[0].get_color())
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_76 (Full data)
        x_values = [74.56,  64.04,  57.89,  47.37,  38.60,  28.95, 24.56,  19.30,  16.67,  14.91, 14.91, 14.91, 14.04, 14.04, 14.04, 14.04,     12.28, 12.28, 11.40, 11.40,  11.40,   10.53,    07.89]
        y_values = [01.64,  01.78,  02.15,  02.54,  03.24,  04.02, 06.21,  09.87,  18.27,  24.29, 26.98, 32.08, 35.56, 42.11, 45.71, 47.06,     45.16, 56.00, 61.90, 65.00,  86.67,   92.31,    90.00]
        labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5',    '', '1e-3', '0.01', '0.03',    '', '0.1', '0.2',    '',    '', '0.5', '0.6-0.7', '0.8', '0.9',    '', '0.99', '0.999', '0.9999']
        #labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.03', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6-0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-76', color = '#A20698')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_152 (Full data)
        x_values = [70.18,  61.40,  53.51,  42.11,  33.33,  30.70,  25.44,  21.93,  18.42, 18.42,  18.42, 15.79, 14.91, 14.04, 14.04, 14.04, 13.16, 13.16, 12.28,      11.40, 09.65, 08.77,     07.89]
        y_values = [01.77,  01.96,  02.27,  02.58,  03.21,  04.99,  07.14,  11.06,  17.95, 24.14,  30.43, 30.51, 35.42, 40.00, 47.06, 48.48, 51.72, 55.56, 63.64,      65.00, 78.57, 83.33,     90.00]
        labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01',    '', '0.05', '0.1',    '', '0.3',    '', '0.5', '0.6', '0.7', '0.8', '0.9-0.95',    '',    '', '0.99995']
        #labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.03', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9-0.95', '0.99-0.999', '0.9999', '0.99995']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-152', color = '#FFA800')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_228 (Full data)
        x_values = [64.91,  53.51,  48.25,  42.11,  33.33, 29.82,  25.44,  22.81, 19.30,  17.54,  17.54, 15.79, 15.79, 15.79, 14.91, 14.04, 14.04, 13.16, 13.16, 11.40, 10.53,   10.53,   09.65,    07.02]
        y_values = [01.99,  02.11,  02.56,  03.33,  04.25, 06.12,  09.32,  14.44, 20.37,  23.81,  26.67, 30.00, 36.73, 40.00, 44.74, 47.06, 53.33, 55.56, 57.69, 54.17, 57.14,   70.59,   78.57,    80.00]
        labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6',    '', '1e-4', '1e-3',    '', '0.03', '0.05', '0.1',    '', '0.3', '0.4', '0.5', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999']
        #labels = ['1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.03', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-228', color = '#00B207')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_76 (Sequence only, new)
        x_values = [35.09, 34.21, 32.46, 31.58, 29.82,  28.95, 26.32, 24.56,  22.81,  22.81, 21.93,  21.05, 21.05, 21.05, 21.05, 21.05, 21.05,     21.05, 20.18, 20.18, 20.18, 20.18,  20.18, 19.30,   19.30,    16.67]
        y_values = [03.30, 03.64, 04.02, 04.68, 05.35,  06.06, 06.70, 07.41,  08.72,  10.61, 12.25,  14.04, 17.39, 19.05, 20.00, 21.43, 22.22,     22.86, 23.00, 23.96, 24.21, 25.00,  27.06, 27.85,   29.73,    27.54]
        labels = ['1e-14',    '',    '',    '',    '', '1e-9',    '',    '', '1e-6', '1e-5',    '', '1e-3',    '',    '',    '',    '',    '', '0.4-0.5', '0.6',    '',    '',    '', '0.95',    '', '0.999', '0.9995']
        #labels = ['1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4-0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9995']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-76 (Seq only)', color = '#FF0000')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_76 (Sequence only, old)
        #x_values = [45.61,   45.61,   42.98,   38.60,   36.84,  35.09,  34.21,  30.70,  29.82,  28.95,  28.95,  28.07,  27.19,  27.19, 26.32, 25.44, 24.56, 24.56, 24.56, 24.56, 24.56, 24.56, 22.81,  22.81,  20.18,   16.67]
        #y_values = [02.74,   03.15,   03.45,   03.68,   04.13,  04.57,  05.28,  05.74,  06.53,  07.55,  08.85,  10.60,  11.79,  13.72, 14.02, 14.15, 14.58, 14.97, 15.56, 16.09, 16.87, 17.83, 17.57,  18.31,  19.17,   18.81]
        #labels = ['1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999']
        #ANN_line = plt.plot(x_values, y_values, '--', label = 'Our RNN-76 (Seq only) OLD')
        #for idx in range(len(labels)):
        #    my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
        #    my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN_152 (Sequence only)
        x_values = [41.23,   36.84, 35.09,   32.46, 28.07,  27.19,  23.68, 21.93,  20.18, 20.18, 20.18, 20.18,  20.18,  19.30, 19.30, 19.30, 19.30, 19.30, 19.30, 19.30, 19.30,     18.42, 18.42, 18.42,  18.42,   15.76, 14.91]
        y_values = [04.00,   04.27, 04.87,   05.60, 06.00,  07.16,  07.61, 08.42,  09.31, 11.22, 13.22, 15.44,  19.49,  21.15, 22.22, 23.66, 24.18, 24.44, 25.00, 26.19, 27.50,     27.27, 28.00, 31.34,  32.81,   36.00, 36.17]
        labels = ['1e-14', '1e-13',    '', '1e-11',    '', '1e-9', '1e-8',    '', '1e-6',    '',    '',    '', '0.01', '0.03',    '',    '',    '',    '',    '',    '', '0.6', '0.7-0.8',    '',    '', '0.99', '0.999',    '']
        #labels = ['1e-14', '1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.03', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7-0.8', '0.9', '0.95', '0.99', '0.999', '0.9995']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-152 (Seq only)', color = '#2300FF')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 1.26, 's', label = 'No model')

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
        # Plotting - Our RNN-75; window size = 10
        x_values = [70.81, 69.32,  68.49, 67.50,  66.33, 65.51,  63.85, 61.86,  60.86, 60.20, 59.87, 59.87, 59.37, 59.37, 59.37, 58.87, 58.71, 58.21, 58.04,  57.71, 56.22,   56.05, 55.06, 54.56, 53.57, 52.74, 51.74]
        y_values = [05.09, 05.41,  05.75, 06.12,  06.48, 06.80,  07.07, 07.21,  07.45, 07.58, 07.63, 07.72, 07.73, 07.81, 07.88, 07.88, 07.90, 07.89, 07.96,  08.00, 07.99,   08.25, 08.38, 08.64, 08.85, 09.10, 09.23]
        labels = ['1e-10',    '', '1e-8',    '', '1e-6',    '', '1e-4',    '', '0.01',    '',    '', '0.2',    '',    '', '0.5',    '',    '', '0.8',    '', '0.95',    '', '0.999',    '',    '',    '',    '',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Our RNN-75', color = '#FF0000')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Wulff-Fuentes et al. motif (page 3 of their paper) -- only 28 unique sites follow the motif; only 2 are O-GlcNAcylated
        plt.plot(0.06, 7.14, 'o', label = r'Wulff-Fuentes $\it{et}$ $\it{al.}$ motif')
        # No model / testing everything
        plt.plot(100, 0.524, 's', label = 'No model')

    # F curves
    rec = np.linspace(0.030, 1, 195)
    beta = 1
    x_pos = 100.3
    # These y_pos and extra rec points are manually set up
    if beta == 1:
        y_pos = [1.8, 4.4, 7.2, 10.2, 13.4, 16.7, 20.3, 24.1]
        x_pos_top = [1.9, 4.3, 7.3, 10.3, 13.5, 16.9, 20.4, 24.2]
        rec = np.sort( np.concatenate(([0.0255, 0.026, 0.027, 0.028, 0.029, 0.052, 0.053, 0.054], rec)) ) # The F = 0.05 and F = 0.10 curves need a few additional points
    elif beta == 2:
        y_pos = [0, 1.3, 0, 3.8, 0, 6.9, 0, 11.0]
        x_pos_top = [3.3, 7.3, 11.2, 15.5, 20.1, 24.7, 29, 33.9]
        rec = np.sort( np.concatenate((np.linspace(0.0402, 0.0408, 4), np.linspace(0.041, 0.044, 4), [0.046, 0.047], np.linspace(0.081, 0.084, 4), [0.123, 0.124], rec)) ) # The F = 0.05, 0.10, and 0.15 curves need a few additional points
    # Calculating the precision to get a given F score at each recall point
    for idx, F_score in enumerate([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]):
        pre = F_score*rec / (rec + (beta**2*rec) - F_score*beta**2)
        plt.plot(rec[pre>0]*100, pre[pre>0]*100, ':k') # Plotting >0 to avoid impossible numbers (because recall is too low)
        if 'y_pos' in locals() and y_pos[idx] > 0: # Right-side labels for F1 isolines
            plt.text(x_pos, y_pos[idx], f'{round(F_score*100)}%', fontsize = 16)
        if 'x_pos_top' in locals(): # Upper-side labels for F1 isolines
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
