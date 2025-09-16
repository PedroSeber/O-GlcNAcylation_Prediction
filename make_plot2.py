import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

def main(plot_type):
    """
    Generates Precision-Recall curves for the models trained on different conditions

    Parameters
    ----------
    plot_type : string
        The plot to be generated. The possibilities are:
        'cross_entropy' = Transformers trained with the weighted cross-entropy (CE) loss function. Includes previous models from other works.
        'diffmcc' = Transformers and RNNs trained with the weighted focal differentiable MCC loss function or first trained with CE then fine-tuned.
        'multilayer' = RNNs with two layers that were either trained with either loss function or first trained with CE then fine-tuned.
        'nested' = A nested validation with the best model from this work, a multilayer RNN trained with the weighted focal differentiable MCC loss function.
    """
    plot_type = plot_type.casefold() # Avoiding issues with letter case
    # Plotting settings
    plt.rcParams.update({'font.size': 24, 'lines.markersize': 10})
    annotation_fontsize = 6
    bbox_dict = dict(facecolor='white', alpha=1, edgecolor='white', linewidth = 0, pad = 0.15)
    fig, ax = plt.subplots(figsize = (16, 9), dpi = 500)
    if plot_type.casefold() == 'transformer':
        # Plotting - YoY
        x_values = [100.0, 96.21, 72.74, 48.30, 28.19, 13.59,  4.22,  0.26]
        y_values = [ 2.43,  2.50,  3.12,  4.18,  5.69,  8.08, 11.98, 18.56]
        labels = [ '0.12', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
        yoy_line = plt.plot(x_values, y_values, '-', label = 'YinOYang')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = yoy_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Seber and Braatz RNN-75; window size = 10
        x_values = [70.79, 64.24, 57.84, 52.97, 48.64, 44.88, 41.16, 38.8 , 37.8 , 36.65, 35.65, 34.79, 34.25, 33.5 , 32.75, 31.68, 30.42, 29.35, 26.84, 22.83, 18.54,  8.77]
        y_values = [ 4.8 ,  5.84,  7.06,  8.74, 10.74, 13.07, 15.65, 17.48, 18.39, 19.44, 20.01, 20.55, 21.13, 21.54, 22.06, 22.46, 23.33, 24.22, 26.09, 27.91, 29.07, 34.36]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-75; 10 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Seber and Braatz RNN-225; window size = 20
        x_values = [62.92,  56.41,  51.04,  47.14,  44.42,  42.13,  40.3 ,  38.55, 37.83, 37.08, 36.58, 36.22, 35.83, 35.47, 34.9 , 34.29, 33.39,  32.82,  31.1 ,   28.31,    25.41, 14.03]
        y_values = [ 5.87,   8.57,  11.96,  15.89,  20.43,  24.81,  29.4 ,  32.08, 33.16, 34.3 , 35.11, 35.86, 36.37, 36.9 , 37.43, 37.84, 38.81,  39.75,  41.56,   43.39,    45.81, 53.04]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-225; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our Transformer; window size = 5
        x_values = [  100,  50.93,  41.30,  37.47, 35.76, 32.93, 28.92, 25.95, 23.37, 20.54, 17.47, 14.71, 12.06]
        y_values = [ 2.50,   6.07,   7.82,   8.91,  9.46, 10.58, 11.51, 12.83, 14.23, 15.63, 16.72, 17.62, 19.93]
        labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        #labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Transformer-15;  5 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our Transformer; window size = 10
        x_values = [  100,  41.62,  34.29,  30.78, 29.24, 26.06, 21.44, 16.86, 10.95,  5.94,  3.29,  1.90,  0.82]
        y_values = [ 2.50,   8.53,  11.00,  12.11, 12.62, 13.77, 15.47, 18.65, 22.47, 25.50, 26.90, 29.44, 32.39]
        labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        #labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Transformer-15; 10 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our Transformer; window size = 15
        x_values = [  100,  51.47,  44.42,  41.30, 39.84, 35.93, 32.78, 29.28, 26.02, 22.37, 19.29, 17.29, 15.32]
        y_values = [ 2.50,   8.39,  11.47,  13.00, 13.80, 15.24, 17.38, 19.43, 21.28, 22.95, 24.87, 26.82, 29.25]
        labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        #labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Transformer-15; 15 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our Transformer; window size = 20
        x_values = [  100,  51.18,  46.49,  43.63, 42.02, 40.19, 37.80, 34.36, 30.89, 27.24, 23.87, 21.76, 19.58]
        y_values = [ 2.50,   9.10,  11.28,  12.50, 13.11, 14.07, 15.77, 17.76, 20.04, 21.63, 22.76, 24.17, 25.92]
        labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        #labels =  ['1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Transformer-15; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our Transformer; window size = 20; differentiable MCC
        x_values = [  100, 99.36, 93.45, 53.90, 29.21, 25.70, 24.30, 23.44, 22.62, 22.12, 21.15]
        y_values = [ 2.50,  2.68,  3.19,  7.11, 20.07, 25.48, 26.30, 26.73, 27.05, 27.66, 28.22]
        labels =  ['0.01',    '', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',    '', '0.8', '0.9']
        #labels = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Transformer-240; 20 win; diff MCC')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 2.44, 's', label = 'No model')

    elif plot_type.casefold() in {'finetune', 'transformer-finetune', 'transformer_finetune'}:
        plot_type = 'transformer-finetune'

        # Plotting - Finetuned ProteinBERT; window size = 10
        x_values = [100.00,  96.37,  48.48,  34.05,  23.91,  17.95,  13.32,   9.73,   7.22,   4.77,   2.70,   0.74,   0.15]
        y_values = [  2.34,   2.73,  10.26,  18.74,  31.62,  41.24,  49.05,  55.48,  64.14,  70.49,  73.74,  90.91, 100.00]
        labels =   ['1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95']
        # labels =   ['1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Finetuned ProteinBERT; 10 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Finetuned ProteinBERT; window size = 15
        x_values = [100.00,  99.64,  86.55,  50.54,  42.95,  35.10,  30.33,  26.32,  22.70,  18.73,  14.89,  11.03,   6.15,   2.93,   0.36]
        y_values = [  2.35,   2.42,   3.77,  13.31,  21.00,  30.71,  38.42,  44.91,  50.69,  55.16,  61.13,  66.02,  70.54,  86.17, 100.00]
        labels =   ['1e-4', '1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        #labels =   ['1e-4', '1e-3', '0.01', '0.05', '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Finetuned ProteinBERT; 15 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Finetuned ProteinBERT; window size = 20
        x_values = [100.00,  99.64,  97.26,  72.17,  40.23,  30.57,  22.24,  17.05,  13.45,  10.56,   7.97,   5.88,   3.68,   1.66,   0.47,   0.04]
        y_values = [  2.35,   2.38,   2.76,   5.74,  20.25,  33.97,  48.74,  56.99,  63.98,  67.51,  74.66,  79.13,  85.71,  95.83,  92.86, 100.00]
        labels =   ['1e-5', '1e-4', '1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        #labels =   ['1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Finetuned ProteinBERT; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Finetuned ProteinBERT; window size = 25
        x_values = [100.00,  99.19,  78.91,  38.75,  30.71,  23.25,  18.68,  14.69,  11.82,   8.46,   5.59,   3.08,   0.89,   0.18,   0.03]
        y_values = [  2.36,   2.50,   4.40,  21.60,  33.62,  45.91,  53.39,  58.20,  64.73,  69.27,  74.53,  82.86,  89.29,  71.43, 100.00]
        labels =   ['1e-4', '1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        #labels =   ['1e-4', '1e-3', '0.01', '0.05', '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95', '0.99']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Finetuned ProteinBERT; 25 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Finetuned ProteinBERT; window size = 30
        x_values = [100.00,  99.82,  88.60,  50.24,  40.91,  32.02,  26.06,  21.16,  16.23,  12.27,   8.57,   4.32,   1.09,   0.07]
        y_values = [  2.29,   2.32,   3.72,  13.89,  23.50,  34.95,  42.38,  49.20,  53.53,  60.25,  68.41,  73.46,  88.23, 100.00]
        labels =   ['1e-4', '1e-3', '0.01', '0.05',  '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95']
        #labels =   ['1e-4', '1e-3', '0.01', '0.05', '0.1',  '0.2',  '0.3',  '0.4',  '0.5',  '0.6',  '0.7',  '0.8',  '0.9', '0.95']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Finetuned ProteinBERT; 30 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Seber and Braatz RNN-225; window size = 20; CE
        x_values = [62.92,  56.41,  51.04,  47.14,  44.42,  42.13,  40.3 ,  38.55, 37.83, 37.08, 36.58, 36.22, 35.83, 35.47, 34.9 , 34.29, 33.39,  32.82,  31.1 ,   28.31,    25.41, 14.03]
        y_values = [ 5.87,   8.57,  11.96,  15.89,  20.43,  24.81,  29.4 ,  32.08, 33.16, 34.3 , 35.11, 35.86, 36.37, 36.9 , 37.43, 37.84, 38.81,  39.75,  41.56,   43.39,    45.81, 53.04]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-225; 20 win')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 2.35, 's', label = 'No model')

    elif 'single' in plot_type.casefold() and ('layer' in plot_type.casefold() or 'rnn' in plot_type.casefold()):
        plot_type = 'single-layer'

        # Plotting - Seber and Braatz RNN-225; window size = 20; CE
        x_values = [62.92,  56.41,  51.04,  47.14,  44.42,  42.13,  40.3 ,  38.55, 37.83, 37.08, 36.58, 36.22, 35.83, 35.47, 34.9 , 34.29, 33.39,  32.82,  31.1 ,   28.31,    25.41, 14.03]
        y_values = [ 5.87,   8.57,  11.96,  15.89,  20.43,  24.81,  29.4 ,  32.08, 33.16, 34.3 , 35.11, 35.86, 36.37, 36.9 , 37.43, 37.84, 38.81,  39.75,  41.56,   43.39,    45.81, 53.04]
        labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2',    '', '0.4',    '', '0.6',    '', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-225; 20 win; CE')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-1475; window size = 20; differentiable MCC
        x_values = [95.13, 83.93, 45.38, 34.11, 32.32, 31.39, 30.67, 29.67, 27.81,  26.41,  23.37,   20.01]
        y_values = [ 3.05,  3.70, 10.34, 35.32, 42.55, 45.02, 46.70, 48.23, 50.16,  51.50,  53.57,   56.24]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-1475; 20 win; diff MCC')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Seber and Braatz RNN-225; window size = 20; CE then differentiable MCC
        x_values = [97.10, 42.09,  34.04, 31.93, 30.35, 29.46, 28.74, 27.99, 27.06, 25.73,  24.59,  22.19,   18.93,    15.68,  4.44]
        y_values = [ 2.85, 12.13,  33.03, 41.53, 45.84, 47.82, 49.32, 50.39, 51.82, 53.54,  54.78,  57.09,   61.23,    64.41, 68.51]
        labels =   ['0.1', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999',   '1']
        #labels =  ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-225; 20 win; CE then diff MCC')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 2.44, 's', label = 'No model')

    elif 'multi' in plot_type.casefold() and ('layer' in plot_type.casefold() or 'rnn' in plot_type.casefold()):
        plot_type = 'multi-layer'

        # Plotting - Our RNN-[450, 75]; window size = 20; CE
        x_values = [ 50.89,  48.46,  46.74,  45.13,  43.66,  41.84,  40.34, 39.48, 39.08, 38.73, 38.37, 38.12, 37.97, 37.58, 37.37, 36.97, 36.69, 35.90,  34.43,   32.5 ,    30.49, 20.62]
        y_values = [ 16.14,  19.19,  22.90,  26.40,  29.65,  32.25,  34.74, 36.66, 37.35, 38.10, 38.51, 38.88, 39.22, 39.58, 40.11, 40.49, 41.02, 41.22,  42.21,   43.76,    46.15, 53.63]
        labels =   ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01',    '', '0.1',    '', '0.3',    '', '0.5',    '', '0.7',    '', '0.9',    '', '0.99', '0.999', '0.9999',   '1']
        #labels =   ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-[450, 75]; 20 win; CE')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-[450, 75]; window size = 20; CE then differentiable MCC
        x_values = [97.14, 78.60, 42.95, 40.30, 37.62, 35.00, 33.82, 32.82, 31.35,  29.56,  27.20,   23.41,  8.16]
        y_values = [ 2.62,  3.64, 24.98, 35.54, 40.11, 43.03, 44.45, 45.9 , 47.02,  47.80,  49.71,   53.43, 66.28]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        #labels =   ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999', '0.9999', '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-[450, 75]; 20 win; CE then diff MCC')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Our RNN-[600, 75]; window size = 20; differentiable MCC
        x_values = [92.95, 87.58, 55.51, 38.12, 34.57, 33.11, 31.64, 30.49, 29.46,  28.99,  27.85,   26.23, 19.26]
        y_values = [ 3.42,  3.82,  8.39, 29.78, 43.11, 47.1 , 49.52, 50.68, 51.89,  53.15,  54.52,   55.66, 61.63]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'RNN-[600, 75]; 20 win; diff MCC')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # No model / testing everything
        plt.plot(100, 2.44, 's', label = 'No model')

    elif plot_type.casefold() == 'nested':
        # Plotting - Fold 1
        x_values = [92.95, 87.58, 55.51, 38.12, 34.57, 33.11, 31.64, 30.49, 29.46,  28.99,  27.85,   26.23, 19.26]
        y_values = [ 3.42,  3.82,  8.39, 29.78, 43.11, 47.1 , 49.52, 50.68, 51.89,  53.15,  54.52,   55.66, 61.63]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Fold 1')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Fold 2
        x_values = [87.66, 79.23, 52.32, 40.88, 37.83, 36.49, 35.15, 34.15, 33.00,  32.22,  30.32,   28.84, 19.18]
        y_values = [ 3.86,  4.54, 10.29, 23.45, 32.24, 38.3 , 42.14, 45.18, 47.84,  49.88,  51.91,   54.57, 61.72]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Fold 2')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Fold 3
        x_values = [91.24, 75.17, 42.13, 35.47, 33.22, 31.7 , 30.66, 29.79, 28.45,  27.72,  26.02,   24.10, 12.74]
        y_values = [ 3.49,  4.71, 17.12, 36.83, 43.42, 46.65, 50.24, 52.35, 53.84,  54.79,  56.13,   57.27, 61.43]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Fold 3')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Fold 4
        x_values = [90.57, 76.91, 41.64, 34.25, 32.4 , 31.01, 29.65, 28.63, 27.35,  26.78,  25.16,   23.16, 15.47]
        y_values = [ 3.35,  4.31, 15.9 , 35.92, 43.98, 47.96, 49.68, 51.25, 52.73,  53.91,  55.72,   57.12, 66.45]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Fold 4')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

        # Plotting - Fold 5
        x_values = [85.76, 70.05, 43.9 , 36.78, 34.4 , 33.09, 32.21, 31.23, 29.88,  28.89,  27.43,   25.75, 18.55]
        y_values = [ 3.9 ,  5.31, 15.77, 30.07, 39.43, 44.41, 47.62, 50.24, 51.84,  53.55,  55.63,   58.07, 65.38]
        labels =   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '0.999',   '1']
        ANN_line = plt.plot(x_values, y_values, '-', label = 'Fold 5')
        for idx in range(len(labels)):
            my_label = ax.annotate(labels[idx], (x_values[idx], y_values[idx]), fontsize = annotation_fontsize, va = 'center', ha = 'center', color = ANN_line[0].get_color())
            my_label.set_bbox(bbox_dict)

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
    if plot_type not in {'BLANK'}:
        ax.legend(fontsize = 22)
    plt.tight_layout()
    plt.savefig(f'O-Gly_model_eval_F{beta}_{plot_type}-data.svg', bbox_inches = 'tight')

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Generates Precision-Recall curves for the models trained on different conditions')
    parser.add_argument('plot_type', type = str, nargs = 1, help = 'The plot to be created. Should be in {"ce", "diffmcc", "multilayer", "nested"}')
    my_args = parser.parse_args()
    main(my_args.plot_type[0])
