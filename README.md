## Recurrent Neural Network-based Prediction of the Location of O-GlcNAcylation Sites in Mammalian Proteins
These are the datasets and model files associated with the publication [Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins](https://doi.org/10.1101/2023.08.24.554563). This work uses primarily RNN models to predict the presence of O-GlcNAcylation sites.
The models were trained on multiple sources of literature data on protein O-GlcNAcylation based on human-selected descriptors (v1 of the datasets) or protein sequences (v3 and v5 of the datasets).<br>

### Reproducing the models and plots
The models can be recreated by downloading the datasets and running the [ANN\_train.py](ANN_train.py) file with the appropriate flags (run `python ANN_train.py --help` for details).

The plots can be recreated by running the [make\_plot.py](make_plot.py) file with the appropriate data version as an input (`python make_plot.py v1` for Fig. 1, `python make_plot.py v5` for Fig. 2, `python make_plot.py shap_heatmap` for Fig. 3, `python make_plot.py v3` for Fig. S1, `python make_plot.py v5_nested` for Fig. S2, and `python make_plot.py v5_shap` for Fig. S3).

### Using the models to predict O-GlcNAcylation sites
The Conda environment defining the specific packages and version numbers used in this work is available as [ANN\_environment.yaml](ANN_environment.yaml). To use our trained model, run the [Predict.py](Predict.py) file as `python Predict.py <sequence> -t <threshold> -bs <batch_size>`.

Alternatively, create an (N+1)x1 .csv with the first row as a header (such as "Sequences") and all other N rows as the actual amino acid sequences, then run the [Predict.py](Predict.py) file as `python ANN_predict.py <path/to/file.csv> -t <threshold> -bs <batch_size>`.
Results will be saved as a new .csv file.

To run Shapley value predictions in addition to the model predictions, run the [Predict.py](Predict.py) file with also the `-shap` flag. Whatever other flags should be included as needed.
