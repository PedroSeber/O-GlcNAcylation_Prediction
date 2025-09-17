## Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins
These are the datasets and model files associated with the publications [Predicting O-GlcNAcylation sites in mammalian proteins with transformers and RNNs trained with a new loss function](https://arxiv.org/abs/2402.17131) (our newer work on this topic) and [Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins](https://doi.org/10.1016/j.compchemeng.2024.108818) (our older work). These works use primarily RNN (specifically, LSTM) models to predict the presence of O-GlcNAcylation sites.
The models from [Predicting O-GlcNAcylation sites in mammalian proteins with transformers and RNNs trained with a new loss function](https://arxiv.org/abs/2402.17131) were trained on the largest O-GlcNAcylation available [dataset](https://doi.org/10.1038/s41597-021-00810-4), which is equivalent to the v5 data of our older work. The models from [Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins](https://doi.org/10.1016/j.compchemeng.2024.108818) were trained on multiple sources of literature data on protein O-GlcNAcylation based on human-selected descriptors (v1 of the datasets) or protein sequences (v3 and v5 of the datasets).

### Reproducing the models and plots
The models can be recreated by downloading the datasets and running the [ANN\_train.py](ANN_train.py) file with the appropriate flags (run `python ANN_train.py --help` for details).

The plots from [Predicting O-GlcNAcylation sites in mammalian proteins with transformers and RNNs trained with a new loss function](https://arxiv.org/abs/2402.17131) can be recreated by running the [make\_plot2.py](make_plot2.py) file with the appropriate flag as an input (`python make_plot2.py finetune`} for Fig. 1, `python make_plot2.py single-layer` for Fig. 2, `python make_plot2.py multilayer` for Fig. 3, `python make_plot2.py transformer` for Fig. S1, and `python make_plot2.py nested` for Fig. S2.

The plots from [Recurrent Neural Network-based Prediction of O-GlcNAcylation Sites in Mammalian Proteins](https://doi.org/10.1016/j.compchemeng.2024.108818) can be recreated by running the [make\_plot.py](make_plot.py) file with the appropriate data version as an input (`python make_plot.py v1` for Fig. 1, `python make_plot.py v5` for Fig. 2, `python make_plot.py shap_heatmap` for Fig. 3, `python make_plot.py v3` for Fig. S1, `python make_plot.py v5_nested` for Fig. S2, and `python make_plot.py v5_shap` for Fig. S3).

### Using the models to predict O-GlcNAcylation sites
The Conda environment defining the specific packages and version numbers used in this work is available as [ANN\_environment.yaml](ANN_environment.yaml). To use our trained model, run the [Predict.py](Predict.py) file as `python Predict.py <sequence> -t <threshold> -bs <batch_size>`. The [Predict.py](Predict.py) file will attempt to use our newer model ([RNN-[600,75]\_20-window\_dict.pt](RNN-[600,75]_20-window_dict.pt)) if available; else, it will default to the older model ([RNN-225\_20-window\_dict.pt](RNN-225_20-window_dict.pt)).

Alternatively, create an (N+1)x1 .csv with the first row as a header (such as "Sequences") and all other N rows as the actual amino acid sequences, then run the [Predict.py](Predict.py) file as `python ANN_predict.py <path/to/file.csv> -t <threshold> -bs <batch_size>`.
Results will be saved as a new .csv file.

To run Shapley value predictions in addition to the model predictions, run the [Predict.py](Predict.py) file with also the `-shap` flag. Whatever other flags should be included as needed.

### Citations
If you have used any of the models in this work, please cite both works. Bibtex-formatted citations are available in [citation.bib](citation.bib).
