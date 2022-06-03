# Dependencies
pytorch == 1.10.0 scikit-learn == 1.0.2

# Dataset
We made our experiments on MIMIC-III dataset. In order to access the dataset, please refer to https://mimic.physionet.org/gettingstarted/access/ .

After downloading MIMIC-III, you can use the following github repo for the pre-processing: https://doi.org/10.5281/zenodo.1306527 .
# Main Entrance 
train_riskpre.py and train_neuralhmm.py contain training code. test_riskpre.py and test_neuralhmm.py contain evaluation code.
