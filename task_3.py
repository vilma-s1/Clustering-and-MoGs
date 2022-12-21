import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

rows = np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2)  # counts all the occurances of phonemes 1 and 2
X_phonemes_1_2 = np.zeros((rows, 2))  # initialises the X_phoneme_1_2 array
j = 0
# ground truth will be used to calculate the accuracy in prediction of the models
ground_truth = np.zeros(rows)  # initialises the ground_truth array

for i in range(0, len(phoneme_id)):
    if phoneme_id[i] == 1:  # checks if a phoneme is phoneme 1
        X_phonemes_1_2[j, :] = X_full[i, :]  # stores phoneme 1 data samples in X_phoneme_1_2
        ground_truth[j] = 1  # stores the phoneme_id label in the ground_truth whenever the required condition is met
        j += 1  # counter for the rows in X_phoneme_1_2
    elif phoneme_id[i] == 2:  # checks if a phoneme is phoneme 2
        X_phonemes_1_2[j, :] = X_full[i, :]  # stores phoneme 2 data samples in X_phoneme_1_2
        ground_truth[j] = 2  # stores the phoneme_id label in the ground_truth whenever the required condition is met
        j += 1

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

# loading the phoneme 1 GMM
phoneme1_model = 'data/GMM_params_phoneme_01_k_{:02}.npy'.format(k)
phoneme1_model = np.load(phoneme1_model, allow_pickle=True)
phoneme1_model = np.ndarray.tolist(phoneme1_model)
# setting the model parameters
mu_phoneme1_model = phoneme1_model['mu']
s_phoneme1_model = phoneme1_model['s']
p_phoneme1_model = phoneme1_model['p']
# computing the posterior probabilities using the get_predictions method
predictions_phoneme1_model = get_predictions(mu_phoneme1_model, s_phoneme1_model, p_phoneme1_model, X_phonemes_1_2)


predictions1 = np.zeros(len(X_phonemes_1_2))
# adds the posterior probabilities in each row to obtain the likelihood of the samples belonging to class phoneme 1
for i in range(0, len(X_phonemes_1_2)):
    for j in range(0, k):
        predictions1[i] += predictions_phoneme1_model[i, j]



# loading the phoneme 2 GMM
phoneme2_model = 'data/GMM_params_phoneme_02_k_{:02}.npy'.format(k)
phoneme2_model = np.load(phoneme2_model, allow_pickle=True)
phoneme2_model = np.ndarray.tolist(phoneme2_model)
# setting the model parameters
mu_phoneme2_model = phoneme2_model['mu']
s_phoneme2_model = phoneme2_model['s']
p_phoneme2_model = phoneme2_model['p']
# computing the posterior probabilities using the get_predictions method
predictions_phoneme2_model = get_predictions(mu_phoneme2_model, s_phoneme2_model, p_phoneme2_model, X_phonemes_1_2)


predictions2 = np.zeros(len(X_phonemes_1_2))
# adds the posterior probabilities in each row to obtain the likelihood of the samples belonging to class phoneme 2
for i in range(0, len(X_phonemes_1_2)):
    for j in range(0, k):
        predictions2[i] += predictions_phoneme2_model[i, j]


predictions = np.zeros(len(X_phonemes_1_2))

for i in range(0, len(X_phonemes_1_2)):
    # assigns class 1 (phoneme 1) to a data sample if the probability of being in class 1 is greater than class 2
    if predictions1[i] > predictions2[i]:
        predictions[i] = 1
    # assigns class 2 (phoneme 2) to a data sample otherwise
    else:
        predictions[i] = 2


incorrect_predictions = 0
# counts the number of times a data sample is misclassified
for i in range(0, rows):
    if predictions[i] != ground_truth[i]:
        incorrect_predictions += 1

# computes mis-classification error, i.e., ratio of points classified incorrectly out of all the data samples
misclassification_error = incorrect_predictions/len(X_phonemes_1_2)
# computes accuracy
accuracy = (1 - misclassification_error)*100


########################################/
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()