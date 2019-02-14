import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns


# Parameters
data_folder = './data/'


# Load data
x_tr = pd.read_csv(data_folder + 'training.csv', index_col=[0])
y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])

x_te = pd.read_csv(data_folder + 'input_test.csv', index_col=[0])

label_pos = (y_tr['TARGET'] == 1)
label_neg = (y_tr['TARGET'] == 0)


# Explore columns
print(x_tr.columns)
print(y_tr.columns)

print(x_tr[label_pos].describe())
print(x_tr[label_neg].describe())

unique_neurons = set(x_tr['neuron_id'].values)
unique_pos_neurons = set(x_tr[label_pos]['neuron_id'].values)
unique_neg_neurons = set(x_tr[label_neg]['neuron_id'].values)
unique_bth_neurons = unique_pos_neurons.intersection(unique_neg_neurons)
print('Number of unique neurons:', len(unique_neurons))
print('Number of unique positive neurons:', len(unique_pos_neurons))
print('Number of unique negatve neurons:', len(unique_neg_neurons))
print('Number of unique neurons that can be both:', len(unique_bth_neurons))

unique_test_neurons = set(x_te['neuron_id'].values)
print('Number of unique neurons in test set:', len(unique_test_neurons))
print('Number of unique neurons both in training and test sets:', len(unique_test_neurons.intersection(unique_neurons)))


# Plot spikes
plot_spikes = False
if plot_spikes:
    colors = np.array(['blue', 'red'])
    neuron_to_plot = x_tr['neuron_id'].isin(random.sample(list(unique_neurons), 1))
    to_plot = x_tr[neuron_to_plot].sort_values(by='timestamp_49').index
    # to_plot = x_tr['neuron_id'] == list(unique_bth_neurons)[0]
    plt.eventplot(x_tr.loc[to_plot].iloc[:, 1:].values, color=colors[y_tr.loc[to_plot].values])
    plt.xlabel('Neuron')
    plt.ylabel('Spike')
    plt.title('Spike raster plot')
    plt.show()


# See densities
plot_timestamp_hist = False
if plot_timestamp_hist:
    neg_timestamps = x_tr[y_tr['TARGET'] == 0].iloc[:, 20].values.flatten()
    pos_timestamps = x_tr[y_tr['TARGET'] == 1].iloc[:, 20].values.flatten()
    plt.hist(neg_timestamps, color='blue', bins=np.arange(0, 50, 0.5), alpha=0.5, density=True)
    plt.hist(pos_timestamps, color='red', bins=np.arange(0, 50, 0.5), alpha=0.5, density=True)
    plt.show()


# Jointplot of first and last timestamps
sns.jointplot(x='timestamp_25', y='timestamp_49', data=x_tr[y_tr['TARGET'] == 0], color='blue', kind='hex')
sns.jointplot(x='timestamp_25', y='timestamp_49', data=x_tr[y_tr['TARGET'] == 1], color='red', kind='hex')
plt.show()