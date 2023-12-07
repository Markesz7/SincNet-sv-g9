#import numpy as np
import re
import matplotlib.pyplot as plt

# Define the file path
file_path_SincNet = 'results_SincNet_500_b128/res.res'
file_path_CNN = 'results_CNN_500_b128/res.res'

# Initialize an empty list to store the data
data_list = []

# Open the file for reading
with open(file_path_SincNet, 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line into columns based on spaces
        columns = line.split()

        # Create a dictionary for the current line
        entry = {
            'epoch': int(columns[1][:-1]),  # Remove the trailing comma from the epoch value
            'loss_tr': float(re.search(r'[\d.]+', columns[2]).group()),
            'err_tr': float(re.search(r'[\d.]+', columns[3]).group()),
            'loss_te': float(re.search(r'[\d.]+', columns[4]).group()),
            'err_te': float(re.search(r'[\d.]+', columns[5]).group()),
            'err_te_snt': float(re.search(r'[\d.]+', columns[6]).group())
        }
        
        # Append the dictionary to the list
        data_list.append(entry)

# Now, data_list contains a list of dictionaries, where each dictionary represents information for a specific epoch
for entry in data_list:
    print(entry)

    # Extract data for plotting
    epochs = [entry['epoch'] for entry in data_list]
    loss_tr = [entry['loss_tr'] for entry in data_list]
    loss_te = [entry['loss_te'] for entry in data_list]
    err_tr = [entry['err_tr'] for entry in data_list]
    err_te = [entry['err_te'] for entry in data_list]
    err_te_snt = [entry['err_te_snt'] for entry in data_list]

file.close()

# Initialize an empty list to store the data
data_list_CNN = []

# Open the file for reading
with open(file_path_CNN, 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line into columns based on spaces
        columns_CNN = line.split()

        # Create a dictionary for the current line
        entry_CNN = {
            'epoch': int(columns_CNN[1][:-1]),  # Remove the trailing comma from the epoch value
            'loss_tr': float(re.search(r'[\d.]+', columns_CNN[2]).group()),
            'err_tr': float(re.search(r'[\d.]+', columns_CNN[3]).group()),
            'loss_te': float(re.search(r'[\d.]+', columns_CNN[4]).group()),
            'err_te': float(re.search(r'[\d.]+', columns_CNN[5]).group()),
            'err_te_snt': float(re.search(r'[\d.]+', columns_CNN[6]).group())
        }
        
        # Append the dictionary to the list
        data_list_CNN.append(entry_CNN)

# Now, data_list contains a list of dictionaries, where each dictionary represents information for a specific epoch
for entry_CNN in data_list_CNN:
    print(entry_CNN)

    # Extract data for plotting
    epochs_CNN = [entry_CNN['epoch'] for entry_CNN in data_list_CNN]
    loss_tr_CNN = [entry_CNN['loss_tr'] for entry_CNN in data_list_CNN]
    loss_te_CNN = [entry_CNN['loss_te'] for entry_CNN in data_list_CNN]
    err_tr_CNN = [entry_CNN['err_tr'] for entry_CNN in data_list_CNN]
    err_te_CNN = [entry_CNN['err_te'] for entry_CNN in data_list_CNN]
    err_te_snt_CNN = [entry_CNN['err_te_snt'] for entry_CNN in data_list_CNN]

file.close()

# Plotting loss curve)
plt.figure(1)
plt.plot(epochs, loss_tr, label='Training Loss for SincNet')
plt.plot(epochs, loss_tr_CNN, color='green', label='Training Loss for CNN')
plt.plot(epochs, loss_te, label='Testing Loss for SincNet')
plt.plot(epochs, loss_te_CNN, color='black', label='Testing Loss for CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()

# Save the figure as a PNG file
plt.savefig('loss_curve_combined.png')

# Plotting error curve
plt.figure(2)
plt.plot(epochs, err_tr, label='Training error for SincNet')
plt.plot(epochs, err_tr_CNN, color='green',label='Training error for CNN')
plt.plot(epochs, err_te, label='Testing error for SincNet')
plt.plot(epochs, err_te_CNN, color='black',label='Testing error for CNN')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Identification error in training and testing over epochs')
plt.legend()

# Save the figure as a PNG file
plt.savefig('identification_error_combined.png')

# Plotting sentence error rate curve
plt.figure(3)
plt.plot(epochs, err_te_snt, label='Sentence error rate for SincNet')
plt.plot(epochs, err_te_snt_CNN, color='black',label='Sentence error rate for CNN')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Sentence error rate for testing over epochs')
plt.legend()

# Save the figure as a PNG file
plt.savefig('sentence_error_rate_combined.png')




