#import numpy as np
import re
import matplotlib.pyplot as plt

# Define the file path
file_path = 'results_650_b128_1024_1024_1024/res.res'

# Initialize an empty list to store the data
data_list = []

# Open the file for reading
with open(file_path, 'r') as file:
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

# Plotting loss curve)
plt.figure(1)
plt.plot(epochs, loss_tr, label='Training Loss')
plt.plot(epochs, loss_te, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()

# Save the figure as a PNG file
plt.savefig('loss_curve.png')

# Plotting error curve
plt.figure(2)
plt.plot(epochs, err_tr, label='Training error')
plt.plot(epochs, err_te, label='Testing error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Identification error in training and testing over epochs')
plt.legend()

# Save the figure as a PNG file
plt.savefig('identification_error.png')




