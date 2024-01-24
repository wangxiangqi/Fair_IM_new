#import torch

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./Loss_of_probability/NBA/wel_func/alpha_2/')

data_list = []

pref="./Loss_of_probability/NBA/wel_func/alpha_2/"

file_names = ['Loss-IMFB.npy', 'Loss-LinUCB.npy', 'Loss-CUCB.npy', 'Loss-IMFB-FW.npy', 'Loss-LinUCB-FW.npy', 'Loss-CUCB-FW.npy']  # Replace with your file names

plt.subplot(4,5,2)
#file_names = ['Loss-IMFB.npy']
def is_one_dimensional(lst):
    for element in lst:
        if isinstance(element, list):
            return False
    return True

for file_name in file_names:
    data = np.load(pref+file_name).tolist()
    if is_one_dimensional(data)==False:
        #print(data)
        first_elements = [row[0] for row in data]
        data=first_elements
    data_list.append(data)
#fig = plt.figure(figsize=(8, 6))
# Assuming the data in each file is a 1D array
for i, data in enumerate(data_list):
    plt.plot(data, label=''.join(list(file_names[i])[5:-4]), linewidth=1.0)

#fig=plt.figure(figuresize=(5,3))
plt.ylabel("{}".format("Estimation Error"))
#plt.tick_params(axis="both",label_size=14)
plt.legend(prop={"size":5})  # Add legend
plt.xlabel("{}".format("Iteration Times"))
plt.tight_layout()
plt.title("{}".format("NBA alpha=2 plot"))
#plt.tick_params(axis='x', labelsize=30)
#plt.tick_params(axis='y', labelsize=30)
#fig.subplots_adjust(top=0.9, bottom=0.1,left=0.15,right=0.9)
plt.show()  # Show the plot
