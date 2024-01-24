import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all .npy files in the current directory
import pickle

#fig, (plt, plt) = plt.subplots(1, 2, figsize=(16, 12))
files = glob.glob('./MIP_oracle_comp/diversity/*.pkl')

# Create a new figure

plt.subplot(4,5,2)
# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    plt.plot(data, label=file[28:-4])

plt.ylabel("{}".format("Regret"))
#plt.tick_params(axis="both",label_size=14)
#plt.legend(prop={"size":12})  # Add legend
plt.xlabel("{}".format("Iteration Times"))
#plt.tight_layout()
plt.title("{}".format("MIP Diversity Regret"))
#plt.tick_params(axis='x', labelsize=14)
#plt.tick_params(axis='y', labelsize=14)

files = glob.glob('./MIP_oracle_comp/maximin/*.pkl')

# Create a new figure
plt.subplot(4,5,3)

# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    plt.plot(data, label=file[26:-4])

plt.ylabel("{}".format("Regret"))
#plt.tick_params(axis="both",label_size=14)
#plt.legend(prop={"size":12})  # Add legend
plt.xlabel("{}".format("Iteration Times"))
#plt.tight_layout()
plt.title("{}".format("MIP Maximin Regret"))


#plt.tick_params(axis='x', labelsize=14)
#plt.tick_params(axis='y', labelsize=14)
#fig.subplots_adjust(top=0.9, bottom=0.1,left=0.15,right=0.9)
# Add a legend and show the plot
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend(loc='center', bbox_to_anchor=(-0.1, 1.3), ncol=4)
plt.show()