import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all .npy files in the current directory
import pickle

#fig, (plt, plt) = plt.subplots(1, 2, figsize=(16, 12))
files = glob.glob('./MIP_oracle_comp/NBA-10/diversity/*.pkl')

# Create a new figure

fig, axs = plt.subplots(1, 2, figsize=(20, 6))
# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    axs[0].plot(data, label=file[35:-4])

axs[0].set_ylabel("{}".format("Regret"), fontsize=35)
#plt.tick_params(axis="both",label_size=14)
#plt.legend(prop={"size":12})  # Add legend
axs[0].set_xlabel("{}".format("Iteration Times"), fontsize=35)
#plt.tight_layout()
axs[0].set_title("{}".format("MIP Diversity Regret"), fontsize=30)
#plt.tick_params(axis='x', labelsize=14)
#plt.tick_params(axis='y', labelsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=30)

axs[0].legend(loc='lower center',bbox_to_anchor=(1, 1.04), ncol=4, fontsize=30)

files = glob.glob('./MIP_oracle_comp/NBA-10/maximin/*.pkl')

# Create a new figure


# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    axs[1].plot(data, label=file[26:-4])

axs[1].set_ylabel("{}".format("Regret"), fontsize=35)
#plt.tick_params(axis="both",label_size=14)
#plt.legend(prop={"size":12})  # Add legend
axs[1].set_xlabel("{}".format("Iteration Times"), fontsize=35)
#plt.tight_layout()
axs[1].set_title("{}".format("MIP Maximin Regret"), fontsize=30)


#plt.tick_params(axis='x', labelsize=14)
#plt.tick_params(axis='y', labelsize=14)
#fig.subplots_adjust(top=0.9, bottom=0.1,left=0.15,right=0.9)
# Add a legend and show the plot
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.legend(loc='center', bbox_to_anchor=(2, 1.3), ncol=4)
plt.tick_params(axis='both', which='major', labelsize=25)
#plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig('./MIP.png', bbox_inches='tight', dpi=200)