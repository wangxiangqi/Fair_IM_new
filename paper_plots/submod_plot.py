import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all .npy files in the current directory
import pickle

files = glob.glob('./submodular_oracle_comp/NBA-10/*.pkl')

# Create a new figure
#plt.subplot(4,5,3)

# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    plt.plot(data, label=file[32:-4])

plt.ylabel("{}".format("Regret"),fontsize=20)
#plt.tick_params(axis="both",labe,,l_size=14)
plt.legend(prop={"size":10},fontsize=20)  # Add legend
plt.xlabel("{}".format("Iteration Times"), fontsize=20)
#plt.tight_layout()
plt.title("{}".format("Submodular Oracle"),fontsize=20)


# Add a legend and show the plot
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tick_params(axis='both', which='major', labelsize=20)
#plt.legend(loc='center', bbox_to_anchor=(-0.1, 1.3), ncol=4)
plt.savefig('./sub_mod.png', bbox_inches='tight', dpi=200)