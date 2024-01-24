with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[4].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[4].plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[4].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[4].plot(X_list, mean_arr, color='brown', markersize=1, label="IMFB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[4].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[4].plot(X_list, mean_arr, color='purple', markersize=1, label="egred")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[4].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[4].plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")


#axs[4].set_ylim(-1, 1)

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/egredpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/egredpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")

axs[4].set_xlabel('Iteration time')
axs[4].set_ylabel("Regret",loc='center')
axs[4].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[4].set_title('Maximin Regret plot')