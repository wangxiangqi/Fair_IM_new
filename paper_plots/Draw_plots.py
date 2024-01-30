import matplotlib.pyplot as plt
import numpy as np
import pickle

# 创建子图
fig, axs = plt.subplots(1, 5, figsize=(60, 8))
#fig.set_size_inches(30, 3)
plt.tight_layout(pad=10)
# 画图
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/CUCBbailIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/CUCBbailIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/CUCBbailIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[0].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[0].plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")

with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/IMFBbailIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/IMFBbailIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/IMFBbailIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[0].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[0].plot(X_list, mean_arr, color='brown', markersize=1, label="IMFB")

with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/egredbailIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/egredbailIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/egredbailIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[0].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[0].plot(X_list, mean_arr, color='purple', markersize=1, label="egred")

with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/LinUCBbailIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/LinUCBbailIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/IAC/LinUCBbailIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[0].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[0].plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")


#axs[0].set_ylim(-1, 1)

with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/LinUCBbailfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/LinUCBbailfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[0].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-MIP")

with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/CUCBbailfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/CUCBbailfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[0].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-MIP")

with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/egredbailfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/egredbailfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[0].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-MIP")


with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/IMFBbailfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/maxmin_IAC/fair/IMFBbailfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[0].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[0].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-MIP")

axs[0].set_xlabel('Iteration time', fontsize=40)
axs[0].set_ylabel("Regret", loc='center', fontsize=40)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#axs[0].yaxis.major.formatter._useMathText = True
axs[0].yaxis.offsetText.set_fontsize(30)
#axs[0].yaxis.get_offset_text().set_fontsize(20)
axs[0].set_title('Maximin Regret', fontsize=40)
#axs[0].set_ylim(-5, 10)
#Set for bail
#Special for bail dataset
axs[0].legend(loc='lower center',bbox_to_anchor=(2.65, 1.035), ncol=8, fontsize=40)


with open('./presentation/bail/STD_VAR/div_IAC/IAC/CUCBbailIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/CUCBbailIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/CUCBbailIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[1].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[1].plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")

with open('./presentation/bail/STD_VAR/div_IAC/IAC/IMFBbailIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/IMFBbailIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/IMFBbailIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[1].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[1].plot(X_list, mean_arr, color='brown', markersize=1, label="IMFB")

with open('./presentation/bail/STD_VAR/div_IAC/IAC/egredbailIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/egredbailIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/egredbailIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[1].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[1].plot(X_list, mean_arr, color='purple', markersize=1, label="egred")

with open('./presentation/bail/STD_VAR/div_IAC/IAC/LinUCBbailIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/LinUCBbailIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/IAC/LinUCBbailIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(200))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[1].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)[:200]
axs[1].plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")




with open('./presentation/bail/STD_VAR/div_IAC/fair/LinUCBbailfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/fair/LinUCBbailfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[1].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-MIP")

with open('./presentation/bail/STD_VAR/div_IAC/fair/CUCBbailfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/fair/CUCBbailfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[1].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-MIP")

with open('./presentation/bail/STD_VAR/div_IAC/fair/egredbailfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/fair/egredbailfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[1].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-MIP")


with open('./presentation/bail/STD_VAR/div_IAC/fair/IMFBbailfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/div_IAC/fair/IMFBbailfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)[:200]
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)[:200]
axs[1].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[1].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-MIP")

axs[1].set_xlabel('Iteration time', fontsize=40)
axs[1].set_ylabel("Regret", loc='center', fontsize=40)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1].yaxis.get_offset_text().set_fontsize(30)
axs[1].set_title('Diversity Regret', fontsize=40)

#axs[1].set_ylim(-1000, 5000)


with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBbailIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBbailIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBbailIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[2].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[2].plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBbailIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBbailIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBbailIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[2].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[2].plot(X_list, mean_arr, color='brown', markersize=1, label="IMFB")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/egredbailIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/egredbailIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/egredbailIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[2].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[2].plot(X_list, mean_arr, color='purple', markersize=1, label="egred")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBbailIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBbailIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBbailIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[2].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[2].plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")


#axs[2].set_ylim(-1, 1)

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/LinUCBbailfair_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/LinUCBbailfair_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[2].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/CUCBbailfair_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/CUCBbailfair_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[2].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/egredbailfair_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/egredbailfair_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[2].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-MIP")


with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/IMFBbailfair_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_0.5/fair/IMFBbailfair_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[2].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[2].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-MIP")

axs[2].set_xlabel('Iteration time', fontsize=40)
axs[2].set_ylabel("Regret", loc='center', fontsize=40)
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[2].yaxis.get_offset_text().set_fontsize(30)
axs[2].set_title('alpha=0.5 Regret', fontsize=40)
#axs[2].set_ylim(-200, 2000)

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/CUCBbailIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/CUCBbailIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/CUCBbailIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[3].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[3].plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/IMFBbailIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/IMFBbailIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/IMFBbailIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[3].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[3].plot(X_list, mean_arr, color='brown', markersize=1, label="IMFB")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/egredbailIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/egredbailIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/egredbailIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[3].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[3].plot(X_list, mean_arr, color='purple', markersize=1, label="egred")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBbailIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBbailIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBbailIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

X_list=list(range(len(data1)))
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=np.std([data1, data2, data3], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#axs[3].gca().set_zorder(1)
mean_arr=np.mean([data1, data2, data3], axis=0)
axs[3].plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")


#axs[3].set_ylim(-1, 1)

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/LinUCBbailfair_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/LinUCBbailfair_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[3].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/CUCBbailfair_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/CUCBbailfair_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[3].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/egredbailfair_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/egredbailfair_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[3].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-MIP")


with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/IMFBbailfair_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_2/fair/IMFBbailfair_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[3].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[3].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-MIP")

axs[3].set_xlabel('Iteration time', fontsize=40)
axs[3].set_ylabel("Regret", loc='center', fontsize=40)
axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[3].yaxis.get_offset_text().set_fontsize(30)
axs[3].set_title('alpha=2 Regret', fontsize=40)
#axs[3].set_ylim(-200, 2500)

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBbailIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)
# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBbailIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBbailIAC_AC_p_n_3.pkl', 'rb') as f:
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

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBbailIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBbailIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBbailIAC_AC_p_n_3.pkl', 'rb') as f:
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

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredbailIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredbailIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredbailIAC_AC_p_n_3.pkl', 'rb') as f:
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

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBbailIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBbailIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBbailIAC_AC_p_n_3.pkl', 'rb') as f:
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

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBbailfair_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBbailfair_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBbailfair_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBbailfair_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-MIP")

with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/egredbailfair_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/egredbailfair_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='grey', markersize=1, label="egred-MIP")


with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBbailfair_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('./presentation/bail/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBbailfair_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
std_combined=np.std([data1, data2], axis=0)
axs[4].fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
axs[4].plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-MIP")

axs[4].set_xlabel('Iteration time', fontsize=40)
axs[4].set_ylabel("Regret", loc='center', fontsize=40)
axs[4].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[4].yaxis.get_offset_text().set_fontsize(30)
axs[4].set_title('alpha=-2 Regret', fontsize=40)
#axs[4].set_ylim(-2000, 10000)
# 显示图形
#plt.subplots_adjust(wspace=0.3)  # Adjust the width space between subplots
axs[0].tick_params(axis='both', labelsize=30)
axs[1].tick_params(axis='both', labelsize=30)
axs[2].tick_params(axis='both', labelsize=30)
axs[3].tick_params(axis='both', labelsize=30)
axs[4].tick_params(axis='both', labelsize=30)
plt.savefig('./reploted_bail_2.png', dpi=200)  # Adjust the dpi value as needed
