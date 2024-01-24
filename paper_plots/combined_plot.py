import matplotlib.pyplot as plt
import numpy as np

import pickle
import numpy as np
import matplotlib.pyplot as plt

#fig, (ax1, ax2) = plt.subplots(, 1, sharex=True)
#plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
# Read data from the first .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/CUCBpokecIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/CUCBpokecIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/CUCBpokecIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation
#std_combined=[]
#for i in range(len(X_list)):
#    std=np.std([data1[i],data2[i],data3[i]])
#    std_combined.append(std)

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
#print(len(std_combined))
# Plot the combined data with error bars
#plt.figure(1)
#plt.errorbar(X_combined,y_combined, yerr=std_combined, color='lightblue', ecolor='lightblue', capsize=0)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(1)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
#plt.figure(figsize=(10,6))
plt.subplot(4,5,1)
#plt.set_aspect('equal')
#plt.figure(figsize=(10, 5))
plt.plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")
#plt.plot(mean_arr, color='black', linestyle='-', marker='o', markersize=5, linewidth=2)

#plt.legend()

# Show the plot
# Set labels and title

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/IMFBpokecIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/IMFBpokecIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/IMFBpokecIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(2)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='brown', markersize=1,label="IMFB")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/egredpokecIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/egredpokecIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/egredpokecIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(3)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
plt.plot(X_list, mean_arr, color='purple', markersize=1, label="egred")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/LinUCBpokecIAC_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/LinUCBpokecIAC_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/IAC/LinUCBpokecIAC_AC_m_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/LinUCBpokecfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/LinUCBpokecfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/CUCBpokecfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/CUCBpokecfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/egredpokecfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/egredpokecfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/IMFBpokecfair_AC_m.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/maxmin_IAC/fair/IMFBpokecfair_AC_m_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")


plt.xlabel('Iteration time')
plt.ylabel("Regret",loc='center')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('Maximin Regret plot')
plt.ylim(-3, 12)
#plt.legend()
# Show the plot
#plt.show()

"""
fgdhkjshkdhsdhksj
dhsajdjashgfhjgdajhsdkljhkj
gfdshjagdhjsghjdagj
gdshajgdhsjga]
gdshajgdhjas'
"""

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
#x1 = np.array([1, 2, 3, 4, 5])
#y1 = np.array([2, 4, 6, 8, 10])
#std1 = np.array([0.5, 0.8, 1.2, 1.5, 1.7])

#x2 = np.array([1, 2, 3, 4, 5])
#y2 = np.array([1, 3, 5, 7, 9])
#std2 = np.array([0.3, 0.6, 0.9, 1.2, 1.4])

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data from the first .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/CUCBpokecIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)
    data1=data1[:200]

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/CUCBpokecIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
    data2=data2[:200]

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/CUCBpokecIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
    data3=data3[:200]

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation
#std_combined=[]
#for i in range(len(X_list)):
#    std=np.std([data1[i],data2[i],data3[i]])
#    std_combined.append(std)

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
#print(len(std_combined))
# Plot the combined data with error bars
#plt.figure(1)
#plt.errorbar(X_combined,y_combined, yerr=std_combined, color='lightblue', ecolor='lightblue', capsize=0)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(1)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
plt.subplot(4,5,2)
plt.plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")
#plt.plot(mean_arr, color='black', linestyle='-', marker='o', markersize=5, linewidth=2)

#plt.legend()

# Show the plot
# Set labels and title

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/IMFBpokecIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)
    data1=data1[:200]

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/IMFBpokecIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
    data2=data2[:200]

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/IMFBpokecIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
    data3=data3[:200]

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(2)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='brown', markersize=1,label="IMFB")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/egredpokecIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)
    data1=data1[:200]

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/egredpokecIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
    data2=data2[:200]

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/egredpokecIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)
    data3=data3[:200]

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(3)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='purple', markersize=1, label="egred")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/LinUCBpokecIAC_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/LinUCBpokecIAC_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/IAC/LinUCBpokecIAC_AC_d_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/LinUCBpokecfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/LinUCBpokecfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/CUCBpokecfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/CUCBpokecfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/egredpokecfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/egredpokecfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/IMFBpokecfair_AC_d.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/div_IAC/fair/IMFBpokecfair_AC_d_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")

#s_adjust(hspace=0)
plt.xlabel('Iteration Time')
plt.ylabel("Regret",loc='center')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('Diversity Regret')
#plt.legend()
#plt.ylim(-2000, 8000)
plt.ylim(-20000,80000)
#plt.title.set_position([.5, 1.05])
# Show the plot
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
#x1 = np.array([1, 2, 3, 4, 5])
#y1 = np.array([2, 4, 6, 8, 10])
#std1 = np.array([0.5, 0.8, 1.2, 1.5, 1.7])

#x2 = np.array([1, 2, 3, 4, 5])
#y2 = np.array([1, 3, 5, 7, 9])
#std2 = np.array([0.3, 0.6, 0.9, 1.2, 1.4])

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data from the first .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/CUCBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation
#std_combined=[]
#for i in range(len(X_list)):
#    std=np.std([data1[i],data2[i],data3[i]])
#    std_combined.append(std)

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
#print(len(std_combined))
# Plot the combined data with error bars
#plt.figure(1)
#plt.errorbar(X_combined,y_combined, yerr=std_combined, color='lightblue', ecolor='lightblue', capsize=0)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(1)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
plt.subplot(4,5,3)
plt.plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")
#plt.plot(mean_arr, color='black', linestyle='-', marker='o', markersize=5, linewidth=2)

#plt.legend()

# Show the plot
# Set labels and title

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/IMFBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(2)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='brown', markersize=1,label="IMFB")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/egredpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(3)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='purple', markersize=1, label="egred")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/IAC/LinUCBpokecIAC_AC_p_n_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/LinUCBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/CUCBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/egredpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/egredpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBpokecfair_AC_wel_p_n.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_neg_2/fair/IMFBpokecfair_AC_wel_p_n_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")


plt.xlabel('Iteration time')
#('Regret')
plt.ylabel('Regret')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('alpha=-2 regret')
plt.legend(loc='lower center', bbox_to_anchor=(1.3, 1.2), ncol=8)
# Show the plot
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
#x1 = np.array([1, 2, 3, 4, 5])
#y1 = np.array([2, 4, 6, 8, 10])
#std1 = np.array([0.5, 0.8, 1.2, 1.5, 1.7])

#x2 = np.array([1, 2, 3, 4, 5])
#y2 = np.array([1, 3, 5, 7, 9])
#std2 = np.array([0.3, 0.6, 0.9, 1.2, 1.4])

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data from the first .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/CUCBpokecIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/CUCBpokecIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/CUCBpokecIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation
#std_combined=[]
#for i in range(len(X_list)):
#    std=np.std([data1[i],data2[i],data3[i]])
#    std_combined.append(std)

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
#print(len(std_combined))
# Plot the combined data with error bars
#plt.figure(1)
#plt.errorbar(X_combined,y_combined, yerr=std_combined, color='lightblue', ecolor='lightblue', capsize=0)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(1)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
plt.subplot(4,5,4)
plt.plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")
#plt.plot(mean_arr, color='black', linestyle='-', marker='o', markersize=5, linewidth=2)

#plt.legend()

# Show the plot
# Set labels and title

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/IMFBpokecIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/IMFBpokecIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/IMFBpokecIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(2)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='brown', markersize=1,label="IMFB")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/egredpokecIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/egredpokecIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/egredpokecIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(3)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='purple', markersize=1, label="egred")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBpokecIAC_AC_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBpokecIAC_AC_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/IAC/LinUCBpokecIAC_AC_p_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/LinUCBpokecfair_AC_wel_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/LinUCBpokecfair_AC_wel_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/CUCBpokecfair_AC_wel_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/CUCBpokecfair_AC_wel_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/egredpokecfair_AC_wel_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/egredpokecfair_AC_wel_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/IMFBpokecfair_AC_wel_p.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_2/fair/IMFBpokecfair_AC_wel_p_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")


plt.xlabel('Iteration time')
plt.ylabel("Regret",loc='center')
#('regret')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('alpha=2 regret')
#plt.ylim(-300,1000)
plt.ylim(-5000,20000)
#plt.legend()
# Show the plot
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
#x1 = np.array([1, 2, 3, 4, 5])
#y1 = np.array([2, 4, 6, 8, 10])
#std1 = np.array([0.5, 0.8, 1.2, 1.5, 1.7])

#x2 = np.array([1, 2, 3, 4, 5])
#y2 = np.array([1, 3, 5, 7, 9])
#std2 = np.array([0.3, 0.6, 0.9, 1.2, 1.4])

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data from the first .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBpokecIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBpokecIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/CUCBpokecIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation
#std_combined=[]
#for i in range(len(X_list)):
#    std=np.std([data1[i],data2[i],data3[i]])
#    std_combined.append(std)

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
#print(len(std_combined))
# Plot the combined data with error bars
#plt.figure(1)
#plt.errorbar(X_combined,y_combined, yerr=std_combined, color='lightblue', ecolor='lightblue', capsize=0)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(1)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)
plt.subplot(4,5,5)
plt.plot(X_list, mean_arr, color='red', markersize=1, label="CUCB")
#plt.plot(mean_arr, color='black', linestyle='-', marker='o', markersize=5, linewidth=2)

#plt.legend()

# Show the plot
# Set labels and title

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBpokecIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBpokecIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/IMFBpokecIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(2)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='brown', markersize=1,label="IMFB")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/egredpokecIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/egredpokecIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/egredpokecIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()
plt.gca().set_zorder(3)

#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='purple', markersize=1, label="egred")


with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBpokecIAC_AC_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBpokecIAC_AC_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/IAC/LinUCBpokecIAC_AC_p_h_3.pkl', 'rb') as f:
    data3 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2, data3], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
var3=[np.std(data3)]*len(data3)
std_combined=[var1]+[var2]+[var3]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2, data3], axis=0)

plt.plot(X_list, mean_arr, color='black', markersize=1, label="LinUCB")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/LinUCBpokecfair_AC_wel_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/LinUCBpokecfair_AC_wel_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/CUCBpokecfair_AC_wel_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/CUCBpokecfair_AC_wel_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/egredpokecfair_AC_wel_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/egredpokecfair_AC_wel_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)
#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred-FW")

with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/IMFBpokecfair_AC_wel_p_h.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Read data from the second .pkl file
with open('E:/summer_intern/Hua_zheng_Wang/IMFB-KDD2019-master/presentation/pokec/STD_VAR/wel_IAC/alpha_0.5/fair/IMFBpokecfair_AC_wel_p_h_2.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Read data from the third .pkl file

#print(data1)
X_list=list(range(len(data1)))
# Calculate the standard deviation

X_combined = X_list
mean_arr=np.mean([data1, data2], axis=0)
y_combined = mean_arr
#std_combined = np.concatenate((std1, std2,std3))
#std_combined=np.asarray(std_combined).T
std_combined=[]
var1=[np.std(data1)]*len(data1)
var2=[np.std(data2)]*len(data2)
std_combined=[var1]+[var2]
std_combined=np.asarray(std_combined).flatten()
std_combined=np.mean(std_combined)
print(len(X_combined))
print(len(y_combined))
# Plot the combined data with error bars
#plt.figure(1)
plt.fill_between(X_combined, y_combined - std_combined, y_combined + std_combined, alpha=0.1)
#plt.legend()

plt.gca().set_zorder(4)
#plt.figure(2)
mean_arr=np.mean([data1, data2], axis=0)

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB-FW")


plt.xlabel('Iteration time')
plt.ylabel("Regret",loc='center')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('alpha=0.5 regret')
#plt.legend()
# Show the plot
#plt.tight_layout()
plt.show()








