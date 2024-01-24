import matplotlib.pyplot as plt
import numpy as np

import pickle
import numpy as np
import matplotlib.pyplot as plt

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

plt.plot(X_list, mean_arr, color='orange', markersize=1, label="LinUCB_fair")

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

plt.plot(X_list, mean_arr, color='green', markersize=1, label="CUCB_fair")

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

plt.plot(X_list, mean_arr, color='grey', markersize=1, label="egred_fair")

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

plt.plot(X_list, mean_arr, color='Turquoise', markersize=1, label="IMFB_fair")


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Combined Plot with Standard Variance')
plt.legend()
# Show the plot
plt.show()


