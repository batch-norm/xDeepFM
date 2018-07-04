import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('2RI_process_1547s.csv')
data1 = pd.read_csv('deepFM_2078s.csv')
data2 = pd.read_csv('mini512_process_1386.csv')
plt.title('valid F1 score')
plt.xlabel('iteration')
plt.ylabel('score')

# plt.plot(data.batch_num[100:],data.valid_auc[100:],'r',label='RI valid auc')
plt.plot(data.batch_num[100:],data.valid_f1[100:],'b',label='GD')
plt.plot(data1.batch_num[100:],data1.valid_f1[100:],'g',label='RI-GD')
plt.plot(data2.batch_num[100:],data2.valid_f1[100:],'r',label='mini')
# plt.plot(data1.batch_num[100:],data1.valid_auc[100:],'y',label='valid auc')
plt.yticks(np.linspace(0.755,0.815,20))
plt.legend(bbox_to_anchor=[0.3, 0.4])
plt.grid()
plt.show()