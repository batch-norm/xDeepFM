import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_csv('DNN_loss_result.csv')
# plt.title('DNN')
# plt.xlabel('step')
# plt.ylabel('loss')
#
# plt.plot(data.step,data.valid_auc,'b',label='valid_loss')
# plt.plot(data.step,data.train_auc,'g',label='train_loss')
# plt.legend(bbox_to_anchor=[0.3, 0.4])
# plt.yticks(np.linspace(0.45,0.7,20))
# plt.grid()
# plt.show()

data1 = pd.read_csv('DNN_loss_result.csv')
data2 = pd.read_csv('FM_loss_result.csv')
data3 = pd.read_csv('xDeepFM_loss_result.csv')
data4 = pd.read_csv('DeepFM_loss_result.csv')

plt.title('Model loss')
plt.xlabel('step')
plt.ylabel('loss')

plt.plot(data1.step,data1.valid_auc,'b',label='DNN loss')
plt.plot(data2.step,data2.valid_auc,'g',label='FM loss')
plt.plot(data3.step,data3.valid_auc,'r',label='xDeepFM loss')
plt.plot(data4.step[10:],data4.valid_auc[10:],'y',label='DeepFM loss')
plt.legend(bbox_to_anchor=[0.3, 0.4])
plt.yticks(np.linspace(0.45,0.7,20))
plt.grid()
plt.show()