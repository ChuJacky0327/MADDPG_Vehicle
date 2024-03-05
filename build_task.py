'''
設定任務量(task),也就是csv裡有幾個狀態
'''

#這邊先預設我只跳點跳一層
import numpy as np
import csv

K1_DataRate = []
K2_DataRate = []
K3_DataRate = []
MEC_DataRate = []
packet_size = []
K1_direction = []
K2_direction = []
K3_direction = []
MEC_direction = []
state = []
task = 100

for i in range(task): #Mbps
    K1_DataRate.append(round(np.random.uniform(35,120),3))#action = 35 50 65 85 100
    K2_DataRate.append(round(np.random.uniform(35,120),3))
    K3_DataRate.append(round(np.random.uniform(35,120),3))
    MEC_DataRate.append(round(np.random.uniform(35,120),3))
    packet_size.append(np.random.randint(15,21)) #取15~20
    K1_direction.append(round(np.random.uniform(50,100),3))
    K2_direction.append(round(np.random.uniform(50,100),3))
    K3_direction.append(round(np.random.uniform(50,100),3))
    MEC_direction.append(100)

        
state.append(K1_DataRate)
state.append(K1_direction)
state.append(K2_DataRate)
state.append(K2_direction)
state.append(K3_DataRate)
state.append(K3_direction)
state.append(MEC_DataRate)
state.append(MEC_direction)
state.append(packet_size)
state_T = np.transpose(state)
#print(np.shape(state_T))

with open('simulation1_singlehop.csv','w',newline='') as file: #'w'為覆寫
    write = csv.writer(file)
    write.writerow(["K1_DataRate","K1_direction","K2_DataRate","K2_direction","K3_DataRate","K3_direction","MEC_DataRate","MEC_direction","packet_size"])
    for i in range(task):    
        write.writerow(state_T[i])
    print('build stete file')
