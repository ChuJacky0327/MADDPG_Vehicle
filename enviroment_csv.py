import csv
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow_probability as tfp

class single_env():
    def __init__(self):
        self.n_agents = 4 #3車+1MEC
        self.action_space = 5 #總共有幾個動作
        self.beta1 = 1
        self.beta2 = -0.1

        
        self.veh_DataRate = None
        self.veh_direction = None
        self.veh_packet_size = None
        self.beta = None
        
        self.max_delay = 0.58 #封包帶20,data_rate帶35
        self.min_delay = 0.14 #封包帶15,data_rate帶100
        
        
        self.max_data_rate = 100
        self.price = 0
        
        self.c = 1 #money_token
        self.rd = 0.1655 #每1Mbit的功耗
        self.rt = 0.7438 #每個時間段的功耗
        self.max_power_consumption = 18
        self.min_power_consumption = 6.5
        
    def observation_space_shape_calculate(self): #計算observation的矩陣各子矩陣輸出維度,要讓它變成actor_dims
        #我的所有代理人的observation維度都一樣，所以直接讀第一行csv算維度就好
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        
        float_row = list(map(self.convert, rows[1])) #第0列是title,用map直接全部轉換
        #print(float_row)
        agent_1_obs = []
        agent_2_obs = []
        agent_3_obs = []
        agent_4_obs = []
        
        agent_1_obs.append(float_row[0])
        agent_1_obs.append(float_row[1])
        agent_1_obs.append(float_row[8])

        agent_2_obs.append(float_row[2])
        agent_2_obs.append(float_row[3])
        agent_2_obs.append(float_row[8])
        
        agent_3_obs.append(float_row[4])
        agent_3_obs.append(float_row[5])
        agent_3_obs.append(float_row[8])

        agent_4_obs.append(float_row[6])
        agent_4_obs.append(float_row[7])
        agent_4_obs.append(float_row[8])
        
        observation_space_shape = []
        #print(len(agent_1_obs))
        observation_space_shape.append(len(agent_1_obs))
        observation_space_shape.append(len(agent_2_obs))
        observation_space_shape.append(len(agent_3_obs))
        observation_space_shape.append(len(agent_4_obs))
        # print(self.observation_space_shape[0])
        # print(self.observation_space_shape[1])
        # print(self.observation_space_shape[2])
        return observation_space_shape
    
    def minmax_norm(self,data, max_data, min_data):
        norm_data = (data-min_data) / (max_data-min_data)
        return norm_data
    
    def action_correspond(self,action_sample,obs_veh): #動作對應,選擇出來的動作對應bitrate,先假設只有5個調控
        obs_DataRate = obs_veh[0]
        #print('Throughput:',Throughput)
        if (action_sample == 0):
            action = 35
            if action > obs_DataRate:
                action = int(obs_DataRate)
                
        elif (action_sample == 1):
            action = 50
            if action > obs_DataRate:
                action = int(obs_DataRate)
                
        elif (action_sample == 2):
            action = 65
            if action > obs_DataRate:
                action = int(obs_DataRate)
                
        elif (action_sample == 3):
            action = 85
            if action > obs_DataRate:
                action = int(obs_DataRate)
                
        elif (action_sample == 4):
            action = 100
            if action > obs_DataRate:
                action = int(obs_DataRate)
        #print(bitrate)
        return action
    
    def action_sample(self, action_prob): #因為多代理人的環境所輸出的動作是機率分布並沒有做取樣，所以設一個副函式用單代理人keras的方法進行取樣
        action_probabilities = tfp.distributions.Categorical(probs = action_prob)
        action = action_probabilities.sample() #取樣,挑選了第'幾號'動作(經過sample後，這邊不是機率而是第'幾號'動作)
        #print(action.numpy())
        return action.numpy()
    
    
    def convert(self,string): #文字轉換成float
        try:
            string=float(string)
        except :
            pass    
        return string
    
    def Game_price_set(self, data_rate): #賽局的價格制定,依照bitrate給定
        if (35 <= data_rate < 50 ) :
            price = 1
        elif (50 <= data_rate <65 ) :
            price = 2
        elif (65 <= data_rate < 85) :
            price = 3
        elif (85 <= data_rate < 100) :
            price = 4
        elif (100 <= data_rate ) :
            price = 5
        else:
            price = 0
            print("price error")
        return price
    
    def data_rate_norm(self, data_rate):
        norm_data_rate = data_rate / self.max_data_rate
        return norm_data_rate
    
    def reset(self):#讀取csv第一行，讀出的值是每個代理人自己的觀察，所以是n維矩陣
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        
        float_row = list(map(self.convert, rows[1])) #第0列是title,用map直接全部轉換
        #print(float_row)
        agent_1_obs = []
        agent_2_obs = []
        agent_3_obs = []
        agent_4_obs = []
        
        agent_1_obs.append(float_row[0])
        agent_1_obs.append(float_row[1])
        agent_1_obs.append(float_row[8])

        agent_2_obs.append(float_row[2])
        agent_2_obs.append(float_row[3])
        agent_2_obs.append(float_row[8])
        
        agent_3_obs.append(float_row[4])
        agent_3_obs.append(float_row[5])
        agent_3_obs.append(float_row[8])

        agent_4_obs.append(float_row[6])
        agent_4_obs.append(float_row[7])
        agent_4_obs.append(float_row[8])
        
        observation = [] #這裡的和單代理人不一樣，這裡的觀察是所有代理人自身觀察的子集合，所以是n維
        observation.append(agent_1_obs)
        observation.append(agent_2_obs)
        observation.append(agent_3_obs)
        observation.append(agent_4_obs)
        #print(type(self.observation))

        return observation
    
    def step(self, obs, actions, nextstep_count):
        #寫done:
        terminate = [False] * self.n_agents #設定done
        if nextstep_count == 100: #因為我csv的資料只有100筆,所以當跑到第100筆時就讓done維true結束迴圈
            terminate = [True] * self.n_agents #所以csv裡最後一筆資料訓練不到
        #print('terminate:', terminate)
        
        #寫observation_ 下一個狀態:
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        float_row = list(map(self.convert, rows[nextstep_count])) #去讀第nextstep_count行
        agent_1_obs_ = []
        agent_2_obs_ = []
        agent_3_obs_ = []
        agent_4_obs_ = []
        observation_ = []
        
        agent_1_obs_.append(float_row[0])
        agent_1_obs_.append(float_row[1])
        agent_1_obs_.append(float_row[8])
        
        agent_2_obs_.append(float_row[2])
        agent_2_obs_.append(float_row[3])
        agent_2_obs_.append(float_row[8])
        
        agent_3_obs_.append(float_row[4])
        agent_3_obs_.append(float_row[5])
        agent_3_obs_.append(float_row[8])
        
        agent_4_obs_.append(float_row[6])
        agent_4_obs_.append(float_row[7])
        agent_4_obs_.append(float_row[8])
        
        observation_.append(agent_1_obs_)
        observation_.append(agent_2_obs_)
        observation_.append(agent_3_obs_)
        observation_.append(agent_4_obs_)
        #print('observation_:', observation_)
        
        #寫reward:
        rewards = []
        for i in range (self.n_agents):
            action_sample = self.action_sample(actions[i]) #動作機率分布取樣
            #print(action_sample)
            #print('action_sample:',action_sample)
            

            self.veh_DataRate = obs[i][0]
            self.veh_direction = obs[i][1]
            self.veh_packet_size = obs[i][2]
            
            self.beta = self.beta1 #因為這邊是單跳點所以這邊self.beta = 1
            
            #計算車載傳輸延遲分數:
            data_rate = self.action_correspond(action_sample ,obs[i]) #動作對應bitrate
            Transmission_delay = self.veh_packet_size / data_rate #計算車載傳輸延遲      
            norm_Transmission_delay = self.minmax_norm(Transmission_delay, self.max_delay, self.min_delay) #做min_max正歸化
            
            #計算data_rate分數:
            self.price = self.Game_price_set(data_rate)
            #print('price:',self.price)
            norm_data_rate = self.data_rate_norm(data_rate)
            data_rate_score = self.price * norm_data_rate
            #print('bitrate_score:',bitrate_score)
            
            #計算功耗分數:,pig注意 考慮一下這段要不要做正規化,做正規化的話bitrate怎麼選利潤都會大於成本,不做正規化的話bitrate要選3以上利潤才會大於成本
            power_consumption = self.rd * data_rate + self.rt
            #print('power_consumption:',power_consumption)
            norm_power_consumption = self.minmax_norm(power_consumption, self.max_power_consumption, self.min_power_consumption)
            #print('norm_power_consumption:',norm_power_consumption)
            
            #計算reward :
            reward = self.beta*(-norm_Transmission_delay + data_rate_score - (self.c * norm_power_consumption))
            rewards.append(reward)
            #print(reward)
            #print('--------------')
        
        #print(rewards)
        #print('----------')
        return observation_, rewards, terminate, {}
            
            
        
        
        
        
'''
A = single_env()
B = A.observation_space_shape_calculate()
print(B)

obs = A.reset()
#X = []
#Y = []
nextstep_count = 1
for i in range(2):
    nextstep_count = nextstep_count +1
    actions =  [[1.082,0.145,0.706,0.2,0.6 ],[0.576, 1.1138 ,0.6055 ,0.8,0.9],[1.35, 0.2530, 0.136,0.2,0.6]]
    obs_, rewards, done, info = A.step(obs,actions, nextstep_count)
    print('obs:',obs)
    print('obs_:',obs_)
    print('rewards:',rewards)
    print('done:',done)
    obs = obs_
'''    
  
    