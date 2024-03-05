'''
此程式是建立一個 repaly_buffer，把所有代理人的數據存到repaly_buffer，並從 buffer中選擇其中一筆資料回傳
agent數量 = actor數量 (訓練幾個代理人就代表訓練幾個演員)
actor 輸出為 action
critic 輸出為 價值函數
'''

import numpy as np

class MultiAgentReplayBuffer():
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size): #輸入(最大的容量、critic維度、actor維度,動作,所有代理人,batch)
        self.mem_size = max_size #內存大小
        self.mem_cntr = 0 #內存計數器
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        self.n_agents = n_agents
        self.batch_size = batch_size
        
        #定義內存
        self.state_memory = np.zeros((self.mem_size, critic_dims)) #狀態內存用critic維度建立
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype = bool) #終端狀態，如果是1代表為終端狀態，終端狀態不會有價值分數因為沒有下一步了
        
        self.init_actor_memory() #actor 內存
    
    def init_actor_memory(self): #actor 內存
        #此函數作用，當actor和critic內存達到上限時，要把actor的內存歸零
        self.actor_state_memory = [] #actor的狀態記憶歸零(空矩陣)
        self.actor_new_state_memory = [] #actor的新狀態也為空矩陣
        self.actor_action_memory = [] #演員動作記憶
        
        for i in range(self.n_agents): #範圍是代理人的數量 
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i]))) #actor的狀態內存添加第i個演員的內存狀態
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i]))) #actor的新狀態也添加
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions))) #動作內存，因為actor輸出是action,把每個演員輸出的動作記錄下來
    
    #這個def要做的事是把讀進buffer的資料儲存下來
    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done): #raw_obs是原始的觀察數據，要餵入critic網路。state是所有觀察的flatten組合
        # 將資料存進replay_buffer   
        if (self.mem_cntr % self.mem_size  == 0) and self.mem_cntr > 0 : #如果內存計算器滿了
           self.init_actor_memory() #重啟內存
        
        index = self.mem_cntr % self.mem_size #計算第一個可用的內存位置
        
        for agent_idx in range(self.n_agents): #跑過所有代理人，並將所觀察到的數據、和下一個狀態、和動作都存到內存去
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx] #actor state內存更新，把第agent_idx的原始觀察數據存進去內存
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx] #actor 下一個狀態的內存更新，一樣把第agent_idx的原始觀察數據存進去
            self.actor_action_memory[agent_idx][index] = action[agent_idx] #actor的動作內存，把所有代理人的動作存起來
            
        #critic的內存處理
        self.state_memory[index] = state #狀態的內存為現在的狀態
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr = self.mem_cntr + 1 #計數+1
    
    #取樣緩衝區
    #這個def要做的事是選定緩衝區的其中一筆資料回傳到主程式
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size) #要得知現在內存的最上面那筆是第幾個位置
        
        #從緩衝區內隨機選擇一個作為輸出，包刮狀態、獎勵、新狀態、終端狀態
        batch = np.random.choice(max_mem, self.batch_size, replace=False) #replace=False 確保不會兩次獲得相同內存
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]
        
        actor_states = []
        actor_new_states = []
        actions = []
        
        for agent_idx in range(self.n_agents): #上述是在選定第幾個batch的資料(選緩衝區的哪個內存資料)，下面是在把所有代理人選定的batch資料存下來
            actor_states.append(self.actor_state_memory[agent_idx][batch]) #把所有代理人的所選定batch的狀態內存，存下來
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])#把所有代理人的所選定batch的新狀態內存，存下來
            actions.append(self.actor_action_memory[agent_idx][batch]) #把所有代理人的所選定batch的動作內存，存下來
            
        #最後把緩衝區選擇出來的內存資料傳回主程式
        return actor_states, states, actions, rewards, actor_new_states, states_, terminals
        # actor_states 是單個代理人的狀態numpy數組, states為所有代理人狀態flatten, actor_new_states 是單個代理人的下一個狀態numpy數組, states_為所有代理人下一個狀態flatten
    
    def ready(self): #去看緩衝區的大小有沒有超過 batch size
        if (self.mem_cntr > self.batch_size):
            return True
        return False
        
            