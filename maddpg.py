'''
這隻程式是建置多代理人的配置，包括所有代理人的建置和所有代理人所選的動作機率分布
maddpg 內包含了一個所有的代理列表,並處理學習功能
'''

import torch as T
import torch.nn.functional as F
from agent import Agent

class MADDPG: #建置多代理人的class
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        #定義建構子
        
        self.agents = [] #代理人列表
        self.n_agents = n_agents #代理人數量
        self.n_actions = n_actions #動作數量
        chkpt_dir += scenario #針對不同環境可以存在不同檔內
        
        #創建所有代理人的代理列表
        for agent_idx in range(self.n_agents):
            #呼叫Agent的class,創建所有代理人並為每個代理人調用構建函數,並填到代理列表的矩陣內
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))
            # pig 注意這裡所輸入的actor_dims是矩陣型態了所以在call MADDPG的時候actor_dims要是狀態空間矩陣,且依舊要依照自己的觀察輸入維度而更改，在自己的環境裡要特別注意寫到，可以參考我下面的測試範例。
            
    #設置保存點
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
    
    #設置加載點
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    #選擇動作
    def choose_action(self, raw_obs):#輸入為觀察,raw_obs唯每一個代理人的觀察狀態所以是n維矩陣
        actions = [] #創建動作空列表
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx]) #每個代理人依據自己的觀察狀態選擇出一個動作的機率分布
            actions.append(action) #把所有代理人的動作空間機率分布疊合起來
        
        return actions #這裡回傳為所有代理人的所有動作空間機率分布，不是值
    
    #學習
    def learn(self, memory): #從緩衝區讀取數據和這次的選出來的動作獎勵進行比較
        #將內存數據傳地在這裡，內存為一個全局的重播緩存區
        #要讓緩衝區填滿才開始學習
        if not memory.ready(): #如果緩衝區為準備好就不學習，我的buffer裡的ready函數，當緩衝區的計數大於batch時會回傳True(代表緩衝區滿了)，所以這邊就會變成 if not True = if False 就不會工作
            return #因為要在填滿batch大小時才開始學習
        
        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer() #從緩衝區讀取出資料
        # actor_states 是單個代理人的狀態numpy數組, states為所有代理人狀態flatten, actor_new_states 是單個代理人的下一個狀態numpy數組, states_為所有代理人下一個狀態flatten
        
        device = self.agents[0].actor.device #調用一個設備，我的是cpu
        states = T.tensor(states, dtype=T.float).to(device) #內存讀出來的狀態轉為tensor型態
        actions = T.tensor(actions, dtype=T.float).to(device) #內存讀出來的動作轉為tensor型態
        rewards = T.tensor(rewards).to(device) #內存讀出來的獎勵轉為tensor型態
        states_ = T.tensor(states_, dtype=T.float).to(device) #內存讀出來的下一個狀態轉為tensor型態
        dones = T.tensor(dones).to(device) #內存讀出來的終端狀態轉為tensor型態
        
        all_agents_new_actions = [] #所有代理人所採取的動作
        all_agents_new_mu_actions = [] #所有代理人所採取的mu動作
        old_agents_actions = [] #所有代理人所採取舊的動作
        
        #根據新狀態目標actor動作、當前狀態 actor動作、代理實際採取的動作，這三項來計算損失函數
        for agent_idx, agent in enumerate(self.agents): #所有代理人列舉
            #從內存緩衝區讀取狀態後，target actor的動作更新
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)# actor_new_states 是單個代理人的下一個狀態numpy數組,把它全部轉換成tensor型態
            new_pi = agent.target_actor.forward(new_states) #從內存緩衝區讀取下一個狀態後，新的動作為目標actor的輸出，輸入為每個代理人的下一個狀態tensor
            all_agents_new_actions.append(new_pi) #把從內存緩衝區讀取狀態後，目標actor選擇的新動作存在all_agents_new_actions
            
            #從內存緩衝區讀取狀態後，actor的動作更新
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device) #actor_states 是單個代理人的狀態numpy數組,把它全部轉換成tensor型態
            pi = agent.actor.forward(mu_states) #從內存緩衝區讀取狀態後，新的mu動作為actor的輸出，輸入為每個代理人的狀態tensor
            all_agents_new_mu_actions.append(pi) #把從內存緩衝區讀取狀態後，actor選擇的新動作存在all_agents_new_actions
            
            #從內存緩衝區讀取的舊動作
            old_agents_actions.append(actions[agent_idx])
            
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1) #利用cat將多个 tensor 拼接在一起
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)#利用cat將多个 tensor 拼接在一起
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)#利用cat將多个 tensor 拼接在一起
        
        
        # 處理完actor動作後，要來處理價值函數
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten() #下一個狀態的目標價值函數為目標critic把從內存緩衝區讀取下一個狀態，與目標actor選擇的新動作作為輸入得出的結果
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()#狀態的價值函數為critic，從內存緩衝區讀取狀態，與actor選擇的新動作作為輸入得出的結果
            
            #pig 這邊就要回去看論文了MARL的loss更新公式，target，這邊要補看並補在word的公式裡
            #target:
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            
            #critic loss:
            critic_loss = F.mse_loss(target,critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph = True)
            agent.critic.optimizer.step()
            
            #actor loss:
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph = True)
            agent.actor.optimizer.step()
            
            agent.update_network_parameters()
            
            
            
            
        
'''
A = MADDPG([4,4,4], 128 ,3, 3)
x = []
obs = [[2.002, 83.0, 90.527, 1.0], [1.751, 83.0, 96.508, 1.0], [3.816, 83.0, 97.551, 1.0]]
action = A.choose_action(obs)
print(action)
'''
        
        
