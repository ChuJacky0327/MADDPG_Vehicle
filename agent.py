'''
此程式是定義agent 的類,一個代理人就有4個網路包刮(actor, 目標actor, critic, 目標critic)
所以每多一個代理人,就要呼叫一次這個class

一個agent有一個actor和一個critic，但是是單獨的記憶
每個agent只處理自己的神經網路
'''

import torch as T
from networks import CriticNetwork,ActorNetwork

class Agent():#定義Agent
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.7, tau=0.01):
        #設定agent_idx是為了要追蹤是哪個代理人
        #pig, 注意actor_dims和 critic_dims 要依照自己的觀察輸入維度而更改,ex:actor的observation為[0,1],那actor_dims為2
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx #定義現在創建的是第幾個代理人
        
        #創建ActorNetwork
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,chkpt_dir = chkpt_dir, name=self.agent_name+'_actor' )
        #參數包括(學習率,actor輸入維度,第一層全連接層維度,第二層全連接層維度,幾個動作,存檔案路徑,檔案名稱)
        #因為有多個agent 所以建立多個actor時名稱要不一樣
        
        #創建CriticNetwork
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir = chkpt_dir, name=self.agent_name+'_critic')
        #參數包括(學習率,critic輸入維度,第一層全連接層維度,第二層全連接層維度,幾個代理人,幾個動作,存檔案路徑,檔案名稱)
        #因為有多個agent 所以建立多個critic時名稱要不一樣
        
        #創建target_ActorNetwork
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,chkpt_dir = chkpt_dir, name=self.agent_name+'_target_actor' )
        #參數包括(學習率,target_actor輸入維度,第一層全連接層維度,第二層全連接層維度,幾個動作,存檔案路徑,檔案名稱)
        #因為有多個agent 所以建立多個target_actor時名稱要不一樣
        
        #創建target_CriticNetwork
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir = chkpt_dir, name=self.agent_name+'_target_critic')
        #參數包括(學習率,target_critic輸入維度,第一層全連接層維度,第二層全連接層維度,幾個代理人,幾個動作,存檔案路徑,檔案名稱)
        #因為有多個agent 所以建立多個target_critic時名稱要不一樣
        
        self.update_network_parameters(tau=1) #更新網路參數
    
    def update_network_parameters(self, tau=None): #這個def的目標是把在線網路的權重複製到目標網路
        #pig 這之副程式我看不太懂,後面要再回來搞懂,應該是跟target network有關,還有tau在幹嘛
        if tau == None:
            tau = self.tau
            
        target_actor_params = self.target_actor.named_parameters() #將目標actor的參數命名
        actor_params = self.actor.named_parameters() #將actor的參數命名
        
        target_actor_state_dict = dict(target_actor_params) #轉換成字典型態
        actor_state_dict = dict(actor_params) #轉換成字典型態
        #迭代actor和target_actor參數，利用tau進行乘法運算在相加
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau) * target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict) #最後把迭代完的參數更新上傳
        
        target_critic_params = self.target_critic.named_parameters() #將目標critic的參數命名
        critic_params = self.critic.named_parameters() #將critic的參數命名
        
        target_critic_state_dict = dict(target_critic_params) #轉換成字典型態
        critic_state_dict = dict(critic_params) #轉換成字典型態
        #迭代critic和target_critic參數，利用tau進行乘法運算在相加
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict) #最後把迭代完的參數更新上傳
        
    def choose_action(self, observation):#選擇動作,由環境的觀察狀態為輸入
        state = T.tensor([observation], dtype = T.float).to(self.actor.device)#轉換成tensor型態作為神經網路的輸入, .to()是發送到哪個設備
        #print(state)
        actions = self.actor.forward(state) #進行 actor 網路的前向傳播，輸入狀態得到動作,這邊得到的actions為所有動作的機率分布
        #actions輸出為n個維度的動作概率(所有概率和為1，n代表有幾個動作空間),這樣模型在每個狀態下都會計算所有可能動作的概率
        #print(actions)
        noise = T.rand(self.n_actions).to(self.actor.device) #加上一個雜訊,雜訊的張量維度要和actions的張量維度相同,pig但我不知道為啥要加一個雜訊
        #print(noise)
        action = actions + noise
        #print(action)
        #print(action.detach().cpu().numpy()[0])
        
        return action.detach().cpu().numpy()[0] #action為tensor型態，所以要轉回numpy做後續動作,但這菸的輸出依舊是所有動作的機率分布
        #pig 我看了一下環境的code,應該這邊輸出的動作機率分布,要在自己的車載環境裡面做判別取樣(action_sample),選出一個動作,之後再做動作對應(action_correspond)
        #pig 大不了在自己的車載環境裡面做判別取樣用單代理人keras的方法,取樣後對應在進行reward計算
        #pig 範例的環境裡一定要輸入至少5個機率分布，因為他在環境裡做取樣時有做判別
        #pig 加雜訊這部分要看一下到底需不需要
    
    #儲存模型
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    #加載模型
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
        
'''
A = Agent(2, 128 ,3, 3, 0, '')
x = []
x = A.choose_action([0,1])
print(x)
'''


        