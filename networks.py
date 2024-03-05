'''
此程式是利用 pytorch 建置 cirtic 和 actor 網路(我的環境pytorch只裝cpu版，或許之後可以考慮改成keras)
critic 輸入為 state 和 action,輸出為 價值函數
actor 輸入為 state,輸出為 action
'''
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 有好多個代理人，一個代理人就有4個網路包刮(actor, 目標actor, critic, 目標critic)

# Critic 網路建置, Critic的輸入state+action,Critic的輸出為分數
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir) :#初始化建構子輸入包刮(critic的學習率,輸入的維度，第一個神經網路全連階層的維度,第一個神經網路全連階層的維度,代理人數量,動作數量,名字,存取模型的路徑)
        super(CriticNetwork,self).__init__() #調用CriticNetwork的init值
        
        self.chkpt_file = os.path.join(chkpt_dir, name) #保存模型檔案，放入檔案路徑和檔案名稱
        
        # pytorch裡的全連接層為Linear,keras裡為Dense
        # pytorch的全連接層要有輸入和輸出的維度。不像keras只要定義輸出的維度就好,輸入的維度會自己抓取
        self.fc1 = nn.Linear(input_dims + n_agents*n_actions , fc1_dims) # 建立第一層全連階層，輸入的維度為input_dims加上代理人的數量乘上動作的數量,輸出的維度為fc1_dims
        #critic的輸入是state+action。所以(input_dims為所有代理人的狀態空間(範例是8+10+10=28), n_agents*n_actions 為所有代理人的完整動作空間。兩者相加為神經網路的輸入維度)
         
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims) # 建立第二層全連階層，輸入的維度為fc1_dims,輸出的維度為fc2_dims
        self.q = nn.Linear(fc2_dims, 1) # critic網路的最後的輸出為一個分數，所以最後全連接層輸出為1
        
        #定義優化器
        self.optimizer = optim.Adam(self.parameters(), lr = beta) #使用Adam優化器，優化為我們的參數，學習率為beta
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') #看設備啟用是gpu還是cpu,我的環境pytorch只裝cpu版,所以輸出為cpu
        self.to(self.device) #將critic網路發送到設備執行,to應該是nn.Module裡定義的
        
    #神經網路裡的前向傳播
    def forward(self, state, action): # critic裡的輸入為 state 和 action
        #使用激活函數-relu,作為計算
        #神經網路輸入的資料格式要為 tensor
        #torch 裡的 cat 是concatnate的意思，cat是將兩个 tensor 拼接在一起
        x = F.relu(self.fc1(T.cat([state,action], dim = 1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q 
    
    #定義儲存 chickpoint 函數
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file) #儲存模型
    
    #定義讀取 chickpoint 函數
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file)) #加載模型


# Actor 網路建置, Actor 的輸出為動作機率
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,  name, chkpt_dir):#初始化建構子輸入包刮(actor的學習率,輸入的維度，第一個神經網路全連階層的維度,第一個神經網路全連階層的維度,代理人數量,動作數量,名字,存取模型的路徑)
        super(ActorNetwork, self).__init__() #調用 ActorNetwork 的init值
        
        self.chkpt_file = os.path.join(chkpt_dir, name) #保存模型檔案，放入檔案路徑和檔案名稱
        
        self.fc1 = nn.Linear(input_dims, fc1_dims)# 建立第一層全連階層，輸入的維度為input_dim,輸出的維度為fc1_dims
        #Actor 的輸入是state。所以(input_dims為所有代理人的狀態空間(範例是8+10+10=28))
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims) # 建立第二層全連階層，輸入的維度為fc1_dims,輸出的維度為fc2_dims
        self.pi = nn.Linear(fc2_dims, n_actions) # # Actor網路的最後的輸出為一組動作的機率，所以最後全連接層輸出為n_actions

        #定義優化器
        self.optimizer = optim.Adam(self.parameters(), lr = alpha) #使用Adam優化器，優化為我們的參數，學習率為alpha
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') #看設備啟用是gpu還是cpu,我的環境pytorch只裝cpu版,所以輸出為cpu
        self.to(self.device) #將Actor網路發送到設備執行,to應該是nn.Module裡定義的
    
    #神經網路裡的前向傳播
    def forward(self, state): # Actor裡的輸入為 state 
        #使用激活函數-relu,作為計算
        #神經網路輸入的資料格式要為 tensor
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim = 1)
        #因為最後輸出是機率分,所以用 softmax激活函數 
        
        return pi
    
    #定義儲存 chickpoint 函數
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file) #儲存模型
    
    #定義讀取 chickpoint 函數
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file)) #加載模型
    
    
    
    
    
    