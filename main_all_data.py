'''
**輸出的reward要是多維矩陣ex:[-0.406, -0.1706, -0.1706]，代表每個代理人根據自己選的動作所得到的分數，而且會寫到buffer裡所以不能更改格式
範例會把reward加起來是因為它的環境有對抗+合作所以算整體分數,但我的環境屬於競爭，所以計算score方式要改(取max)
但是score的計算，要去挑選一台車能給我最大的獎勵我就往她傳，所以score應該是選一台最好的車子的獎勵(如果用reward輸出，這樣可能每個迴圈都是挑不同車子)
***所以假如要做的是經過訓練後就是要挑選一台車做傳輸(挑一個總reward最高得車輛做score,不是每個epoch挑一台最高得車輛做score的話)可考慮多一個輸出矩陣把這個epoch所有的蟄存起來再做np.max(np.sum(np.array(rewards), axis = 0)),下一個epoch在清空
'''
from make_env import make_env
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import numpy as np
from utils_all_data import plot_learning_curve #畫圖,用matplotlib畫標記軸
from enviroment_csv_all_data import single_env

#這隻程式目的是把觀察矩陣轉為一維向量，要寫到buffer裡的(所有代理人觀察flatten)
def obs_list_to_state_vector(observation):#ex: observation = [[0, 1], [2, 1], [3, 2]]  => state = [0. 1. 2. 1. 3. 2.]
    state = np.array([]) #先設定一個空矩陣
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
    
if __name__ == '__main__':
    scenario = 'vehicle_SingeleHop'
    #env = make_env(scenario)
    #pig 這裡的env要換成自己得車載環境
    env = single_env()
    
    #n_agents = env.n #代理人的數量
    #pig 這裡的env.n要換成自己車載環境裡代理人的數量，在init裡設定就好
    n_agents = env.n_agents
    
    #actor_dims輸入的維度矩陣
    #actor_dims = [] #actor的維度
    #for i in range(n_agents):
        #actor_dims.append(env.observation_space[i].shape[0]) 
        #範例裡env.observation_space[0].shape[0] 的輸出為 8。
        #範例裡env.observation_space[1].shape[0] 的輸出為 10。
        #範例裡env.observation_space[2].shape[0] 的輸出為 10。
        #所以這邊actor_dims為[8, 10, 10],會是actor的輸入維度
        #pig 這裡的 env.observation_space[i].shape[0] 要換成自己車載環境裡輸出的observation數量，自己寫一個副函式把observation丟進去計算shape
    actor_dims = env.observation_space_shape_calculate()
    
    #critic_dims輸入的維度矩陣
    critic_dims = sum(actor_dims) #actor維度相加
    
    #n_actions = env.action_space[0].n #總共有幾個動作
    # pig 這裡的env.action_space[0].n要換成自己車載環境的動作數量，因為我的動作數量都一樣，所以可以直接在init裡設定輸出就好
    n_actions = env.action_space
    
    #call MADDPG
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.001, beta=0.001, gamma=0.95, scenario=scenario, chkpt_dir='tmp/maddpg/')
    
    #call buffer
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)
    
    #PRINT_INTERVAL = 500 #每500打印一次
    N_GAMES = 2000 #epoch
    #MAX_STEP = 25 #設定是否到達最終狀態
    total_steps = 0
    score_history = []
    evaluate = False #判斷評估代理人的性能
    best_score = 0
    best_Episode_score = 0
    best_data_rate = []
    best_Transmission_delay = []
    best_power_consumption = []
    
    
    if evaluate: #如果evaluate為真,就加載儲存點模型
        maddpg_agents.load_checkpoint()
    
    for i in range(N_GAMES):
        obs = env.reset() #讀出環境的觀察狀態
        # pig env.reset()要在自己的車載環境裡寫,就是讀出環境現在所有代理人的觀察,注意每個代理人有自己的觀察，所以是n維
        # 範例理會有render出來是因為 env.reset()有call到，但我不用render()所以可以不用寫
        
        score = 0 #每個epoch的分數
        done = [False]*n_agents #設定所有代理人的終端狀態
        #episode_step = 0 
        nextstep_count = 1
        score_list = []
        data_rate_list = []
        Transmission_delay_list = []
        power_consumption_list = []
        
        while not any(done): #如果有任何done是false, not done=not false=true=1 ,while判斷為1就等於是無限迴圈,要終止的話就要所有done的值為true,not done才會為false
            #if evaluate: #如果evaluate為True的話
            #    env.render() #跑模擬環境GUI
                # pig render這東西我應該不需用，這是再跑模擬環境GUI的,可以直接註解掉或刪了
                
            nextstep_count = nextstep_count + 1
            actions = maddpg_agents.choose_action(obs) #每個代理人根據環境當下的觀察輸出一個動作機率矩陣
            # pig 這裡的actions是n為機率分布，要在step裡進行取樣計算
            
            obs_, reward, done, info, all_data_rate,all_Transmission_delay, all_power_consumption = env.step(obs, actions, nextstep_count)
            #pig env.step()要在自己的車載環境裡寫,總之return要回傳[下一個狀態(應該是可以直接去讀狀態csv的下一列),獎勵(我的reward function應該寫在這),done(是否跑完全檔案),info]
            #pig 注意 env.step()要回傳的是下一個狀態
            score += max(reward[0],reward[1],reward[2],reward[3])
            #score_list.append(reward)
            if (reward[0] >= reward[1]) and (reward[0] >= reward[2]) and (reward[0] >= reward[3]):
                data_rate_list.append(all_data_rate[0])
                Transmission_delay_list.append(all_Transmission_delay[0])
                power_consumption_list.append(all_power_consumption[0])
                
            elif (reward[1] >= reward[0]) and (reward[1] >= reward[2]) and (reward[1] >= reward[3]):
                data_rate_list.append(all_data_rate[1])
                Transmission_delay_list.append(all_Transmission_delay[1])
                power_consumption_list.append(all_power_consumption[1])
                
            elif (reward[2] >= reward[0]) and (reward[2] >= reward[1]) and (reward[2] >= reward[3]):
                data_rate_list.append(all_data_rate[2])
                Transmission_delay_list.append(all_Transmission_delay[2])
                power_consumption_list.append(all_power_consumption[2])
            
            elif (reward[3] >= reward[0]) and (reward[3] >= reward[1]) and (reward[3] >= reward[2]):
                data_rate_list.append(all_data_rate[3])
                Transmission_delay_list.append(all_Transmission_delay[3])
                power_consumption_list.append(all_power_consumption[3])
            
            else:
                print("repeat score")
            
            state = obs_list_to_state_vector(obs) #把觀察矩陣轉為一維向量，要寫到buffer裡的(所有代理人觀察flatten)
            state_ = obs_list_to_state_vector(obs_) #把下一個觀察矩陣轉為一維向量，要寫到buffer裡的(所有代理人觀察flatten)
            
            #if episode_step > MAX_STEP: #如果超過設置的步驟數，就把done轉為True
            #    done = [True]*n_agents
                # Pig 這段我應該可以不用，因為我可以在step裡做done的轉換
                
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)#把資料存進replay buffer
            
            if total_steps % 100 == 0 and not evaluate:#當total_steps跑100次後，進行學習
                maddpg_agents.learn(memory) #把內存緩衝區的資料讀進去學習，進行梯度下降
            
            obs = obs_ #要在while裡更新新的狀態,不然會一直是舊得observation進行learn
            
            #score += sum(reward)
            # pig 範例會把reward加起來是因為它的環境有對抗+合作所以算整體分數,但我的環境屬於競爭，所以計算score方式要改(取max)
            
            total_steps += 1
            #episode_step += 1
            
        #score += np.max(np.sum(np.array(score_list), axis = 0))
        if score > best_Episode_score:
            best_Episode_score = score
            print("best_Episode_score:",best_Episode_score)
            best_data_rate = data_rate_list
            best_Transmission_delay = Transmission_delay_list
            best_power_consumption = power_consumption_list
            
        score_history.append(score) #把每一個epoch的分數記錄下來
        avg_score = np.mean(score_history[-100:]) #計算平均分數,為最新一百場比賽的分數加起來平均
        if not evaluate:
            if avg_score > best_score: #如果平均分數大於最佳分數
                maddpg_agents.save_checkpoint() #保存模型,這樣就會將每次得到好的分數的模型都存下來
                best_score = avg_score #覆蓋掉最佳分數
        
        #if i % PRINT_INTERVAL == 0 and i > 0:
        print('episode', i, 'averange score {:.1f}'.format(avg_score))
        
    #繪製reward歷史圖表,也可以知道哪個epoch的得分最好
    x = [i+1 for i in range(N_GAMES)] #x軸,i+1=1
    filenames = 'Multi_MADDPG.png' #生成的收斂圖名稱,有調整超參數的話可以改變圖的名稱比較好分辨
    figure_file = 'plots/' + filenames
    plot_learning_curve(x, score_history, figure_file, best_data_rate, best_Transmission_delay, best_power_consumption)
    

        