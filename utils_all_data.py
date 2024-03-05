import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


def plot_learning_curve(x, scores, figure_file, data_rate, Transmission_delay, power_consumption):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    plt.plot(x, running_avg)
    plt.title('Multi_MADDPG Running average of previous 100 scores') #圖的title
    plt.savefig(figure_file)
    
    running_avg = list(map(lambda x:[x],running_avg))
    data_rate = list(map(lambda x:[x],data_rate))
    Transmission_delay = list(map(lambda x:[x],Transmission_delay))
    power_consumption = list(map(lambda x:[x],power_consumption))
    
    with open('data/MADDPG_simulation1-avg_score.csv','w',newline='') as file: #'w'為覆寫
        write = csv.writer(file)
        #write.writerow(["ave_score"])
        for i in range(len(scores)):
            write.writerow(running_avg[i])
            
    with open('data/MADDPG_simulation1-best_data_rate.csv','w',newline='') as file: #'w'為覆寫
        write = csv.writer(file)
        #write.writerow(["ave_score"])
        for i in range(len(data_rate)):
            write.writerow(data_rate[i])
            
    with open('data/MADDPG_simulation1-best_Transmission_delay.csv','w',newline='') as file: #'w'為覆寫
        write = csv.writer(file)
        #write.writerow(["ave_score"])
        for i in range(len(Transmission_delay)):
            write.writerow(Transmission_delay[i])
            
    with open('data/MADDPG_simulation1-best_power_consumption.csv','w',newline='') as file: #'w'為覆寫
        write = csv.writer(file)
        #write.writerow(["ave_score"])
        for i in range(len(power_consumption)):
            write.writerow(power_consumption[i])

    