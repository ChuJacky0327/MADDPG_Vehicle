import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    plt.plot(x, running_avg)
    plt.title('Multi_MADDPG Running average of previous 100 scores') #圖的title
    plt.savefig(figure_file)
    
    running_avg = list(map(lambda x:[x],running_avg))
    
    with open('MADDPG_simulation1-avg_score.csv','w',newline='') as file: #'w'為覆寫
        write = csv.writer(file)
        #write.writerow(["ave_score"])
        for i in range(len(scores)):
            write.writerow(running_avg[i])

    