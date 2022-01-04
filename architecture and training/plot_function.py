# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from os.path import isfile
        
def plot_hitrate_maxavgmin(x_vals, hitrate_vals_max, hitrate_vals_avg, hitrate_vals_min,
                           xlabel, save_path, y_scale = (0, 100)):
    
    #plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["font.size"] = 30
    #plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.linewidth'] = 3
    fig, ax = plt.subplots(figsize = (15, 15))
    
    ax.set_xlabel(xlabel, fontsize = 30, labelpad = 15)
    ax.set_ylabel("HitRate@10 Score/%", fontsize = 30, labelpad = 15)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals)
    ax.tick_params(which = 'major', length = 10, width = 4)
    ax.set_ylim(y_scale)
    
    ax.plot(x_vals, hitrate_vals_max, color = "r", marker = ".", markersize = 30, linewidth = 5)
    ax.plot(x_vals, hitrate_vals_avg, color = "g", marker = ".", markersize = 30, linewidth = 5)
    ax.plot(x_vals, hitrate_vals_min, color = "b", marker = ".", markersize = 30, linewidth = 5)
    ax.legend(labels = ["Maximum", "Average", "Minimum"])
    
    if not isfile(save_path):
        plt.savefig(save_path)
        
def plot_hitrate_vs_hyperparams(x_vals, hitrate_vals_MLP1, hitrate_vals_MLP2, 
                                xlabel, save_path, y_scale = (0, 100)):
    
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["font.size"] = 45
    plt.rcParams['axes.titlesize'] = 60
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (45, 45))
    
    ax.set_xlabel(xlabel, fontsize = 50)
    ax.set_ylabel("HitRate@10 Score/%", fontsize = 50)
    ax.set_xticks(x_vals)
    ax.set_ylim(y_scale)
    
    ax.plot(x_vals, hitrate_vals_MLP1, color = "r")
    ax.plot(x_vals, hitrate_vals_MLP2, color = "g")
    ax.legend(labels = ["MLP Units = (16, 8)", "MLP Units = (32, 16"])
    
    if not isfile(save_path):
        plt.savefig(save_path)
        
if __name__ == "__main__":
    plot_hitrate_maxavgmin(x_vals = [1, 2, 3, 4, 5], hitrate_vals_max = [56.6, 57.8, 59.1, 60.4, 58.9],
                           hitrate_vals_avg = [54.2, 56.6, 57.7, 58.6, 58.0],
                           hitrate_vals_min = [50.3, 53.0, 54.9, 55.3, 54.4], 
                           xlabel = "Negative:Positive Samples Ratio", y_scale = (45, 65),
                           save_path = "D:/Movie Recommendation System Project/model data/results graphs/hitrate_vs_ratio")
    
    plot_hitrate_maxavgmin(x_vals = [100, 250, 500], hitrate_vals_max = [60.4, 52.8, 52.3],
                           hitrate_vals_avg = [58.6, 50.7, 51.0],
                           hitrate_vals_min = [55.3, 49.9, 49.8], 
                           xlabel = "Minimum Samples Per User", y_scale = (45, 65),
                           save_path = "D:/Movie Recommendation System Project/model data/results graphs/hitrate_vs_minsamples")
