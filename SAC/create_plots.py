import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

def plot_paths():
    path = "./data"
    rob = pd.read_csv(f'{path}/rob_positions.csv', header=None)
    ped = pd.read_csv(f'{path}/ped_positions.csv', header=None)
    #rob.plot.scatter(x=0, y=1, c=ped[1], colormap='viridis')
    #data2.plot.scatter(x=0, y=1)
    c = range(0, len(rob[0]))
    cmap = "rainbow" #"turbo" #"jet"
    #plt.xlim([-10, 10])
    #plt.ylim([-1.5, 2.])
    plt.scatter(rob[0], rob[1])
    plt.scatter(ped[0], ped[1])
    #plt.scatter(rob[0], rob[1], c=c, cmap=cmap) #, s=1000)
    #plt.scatter(ped[0], ped[1], c=c, cmap=cmap) #, s=1000)
    plt.colorbar(label="Steps")

    plt.show()


"""
fig,axes=plt.subplots(1,1)
axes.scatter(x,y)
axes.yaxis.set_major_locator(MaxNLocator(5))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes.xaxis.set_major_locator(MaxNLocator(5))
axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


"""