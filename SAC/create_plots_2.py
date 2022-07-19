import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from tbparse import SummaryReader

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

def get_plot_fix(dir, name, label, color=None):
    reader = SummaryReader(dir+name)
    df = reader.scalars
    df = df[df["tag"] == "rollout/ep_rew_mean"]
    print(df["step"].iat[-1])
    last = df["step"].iat[-1]
    print(df["step"].tail(5))
    print(df["step"].tail(5)+1)
    steps = pd.concat((df["step"], df["step"].head(500)+last))
    steps = pd.concat((steps, df["step"].head(500) + last*2))
    values = pd.concat((df["value"], df["value"].tail(500)))
    values = pd.concat((values, df["value"].tail(500)))
    plt.plot(steps/1e6, values, label=label)
    #df["step"].append(ran2)
    #df["value"].append(df["value"].tail(100000))
    #df = pd.concat((df, df.tail(1000000)))
    #plt.plot(df["step"]/1e6, df["value"], label=label)

def get_plot(dir, name, label, color=None):
    reader = SummaryReader(dir+name)
    df = reader.scalars
    df = df[df["tag"] == "rollout/ep_rew_mean"]
    plt.plot(df["step"]/1e6, df["value"], label=label)

def get_plot_smooth(dir, name, label, color=None):
    reader = SummaryReader(dir + name)
    df = reader.scalars
    df = df[df["tag"] == "rollout/ep_rew_mean"]
    smooth = df.ewm(alpha=(1 - 0.9)).mean()
    #plt.plot(df["step"] / 1e6, df["value"], label=label, alpha=0.4)
    plt.plot(smooth["step"] / 1e6, smooth["value"], label=label) #color

def plot_multienv():
    #colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["Env-I", "Env-X", "Env-H", "Env-IXH"]
    dir = "./sac_tensorboard/"
    get_plot("../../Master's/SAC/sac_tensorboard/", "straight_larger_net_120scan_1ped_4wp_ent003_1m", labels[0])
    get_plot(dir, "cross_larger_net_120scan_1ped_4wp_pot_003ent_ero6_2m", labels[1])
    get_plot(dir, "H_larger_net_120scan_1ped_4wp_pot_003ent_ero6_1m", labels[2])
    get_plot(dir, "larger_net_120scan_1ped_4wp_pot_003ent_ero6_3m", labels[3])
    plt.xlim(left=0, right=1)
    plt.ylim([-50, 30])
    plt.xlabel("million steps")
    plt.ylabel("average return")
    plt.legend(loc="best")
    #plt.grid(True) style=ggplot
    plt.show()

def plot_entropy():
    labels = ["0.01", "0.03", "0.1", "auto"]
    dir = "../../Master's/SAC/sac_tensorboard/"
    get_plot(dir, "straight_larger_net_120scan_1ped_4wp_ent001_1m", labels[0])
    get_plot(dir, "straight_larger_net_120scan_1ped_4wp_ent003_1m", labels[1])
    get_plot(dir, "larger_net_120scan_1ped_4wp_1-5m", labels[2])
    get_plot(dir, "larger_net_120_scan_1_ped_long", labels[3])
    plt.xlim(left=0.0, right=1)
    #plt.ylim([-50, 30])
    plt.xlabel("million steps")
    plt.ylabel("average return")
    plt.legend(loc="lower right", title="entropy coeff")
    # plt.grid(True) style=ggplot
    plt.show()

def plot_stacked():
    labels = ["0.003", "0.01", "0.03", "0.05", "auto"]
    dir = "./sac_tensorboard/"
    get_plot(dir, "256net_120scan_1ped_4wp_stack_pot_0003_sde_3m_1", labels[0])
    get_plot(dir, "512net_ls_120scan_1ped_4wp_stack_pot_001_sde_3m_1", labels[1])
    get_plot(dir, "256net_120scan_1ped_4wp_stack_003ent_sde_3m", labels[2])
    get_plot(dir, "cross_1024net_120scan_1ped_4wp_stack_pot_005ent_sde_5m_1", labels[3])
    get_plot(dir, "512net_120scan_1ped_4wp_stack_pot_2et_sde_10m", labels[4])

    plt.xlim(left=0.0, right=1)
    #plt.ylim([-50, 30])
    plt.xlabel("million steps")
    plt.ylabel("average return")
    plt.legend(loc="lower right", title="entropy coeff")
    # plt.grid(True) style=ggplot
    plt.show()

def plot_multiped():
    labels = ["0.005", "0.01", "0.03", "0.05", "auto"]
    dir = "./sac_tensorboard/"
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    get_plot(dir, "X_256net_2peds_0001ent_no_vel_15m_1", labels[0], colors[0])
    get_plot(dir, "larger_net_120scan_mped_4wp_pot_003ent_ero6_3m", labels[1], colors[1])
    get_plot(dir, "larger_net_120scan_123ped_orca_4wp_pot_003ent_sde_3m", labels[2], colors[2])
    #get_plot(dir, "larger_net_120scan_123ped_4wp_pot_002ent_sde_15m", labels[3])
    get_plot(dir, "512net_120scan_123ped_orca_4wp_pot_005ent_sde_3m", labels[3], colors[3])
    get_plot(dir, "larger_net_120scan_123ped_4wp_pot_2te_sde_3m", labels[4], colors[4])

    plt.xlim(left=0.0, right=1.5)
    plt.ylim([-120, 20])
    plt.xlabel("million steps")
    plt.ylabel("average return")
    plt.legend(loc="lower right", title="entropy coeff")
    # plt.grid(True) style=ggplot
    plt.show()

#plot_multienv()
#plot_entropy()
#plot_stacked()
plot_multiped()

"""
fig,axes=plt.subplots(1,1)
axes.scatter(x,y)
axes.yaxis.set_major_locator(MaxNLocator(5))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes.xaxis.set_major_locator(MaxNLocator(5))
axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


"""