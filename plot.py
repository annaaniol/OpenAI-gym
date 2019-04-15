import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

def plot(dir_name, alpha, gamma, eps, buckets):
    file_name = "results_{}/a{}_g{}_e{}_b{}.csv".format(dir_name,alpha,gamma,eps,buckets)
    moving_average_mean = []
    moving_average_stddev = []
    step = 20

    with open(file_name,'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        next(data, None)
        new_mean_until_now = 0
        new_stddev_until_now = 0

        for res in data:
            new_mean_until_now += float(res[1])
            new_stddev_until_now += float(res[2])
            if (int(res[0]))%step == 0:
                moving_average_mean.append(new_mean_until_now/step)
                moving_average_stddev.append(new_stddev_until_now/step)
                new_mean_until_now = 0
                new_stddev_until_now = 0

        x, mean, mean_minus_stddev, mean_plus_stddev = [], [], [], []
        for counter, res in enumerate(zip(moving_average_mean,moving_average_stddev),1):
            mean.append(res[0])
            mean_minus_stddev.append(res[0]-res[1])
            mean_plus_stddev.append(res[0]+res[1])
            x.append(counter*10)

    plt_mean, = plt.plot(x,mean,'b',label='mean')
    plt_mean_minus_stddev, = plt.plot(x,mean_minus_stddev,'--',label='- std dev')
    plt_mean_plus_stddev, = plt.plot(x,mean_plus_stddev,'--',label='+ std dev')
    plt.fill_between(x, mean_minus_stddev, mean_plus_stddev, color='grey', alpha='0.2')
    plt.xlabel('attempt')
    plt.ylabel('reward')
    plt.legend(handles=[plt_mean, plt_mean_minus_stddev, plt_mean_plus_stddev])
    plt.title("alpha={}, gamma={}, epsilon={}, buckets={}".format(alpha,gamma,eps,buckets))
    plt.savefig("plots_{}/a{}_g{}_e{}_b{}.png".format(dir_name,alpha,gamma,eps,buckets))
    plt.clf()

def main():
    argv = sys.argv
    try:
        alfa = float(argv[1])
        gamma = float(argv[2])
        eps = float(argv[3])
        buckets = int(argv[4])
    except Exception:
        print('invalid or missing parameters: alfa gamma eps buckets')
        print(traceback.format_exc())
        sys.exit(1)

    plot('stddev',alfa,gamma,eps,buckets)
    plot('sarsa_stddev',alfa,gamma,eps,buckets)

if __name__ == '__main__':
    main()
