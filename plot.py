import matplotlib.pyplot as plt
import csv
import sys

def plot(alpha, gamma, eps, buckets):
    file_name = "results/a{}_g{}_e{}_b{}.csv".format(alpha,gamma,eps,buckets)
    moving_average = []
    step = 10

    with open(file_name,'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        new_sum_until_now = 0

        for res in data:
            new_sum_until_now += float(res[1])
            if (int(res[0])+1)%step == 0:
                moving_average.append(new_sum_until_now/step)
                new_sum_until_now = 0

        x, y = [], []
        for counter, num in enumerate(moving_average,1):
            x.append(num)
            y.append(counter*10)

    plt.plot(y,x)
    plt.xlabel('attempt')
    plt.ylabel('reward')
    plt.title("alpha={}, gamma={}, epsilon={}, buckets={}".format(alpha,gamma,eps,buckets))
    plt.savefig("plots/a{}_g{}_e{}_b{}.png".format(alpha,gamma,eps,buckets))

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

    plot(alfa,gamma,eps,buckets)

if __name__ == '__main__':
    main()
