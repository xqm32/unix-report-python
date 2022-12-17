import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # axis X
    plt.xlabel("x(n)")
    plt.axhline(color="grey", linestyle="--")

    plt.xlim(100, 1000000)
    plt.xscale("log")
    # axis Y
    plt.ylabel("y(ms)")
    plt.axvline(color="grey", linestyle="--")
    
    with open("a3.txt", "r") as f:
        x_a3, y_a3 = eval(f.read())
    with open("sort.txt", "r") as f:
        x_sort, y_sort = eval(f.read())
    x_a3, y_a3 = np.array(x_a3), np.array(y_a3)
    x_sort, y_sort = np.array(x_sort), np.array(y_sort)

    plt.plot(x_a3, y_a3, "-", label="Algorithm 3 without Optimized f'(x)")
    plt.plot(x_sort, y_sort, "-", label="Bisect after Sort")
    plt.fill_between(x_a3, y_a3, y_sort, alpha=0.1, color="green")
    plt.plot(x_a3, y_sort - y_a3, "-", label="Difference")
    plt.fill_between(x_a3, y_sort - y_a3, alpha=0.1, color="green")
    plt.legend()
    plt.savefig("Algorithm Compare.png", dpi=600)
