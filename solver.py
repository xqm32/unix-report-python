from random import choice
import matplotlib.pyplot as plt
import numpy as np


class Solution:
    def __init__(
        self,
        a: float,
        b: list[float],
        c: list[float],
        d: float,
    ) -> None:
        self.a: float = a
        self.b: list[float] = b.copy()
        self.c: list[float] = c.copy()
        self.d: float = d
        # breakpoints
        self.bps: list[float] = [-ci / bi for bi, ci in zip(self.b, self.c)]

    def f(self, x: float):
        return (
            self.a / 2 * x * x
            + self.d * x
            + sum(bi * x + ci for bi, ci in zip(self.b, self.c) if bi * x + ci >= 0)
        )

    def fp(self, x: float):
        return (
            self.a * x
            + self.d
            + sum(bi for bi, ci in zip(self.b, self.c) if bi * x + ci >= 0)
        )

    def fp_zero(self, x: float):
        z = (
            -(self.d + sum(bi for bi, ci in zip(self.b, self.c) if bi * x + ci >= 0))
            / self.a
        )
        return z

    def find_sort(self, bps: list[float], pre=1e-4) -> float:
        bps = sorted(bps)
        for i, j in zip(bps[:-1], bps[1:]):
            if self.fp(i) * self.fp(j) < 0:
                z = self.fp_zero(i)
                if abs(self.fp(z)) < pre:
                    return z
                else:
                    return i
        return bps[-1]

    def find_a3(self, bps: list[float]):
        x: list[float] = bps.copy()
        k: int = 0
        xk: float = x[k]
        gk: float = self.fp(xk)
        U: set[int] = {i for i in range(len(x))}
        while len(U) != 0:
            k = choice(list(U))
            xk = x[k]
            L = {i for i in U if x[i] < xk}
            G = {i for i in U if x[i] >= xk}
            gk = self.fp(xk)
            if gk < 0:
                gk = gk + abs(b[k])
                if gk >= 0:
                    print("gk >= 0")
                    break
                else:
                    U = G - {k}
            else:
                U = L
            if len(U) == 0:
                print("len(U) == 0")
                xk = xk - gk / a
                break
        return xk

    def plt(self, f: bool = True, fp: bool = True, pre=1e-4):
        F_XS_COLOR = "#C76DA2"
        F_BPS_COLOR = "#8983BF"
        FP_XS_COLOR = "#05B9E2"
        FP_BPS_COLOR = "#32B897"
        FP_ZERO_COLOR = "#BB9727"
        FP_ANS_COLOR = "#F27970"

        xs = np.linspace(min(self.bps) - 1, max(self.bps) + 1, 10000)
        bps = [bp - pre for bp in self.bps]

        # axis X
        plt.xlabel("x")
        plt.axhline(color="grey", linestyle="--")
        # axis Y
        plt.ylabel("y")
        plt.axvline(color="grey", linestyle="--")

        if f:
            f_xs = np.array([self.f(x) for x in xs])
            f_bps = np.array([self.f(x) for x in bps])
            plt.plot(xs, f_xs, color=F_XS_COLOR)
            plt.plot(bps, f_bps, "o", color=F_BPS_COLOR, markersize=3)

        if fp:
            fp_xs = np.array([self.fp(x) for x in xs])
            fp_bps = np.array([self.fp(x) for x in bps])
            plt.plot(xs, fp_xs, color=FP_XS_COLOR)
            plt.plot(bps, fp_bps, "o", color=FP_BPS_COLOR, markersize=3)

            for bp, fp_bp in zip(bps, fp_bps):
                plt.axline(
                    (bp, fp_bp), slope=a, color="grey", linestyle=":", linewidth=0.5
                )
                z = self.fp_zero(bp)
                plt.plot(z, 0, "o", color=FP_ZERO_COLOR, markersize=3)

        z = self.find_a3(bps)
        plt.plot(z, self.fp(z), "o", color=FP_ANS_COLOR, markersize=3)

        plt.show()


# m = 3
# a = 1.5
# d = 1
# b = [1, 1.2, -0.9]
# c = [0.1, -1.4, -1.2]
m = 10
a = abs(np.random.normal(0, 1))
d = np.random.normal(0, 1)
b = list(np.random.normal(0, 1, m))
c = list(np.random.normal(0, 1, m))
Solution(a, b, c, d).plt(f=False)
