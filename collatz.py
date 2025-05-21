import colorama
import matplotlib.pyplot as plt

from functools import cache
from colorama import init, Fore, Back, Style
from colorama.ansi import AnsiFore, AnsiBack

init(autoreset=True)

import numpy as np


def cprint(*text, fg: AnsiFore = Fore.WHITE, bg: AnsiBack = Back.BLACK, st: Style = Style.BRIGHT):
    for t in text:
        print(fg + bg + st + f"{t}", end=" ")


class Collatz:
    @classmethod
    @cache
    def f(cls, n: int) -> int:
        if n % 2 == 0:
            return n // 2
        else:
            return (3 * n + 1) // 2

    @classmethod
    def c(cls, n: int) -> list:
        if n <= 0:
            return []
        s = [n]
        while n != 1:
            n = cls.f(n)
            s.append(n)
        return s

    @staticmethod
    def p(s: list) -> tuple[int, int]:
        e = sum([1 for sk in s if sk % 2 == 0])
        o = sum([1 for sk in s if sk % 2 == 1])
        return e, o

    @classmethod
    def largest_seq(cls, k: int, ak: int = 1):
        a0 = 2 ** k * ak
        n0 = 2 * a0 - 1
        s = cls.c(n0)
        return s

    @classmethod
    def plot_parity_upto(cls, n: int):
        x, y = [], []
        for n in range(3, n):
            s = Collatz.c(n)
            e, o = Collatz.p(s)
            x.append(n)
            y.append(e / o)

        if any([yk <= 1.0 for yk in y]):
            print("Collatz parity ALERT!!!")

        plt.plot(x, y, "-or")
        plt.grid(True)
        plt.show()

    @classmethod
    def sieve_upto(cls, n: int, cols: int = 10):
        s = [cls.c(i) for i in range(1, n)]
        nmax = max([max(si) for si in s])
        tw = len(f"{nmax}")

        ij_pos_fn = lambda n, cols: (n // cols + (n % cols != 0), n % cols)

        rows, _ = ij_pos_fn(nmax, cols)

        # build the sieve
        sieve = np.arange(1, nmax + 1)
        last_line_cols = nmax % cols
        if last_line_cols > 0:
            sieve = np.pad(sieve, (0, cols - last_line_cols), constant_values=0)
        sieve = sieve.reshape((rows, cols))

        # build the adjacent sieve for counting when they appears
        sieve_adj = np.zeros_like(sieve)

        # compute the sieve
        for k, sk in enumerate(s, start=1):
            for n in sk:
                i, j = ij_pos_fn(n, cols)
                if int(sieve_adj[i - 1, j - 1]) == 0:
                    sieve_adj[i - 1, j - 1] = k

        # Print the sieve
        for i in range(rows):
            for j in range(cols):
                sij = sieve[i, j]
                sija = sieve_adj[i, j]
                if sija != 0:
                    cprint(f"{sij:{tw}}", fg=Fore.LIGHTGREEN_EX)
                else:
                    cprint(f"{sij:{tw}}", fg=Fore.YELLOW)
            print("")
        print("")
        # Print the adjacent sieve
        for i in range(rows):
            for j in range(cols):
                sija = sieve_adj[i, j]
                if sija != 0:
                    cprint(f"{sija:{tw}}", fg=Fore.LIGHTGREEN_EX)
                else:
                    cprint(f"{sija:{tw}}", fg=Fore.YELLOW)
            print("")


if __name__ == '__main__':
    # Collatz.plot_parity_upto(10000)
    # a = Collatz.largest_seq(k=5, ak=3)
    # print(a)
    Collatz.sieve_upto(20, cols=10)
