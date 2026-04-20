import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Исходная функция
def f(x):
    return np.log(x)

# Последовательность простых функций fn(x)
def fn(x, n):
    a, b = 1.0, 4.0
    N = 2**n
    h = (b - a) / N

    k = np.floor((x - a) / h).astype(int)
    k = np.clip(k, 0, N - 1)

    x_left = a + k * h
    return np.log(x_left)

# Функция для меры Лебега-Стилтьеса
def F(x):
    return np.ceil(x**2)

# Точное аналитические вычисления интегралов
lebesgue_exact = 4 * math.log(4) - 3
stieltjes_exact = 0.5 * math.log(math.factorial(16))

print(f"∫ ln(x) dx = {lebesgue_exact:.10f}")
print(f"∫ ln(x) dμ_F = {stieltjes_exact:.10f}")

# Сетка значений x для графиков
x_vals = np.linspace(1, 4, 10000)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Графики $f_n(x)$ и $f(x)=ln(x)$ E=[1,4]", fontsize=13)

n_values_plot = [1, 2, 3, 4, 5, 8]

# Построил графики простых функций fn
for ax, n in zip(axes.flat, n_values_plot):
    y_fn = fn(x_vals, n)
    y_f = f(x_vals)
    ax.plot(x_vals, y_f, 'b-', label="f(x)=ln(x)", alpha=0.8)
    ax.step(x_vals, y_fn, where='post', color='red', label=f"fn, n={n}, N={2**n}")
    ax.fill_between(x_vals, y_fn, y_f, alpha=0.2, color='orange', label='погрешность')
    ax.set_title(f"n={n}, N={2**n}")
    ax.set_xlim(1, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вычисление интеграла Лебега
def compute_lebesgue(N):
    a, b = 1.0, 4.0
    h = (b - a) / N
    x_left = a + np.arange(N) * h
    return np.sum(np.log(x_left)) * h

print("Интеграл Лебега")

results_lebesgue = []

# Таблица с результатами: Значение, Погрешность и Время для интеграла Лебега
print(f"{'N':<10} {'Значение':<20} {'Погрешность':<15} {'Время'}")
for N in [10, 100, 1000]:
    t0 = time.time()
    # Вычислил значение интеграла
    val = compute_lebesgue(N)
    t1 = time.time()
    # Вычислил абсолютную погрешность
    err = abs(val - lebesgue_exact)

    print(f"{N:<10} {val:<20.10f} {err:<15.2e} {t1-t0:.6f}")
    results_lebesgue.append((N, val, err))

# Вычисление интеграла Лебега-Стилтьеса
def compute_stieltjes(N):
    s = 0.0
    a, b = 1.0, 4.0
    h = (b - a) / N

    for k in range(1, 17):
        x = np.sqrt(k)
        idx = int(np.floor((x - a) / h))
        idx = min(idx, N - 1)
        x_left = a + idx * h
        s += np.log(x_left)
    return s

print("Интеграл Лебега–Стилтьеса")

results_stieltjes = []

# Таблица с результатами: Значение, Погрешность и Время для интеграла Лебега-Стилтьеса
print(f"{'N':<10} {'Значение':<20} {'Погрешность':<15} {'Время'}")
for N in [50, 500, 5000]:
    t0 = time.time()
    # Вычислил значение интеграла
    val = compute_stieltjes(N)
    t1 = time.time()
    # Вычислил абсолютную погрешность
    err = abs(val - stieltjes_exact)

    print(f"{N:<10} {val:<20.10f} {err:<15.2e} {t1-t0:.6f}")
    results_stieltjes.append((N, val, err))

# Построил графики сходимости погрешности от N
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Сходимость интегралов", fontsize=12)

# Вычисление погрешности интеграла Лебега
Ns = [10, 20, 50, 100, 200, 500, 1000]
leb_vals = [compute_lebesgue(N) for N in Ns]
leb_errs = [abs(v - lebesgue_exact) for v in leb_vals]

ax1.loglog(Ns, leb_errs, 'bo-', linewidth=1.5)
ax1.set_title("Интеграл Лебега")
ax1.set_xlabel("N")
ax1.set_ylabel("Погрешность")

ax1.grid(True, which='major', linestyle='-', alpha=0.5)
ax1.grid(True, which='minor', linestyle=':', alpha=0.3)

# Вычисление погрешности интеграла Лебега-Стилтьеса
Ns2 = [10, 20, 50, 100, 200, 500, 1000, 2000]
st_vals = [compute_stieltjes(N) for N in Ns2]
st_errs = [abs(v - stieltjes_exact) for v in st_vals]

ax2.semilogx(Ns2, st_errs, 'rs-', linewidth=1.5)
ax2.set_title("Интеграл Лебега–Стилтьеса")
ax2.set_xlabel("N")
ax2.set_ylabel("Погрешность")

ax2.grid(True, which='major', linestyle='-', alpha=0.5)
ax2.grid(True, which='minor', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.show()