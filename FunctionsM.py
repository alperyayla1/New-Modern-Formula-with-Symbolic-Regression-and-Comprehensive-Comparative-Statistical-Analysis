import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns
rcParams['font.weight'] = 'bold'
sns.set_theme()

def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value
def clear_db(db):
    DeletingRows = []

    i = 0
    while i < (len(db['test'])):
        checker = 0
        for j in range(4):
            try:
                if isinstance(db['result'].iloc[i + j], int):
                    checker += 1

            except:
                break

        if checker == 4:
            i += 4

        else:
            DeletingRows.append(i)
            i += 1

    db.drop(DeletingRows, inplace=True)
    db.dropna()

def passing_bablok(method1, method2):
    n_points = len(method1)
    sv = [] #List of gradients
    k = 0 # k is the number of gradients less than -1
    for i in range(n_points - 1):
        for j in range(i+1, n_points):
            dy = method2[j]-method1[i]
            dx = method1[j]-method1[i]
            if dx != 0:
                gradient = dy / dx
            elif dy < 0:
                gradient = -1.e+23
            elif dy > 0:
                gradient = 1.e+23
            else:
                gradient = None
            if gradient is not None:
                sv.append(gradient)
                k += (gradient < -1)
    sv.sort()
    m0 = (len(sv) - 1) / 2
    if m0 == int(m0):
        # If odd
        gradient_est = sv[k + int(m0)]
    else:
        # If even
        gradient_est = 0.5 * (sv[k + int(m0 - 0.5)] + sv[k + int(m0 + 0.5)])
    # Calculate the index of the upper and lower confidence bounds
    w = 1.96
    ci = w * math.sqrt((n_points * (n_points - 1) * (2 * n_points + 5)) / 18)
    n_gradients = len(sv)
    m1 = int(round((n_gradients - ci) / 2))
    m2 = n_gradients - m1 - 1
    # Calculate the lower and upper bounds of the gradient
    (gradient_lb, gradient_ub) = (sv[k + m1], sv[k + m2])

    def calc_intercept(method1, method2, gradient):
        """Calculate intercept given points and a gradient."""
        temp = []
        for i in range(len(method1)):
            temp.append(method2[i] - gradient * method1[i])
        return np.median(temp)

    # Calculate the intercept as the median of all the intercepts of all the
    # lines connecting each pair of points
    int_est = calc_intercept(method1, method2, gradient_est)
    int_ub = calc_intercept(method1, method2, gradient_lb)
    int_lb = calc_intercept(method1, method2, gradient_ub)

    return (gradient_est, gradient_lb, gradient_ub), (int_est, int_lb, int_ub)

def bablok_plot(model1, model2, beta, alpha, name):
    ax = plt.axes()
    #ax.set_title('Passing-Bablok Regression')
    ax.set_xlabel('d-LDL-C', fontweight='bold')
    ax.set_ylabel(name, fontweight='bold')
    # Scatter plot
    ax.scatter(model1, model2, c='k', s=20, alpha=0.6, marker='o')
    # Get axis limits
    left, right = plt.xlim(0, 350)
    bottom, top = plt.ylim(-10, 350)
    # Change axis limits
    ax.set_xlim(0, right)
    ax.set_ylim(bottom, top)
    # Reference line
    label = 'Reference line'
    ax.plot([left, right], [left, right], c='grey', ls='--', label=label)
    # Passing-Bablok regression line
    x = np.array([left, right])
    y = beta[0] * x + alpha[0]
    ax.plot(x, y, label=f'{beta[0]:4.2f}x + {alpha[0]:4.2f}')
    # Passing-Bablok regression line - confidence intervals
    x = np.array([left, right])
    y_lb = beta[1] * x + alpha[1]
    y_ub = beta[2] * x + alpha[2]
    label = f'Upper CI: {beta[2]:4.2f}x + {alpha[2]:4.2f}'
    ax.plot(x, y_ub, c='tab:blue', alpha=0.2, label=label)
    label = f'Lower CI: {beta[1]:4.2f}x + {alpha[1]:4.2f}'
    ax.plot(x, y_lb, c='tab:blue', alpha=0.2, label=label,)
    ax.fill_between(x, y_ub, y_lb, alpha=0.2)
    # Set aspect ratio
    ax.set_aspect('equal')
    # Legend
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Add a border around the axis
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    plt.show()

def bland_altman(model1, model2, name):
    mean = np.mean([model1, model2], axis=0)
    diff = model1 - model2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)
    CI_low = md - 1.96 * sd
    CI_high = md + 1.96 * sd

    plt.scatter(mean,diff,linewidth=0)
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    plt.title(f'Bland-Altman Plot of d-LDL-C and {name}')
    plt.xlabel("Means", fontweight='bold')
    plt.ylabel("Difference", fontweight='bold')
    plt.ylim(-120, 120)
    plt.xlim(0, 300)
    plt.legend()


    plt.text(310, md - 1.96 * sd,
             r'-1.96SD:' + "\n" + "%.2f" % CI_low,
             ha="center",
             va="center",
             )
    plt.text(310, md + 1.96 * sd,
             r'+1.96SD:' + "\n" + "%.2f" % CI_high,
             ha="center",
             va="center",
             )
    plt.text(310, md,
             r'Mean:' + "\n" + "%.2f" % md,
             ha="center",
             va="center",
             )

    plt.subplots_adjust(right=0.85)
    plt.show()



