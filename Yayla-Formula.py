import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from gplearn.genetic import SymbolicRegressor
from FunctionsM import *

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
# Read the Excel file
FileDirectory = "C:/Users/alper/Downloads/6 aylık lipit.xlsx"
result_df = pd.read_excel(FileDirectory, usecols='G, F, Q, R')

column_names = ['age', 'gender', 'test', 'result']
result_df.columns = column_names
result_df['result'] = result_df['result'].astype("str").apply(convert_to_int)
result_df.fillna(method='ffill', inplace=True)
# taking age, test name, test result

clear_db(result_df)  # clearing database

result_df.reset_index(drop=True, inplace=True)

age_and_dependents = [];
LDL = []
gender = []
k = 2

# getting variables
while (k < (len(result_df['test']))):
    if result_df['result'][k - 1] < 1500:
        LDL.append(result_df['result'].iloc[k])
        age_and_dependents.append(
            [result_df['age'].iloc[k], result_df['result'].iloc[k - 2], result_df['result'].iloc[k - 1],
             result_df['result'].iloc[k + 1]])
        gender.append(result_df['gender'].iloc[k])
        k += 4
    else:
        k += 4

'''
#Finding formula with symbolic regression model
print("Total Number of Patient: ",len(LDL)) #printing total number of patient we work on
age_and_dependents_train, age_and_dependents_test, LDL_train, LDL_test = train_test_split(age_and_dependents, LDL, test_size=0.2, random_state=42)
est_gp = SymbolicRegressor(generations=10,parsimony_coefficient=0.05,const_range=(-100,100),stopping_criteria=0.001, population_size=8000,
                    verbose=1,function_set=('add','sub','mul','div','sqrt','log'))

est_gp.fit(age_and_dependents_train, LDL_train)
LDL_pred = est_gp.predict(age_and_dependents_test)
mse = mean_squared_error(LDL_test, LDL_pred)
r2 = r2_score(LDL_test, LDL_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
best_model = est_gp._program
print(best_model)
'''

# Sorting and Creating Parameter Arrays
LDL = np.array(LDL)

sorted_indices = np.argsort(LDL)
age_and_dependents = np.array(age_and_dependents)
age_and_dependents = age_and_dependents[sorted_indices]
LDL = LDL[sorted_indices]

gender = np.array(gender)
gender = gender[sorted_indices]

age = age_and_dependents[:, 0].astype(float)
KLS = age_and_dependents[:, 1].astype(float)
TGL = age_and_dependents[:, 2].astype(float)
HDL = age_and_dependents[:, 3].astype(float)


def martin_constant(TGL_Value, HDL_Value):
    martin_path = "C:/Users/alper/OneDrive/Masaüstü/martindataset.xlsx"
    MartinData = pd.read_excel(martin_path, header=None)

    # Ignoring the value at [0, 0]
    MartinData.iloc[0, 0] = None

    # Convert to NumPy array and then to float
    MartinData = MartinData.to_numpy().astype(float)

    row_number = 69
    column_number = 6

    for idx, row in enumerate(MartinData[1:, 0], start=1):  # Iterate through the first column
        if TGL_Value <= row:
            row_number = idx
            break

    for j, column in enumerate(MartinData[0, 1:]):  # Iterate through the first row
        if HDL_Value <= column:
            column_number = j + 1
            break

    return MartinData[row_number, column_number]


# Example usage:
# result = martin_constant(TGL_Value, HDL_Value)
# print(result)

print(martin_constant(161, 161))
# 0.8
# Creating Formulas
YaylaSum = 0.904 * (KLS - HDL - np.sqrt(TGL))
Friedewald = KLS - HDL - TGL / 5
Yayla = KLS - HDL - (np.sqrt(TGL) * KLS / 100)
Sampson = (KLS / 0.948) - (HDL / 0.971) - (TGL / 8.56 + TGL * (KLS - HDL) / 2140 - (TGL ** 2) / 16100) - 9.44
Martin = np.zeros(len(LDL))
for i in range(len(LDL)):
    Martin[i] = KLS[i] - HDL[i] - (TGL[i] / martin_constant(TGL[i], KLS[i] - HDL[i]))

x_values = np.arange(len(LDL))
xhun = np.arange(1, 1001)

# CREATING SAMPLE DATAS
# Show the plot
indices = np.arange(len(LDL))

# Set seed for reproducibility
sample_indices = np.random.choice(indices, size=1000, replace=False)
sorted_sample_indices = np.sort(sample_indices)

sampled_LDL = LDL[sorted_sample_indices]
sampled_Friedewald = Friedewald[sorted_sample_indices]
sampled_Yayla = Yayla[sorted_sample_indices]
sampled_YaylaSum = YaylaSum[sorted_sample_indices]
sampled_Sampson = Sampson[sorted_sample_indices]

print(f"Total Number of Patient {len(LDL)}")
print(f"Age mean, min, max: {np.mean(age):.2f},{np.min(age):.2f},{np.max(age):.2f}")
print(f"Kolesterol mean, min, max: {np.mean(KLS):.2f},{np.min(KLS):.2f},{np.max(KLS):.2f}")
print(f"LDL mean, min, max: {np.mean(LDL):.2f},{np.min(LDL):.2f},{np.max(LDL):.2f}")
print(f"Friedewald mean, min, max: {np.mean(Friedewald):.2f},{np.min(Friedewald):.2f},{np.max(Friedewald):.2f}")
print(f"Sampson mean, min, max: {np.mean(Sampson):.2f},{np.min(Sampson):.2f},{np.max(Sampson):.2f}")
print(f"Yayla mean, min, max: {np.mean(Yayla):.2f},{np.min(Yayla):.2f},{np.max(Yayla):.2f}")
print(f"Friedewald mean, min, max: {np.mean(Friedewald):.2f},{np.min(Friedewald):.2f},{np.max(Friedewald):.2f}")
print(f"Martin-Hopkins mean, min, max: {np.mean(Martin):.2f},{np.min(Martin):.2f},{np.max(Martin):.2f}")
print(f"HDL mean, min, max: {np.mean(HDL):.2f},{np.min(HDL):.2f},{np.max(HDL):.2f}")
print(f"TGL mean, min, max: {np.mean(TGL):.2f},{np.min(TGL):.2f},{np.max(TGL):.2f}")
unique_elements, counts = np.unique(gender, return_counts=True)
total_elements = len(gender)
for element, count in zip(unique_elements, counts):
    percentage = (count / total_elements) * 100
    print(f"{element}: {count} ({percentage:.2f}%)")

print(f"Friedewald mean squared error: {mean_squared_error(LDL, Friedewald):.2f}")
print(f"Sampson mean squared error: {mean_squared_error(LDL, Sampson):.2f}")
print(f"Yayla mean squared error: {mean_squared_error(LDL, Yayla):.2f}")
print(f"Martin-Hopkins mean squared error: {mean_squared_error(LDL, Martin):.2f}")


def set_publication_quality():
    """Set publication quality parameters for matplotlib"""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'Arial'  # Standard font for publications
    rcParams['font.size'] = 8  # Standard font size
    rcParams['axes.linewidth'] = 1  # Frame width
    rcParams['xtick.major.width'] = 1  # Tick width
    rcParams['ytick.major.width'] = 1
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['axes.labelsize'] = 10
    rcParams['legend.fontsize'] = 8
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.format'] = 'tiff'

errorsYayla = Yayla - LDL
errorsFriedewald = Friedewald - LDL
errorsSampson = Sampson - LDL
errorsMartin = Martin - LDL
sns.kdeplot(errorsYayla, color='darkblue', label='Y-LDL-C')
sns.kdeplot(errorsFriedewald, color='green', label='F-LDL-C')
sns.kdeplot(errorsSampson, color='red', label='S-LDL-C')
sns.kdeplot(errorsMartin, color='cornflowerblue', label='EM-LDL-C')
plt.xlabel('Error (mg/dl)', fontweight='bold')
plt.ylabel('Probability Density', fontweight='bold')
plt.xlim(-50, 50)
plt.legend(handlelength=2.0, prop={'weight': 'bold'})
# plt.title('Error Distribution of LDL-C Calculation Methods', fontweight='bold')
plt.show()

tgl99 = []
tgl149 = []
tgl199 = []
tgl399 = []
tgl400plus = []
for i in range(len(TGL)):
    if TGL[i] < 100:
        tgl99.append(i)
    elif TGL[i] < 150:
        tgl149.append(i)
    elif TGL[i] < 200:
        tgl199.append(i)
    elif TGL[i] < 400:
        tgl399.append(i)
    else:
        tgl400plus.append(i)

ldl70 = []
ldl100 = []
ldl130 = []
ldl160 = []
ldl190 = []
ldl190plus = []
for i in range(len(TGL)):
    if LDL[i] < 70:
        ldl70.append(i)
    elif LDL[i] < 100:
        ldl100.append(i)
    elif LDL[i] < 130:
        ldl130.append(i)
    elif LDL[i] < 160:
        ldl160.append(i)
    elif LDL[i] < 190:
        ldl190.append(i)
    else:
        ldl190plus.append(i)

print(f"Friedewald mean squared error LDL < 70: {mean_squared_error(LDL[ldl70], Friedewald[ldl70]):.2f}")
print(f"Friedewald mean squared error LDL 70 to 99: {mean_squared_error(LDL[ldl100], Friedewald[ldl100]):.2f}")
print(f"Friedewald mean squared error LDL 100 to 129: {mean_squared_error(LDL[ldl130], Friedewald[ldl130]):.2f}")
print(f"Friedewald mean squared error LDL 130 to 159: {mean_squared_error(LDL[ldl160], Friedewald[ldl160]):.2f}")
print(f"Friedewald mean squared error LDL 160 to 189: {mean_squared_error(LDL[ldl190], Friedewald[ldl190]):.2f}")
print(f"Friedewald mean squared error LDL >= 190: {mean_squared_error(LDL[ldl190plus], Friedewald[ldl190plus]):.2f}")

print(f"Sampson mean squared error LDL < 70: {mean_squared_error(LDL[ldl70], Sampson[ldl70]):.2f}")
print(f"Sampson mean squared error LDL 70 to 99: {mean_squared_error(LDL[ldl100], Sampson[ldl100]):.2f}")
print(f"Sampson mean squared error LDL 100 to 129: {mean_squared_error(LDL[ldl130], Sampson[ldl130]):.2f}")
print(f"Sampson mean squared error LDL 130 to 159: {mean_squared_error(LDL[ldl160], Sampson[ldl160]):.2f}")
print(f"Sampson mean squared error LDL 160 to 189: {mean_squared_error(LDL[ldl190], Sampson[ldl190]):.2f}")
print(f"Sampson mean squared error LDL >= 190: {mean_squared_error(LDL[ldl190plus], Sampson[ldl190plus]):.2f}")

print(f"Yayla mean squared error LDL < 70: {mean_squared_error(LDL[ldl70], Yayla[ldl70]):.2f}")
print(f"Yayla mean squared error LDL 70 to 99: {mean_squared_error(LDL[ldl100], Yayla[ldl100]):.2f}")
print(f"Yayla mean squared error LDL 100 to 129: {mean_squared_error(LDL[ldl130], Yayla[ldl130]):.2f}")
print(f"Yayla mean squared error LDL 130 to 159: {mean_squared_error(LDL[ldl160], Yayla[ldl160]):.2f}")
print(f"Yayla mean squared error LDL 160 to 189: {mean_squared_error(LDL[ldl190], Yayla[ldl190]):.2f}")
print(f"Yayla mean squared error LDL >= 190: {mean_squared_error(LDL[ldl190plus], Yayla[ldl190plus]):.2f}")

print(f"Martin mean squared error LDL < 70: {mean_squared_error(LDL[ldl70], Martin[ldl70]):.2f}")
print(f"Martin mean squared error LDL 70 to 99: {mean_squared_error(LDL[ldl100], Martin[ldl100]):.2f}")
print(f"Martin mean squared error LDL 100 to 129: {mean_squared_error(LDL[ldl130], Martin[ldl130]):.2f}")
print(f"Martin mean squared error LDL 130 to 159: {mean_squared_error(LDL[ldl160], Martin[ldl160]):.2f}")
print(f"Martin mean squared error LDL 160 to 189: {mean_squared_error(LDL[ldl190], Martin[ldl190]):.2f}")
print(f"Martin mean squared error LDL >= 190: {mean_squared_error(LDL[ldl190plus], Martin[ldl190plus]):.2f}")

print(len(ldl70))
print(len(ldl100))
print(len(ldl130))
print(len(ldl160))
print(len(ldl190))
print(len(ldl190plus))

print(len(tgl99))
print(len(tgl149))
print(len(tgl199))
print(len(tgl399))
print(len(tgl400plus))

print(f"Friedewald mean squared error TGL < 100: {mean_squared_error(LDL[tgl99], Friedewald[tgl99]):.2f}")
print(f"Friedewald mean squared error TGL 100 to 149: {mean_squared_error(LDL[tgl149], Friedewald[tgl149]):.2f}")
print(f"Friedewald mean squared error TGL 150 to 199: {mean_squared_error(LDL[tgl199], Friedewald[tgl199]):.2f}")
print(f"Friedewald mean squared error TGL 200 to 399: {mean_squared_error(LDL[tgl399], Friedewald[tgl399]):.2f}")
print(f"Friedewald mean squared error TGL >= 400: {mean_squared_error(LDL[tgl400plus], Friedewald[tgl400plus]):.2f}")

print(f"Sampson mean squared error TGL < 100: {mean_squared_error(LDL[tgl99], Sampson[tgl99]):.2f}")
print(f"Sampson mean squared error TGL 100 to 149: {mean_squared_error(LDL[tgl149], Sampson[tgl149]):.2f}")
print(f"Sampson mean squared error TGL 150 to 199: {mean_squared_error(LDL[tgl199], Sampson[tgl199]):.2f}")
print(f"Sampson mean squared error TGL 200 to 399: {mean_squared_error(LDL[tgl399], Sampson[tgl399]):.2f}")
print(f"Sampson mean squared error TGL >= 400: {mean_squared_error(LDL[tgl400plus], Sampson[tgl400plus]):.2f}")

print(f"Yayla mean squared error TGL < 100: {mean_squared_error(LDL[tgl99], Yayla[tgl99]):.2f}")
print(f"Yayla mean squared error TGL 100 to 149: {mean_squared_error(LDL[tgl149], Yayla[tgl149]):.2f}")
print(f"Yayla mean squared error TGL 150 to 199: {mean_squared_error(LDL[tgl199], Yayla[tgl199]):.2f}")
print(f"Yayla mean squared error TGL 200 to 399: {mean_squared_error(LDL[tgl399], Yayla[tgl399]):.2f}")
print(f"Yayla mean squared error TGL >= 400: {mean_squared_error(LDL[tgl400plus], Yayla[tgl400plus]):.2f}")

print(f"Martin mean squared error TGL < 100: {mean_squared_error(LDL[tgl99], Martin[tgl99]):.2f}")
print(f"Martin mean squared error TGL 100 to 149: {mean_squared_error(LDL[tgl149], Martin[tgl149]):.2f}")
print(f"Martin mean squared error TGL 150 to 199: {mean_squared_error(LDL[tgl199], Martin[tgl199]):.2f}")
print(f"Martin mean squared error TGL 200 to 399: {mean_squared_error(LDL[tgl399], Martin[tgl399]):.2f}")
print(f"Martin mean squared error TGL >= 400: {mean_squared_error(LDL[tgl400plus], Martin[tgl400plus]):.2f}")

'''
#Scatter plotting Friedewald Yayla and Sampson with real LDL

fig, (ax1,ax2,ax4) = plt.subplots(1,3)
ax1.set_ylabel('mg/dL', fontweight='bold')
ax1.scatter(x_values, Friedewald, label='F-LDL-C',color='seagreen',alpha=0.5,yunits='mg/dL')
ax2.scatter(x_values, Sampson, label='S-LDL-C', color='lime', alpha=0.5)
ax4.scatter(x_values, Yayla, label='Y-LDL-C',color='blueviolet',alpha=0.5)
#ax3.scatter(xhun, sampled_YaylaSum, label='YaylaSum', color='gray', alpha=0.5)


desired_y_ticks = [0,50,100,150,200,250,300,350]
desired_x_ticks = [0,2838]
ax1.set_yticks(desired_y_ticks)
ax2.set_yticks(desired_y_ticks)
#ax3.set_yticks(desired_y_ticks)
ax4.set_yticks(desired_y_ticks)


ax1.set_xticks(desired_x_ticks)
ax2.set_xticks(desired_x_ticks)
ax4.set_xticks(desired_x_ticks)
ax1.plot(x_values, LDL, label='d-LDL-C',linewidth=2,color='blue',alpha=0.5)
ax4.plot(x_values, LDL, label='d-LDL-C',linewidth=2,color='blue',alpha=0.5)
ax2.plot(x_values, LDL, label='d-LDL-C',linewidth=2,color='blue',alpha=0.5, )
#ax3.plot(xhun, sampled_LDL, label='LDL',linewidth=2,color='blue',alpha=0.5)


ax1.legend()
ax2.legend()
#ax3.legend()
ax4.legend()

# Show the plot
plt.show()








#Checking age corelation

correlation_coefficient, p_value = pearsonr(age, LDL)

# Fit a linear regression line
slope, intercept = np.polyfit(age, LDL, 1)
regression_line = slope * age + intercept
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.8f}")
# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(age, LDL, label=f'Pearson Correlation: {correlation_coefficient:.2f}', color='cornflowerblue')
plt.plot(age, regression_line, label='Linear Regression Line', color='red', linestyle='--',linewidth=2)
plt.xlabel('Age', fontweight='bold')
plt.ylabel('LDL-C (mg/dl)', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()












#checking gender corelation
gender_numeric = np.where(gender == 'E', 0, 1)

# Calculate Spearman correlation coefficient
correlation_coefficient, p_value = spearmanr(gender_numeric, LDL)

print(f"Spearman Correlation Coefficient: {correlation_coefficient:.8f}")
print(f"P-value: {p_value:.4f}")

# Create a swarm plot using seaborn
plt.figure(figsize=(10, 6))
sns.swarmplot(x=gender, y=LDL, palette='viridis')
plt.xlabel('Gender', fontweight='bold')
plt.ylabel('LDL-C (mg/dl)', fontweight='bold')
plt.show()

'''

# passing bablok for graphs:
beta, alpha = passing_bablok(LDL, Sampson)
bablok_plot(LDL, Sampson, beta, alpha, 'S-LDL-C')

beta, alpha = passing_bablok(LDL, Friedewald)
bablok_plot(LDL, Friedewald, beta, alpha, 'F-LDL-C')

beta, alpha = passing_bablok(LDL, Yayla)
bablok_plot(LDL, Yayla, beta, alpha, 'Y-LDL-C')

beta, alpha = passing_bablok(LDL, Martin)
bablok_plot(LDL, Martin, beta, alpha, 'EM-LDL-C')

bland_altman(LDL, Sampson, 'S-LDL-C')

bland_altman(LDL, Yayla, 'Y-LDL-C')

bland_altman(LDL, Friedewald, 'F-LDL-C')

bland_altman(LDL, Martin, 'EM-LDL-C')


def analyze_ldl_classification(LDL, Friedewald, Yayla, Sampson, Martin):
    """Calculate classification metrics for all methods"""

    def get_ldl_group(value):
        if value < 70:
            return 0  # 'LDL<70'
        elif value < 100:
            return 1  # '70≤LDL<100'
        elif value < 130:
            return 2  # '100≤LDL<130'
        elif value < 160:
            return 3  # '130≤LDL<160'
        elif value < 190:
            return 4  # '160≤LDL<190'
        else:
            return 5  # 'LDL≥190'

    # Convert actual LDL values to groups
    true_groups = np.array([get_ldl_group(x) for x in LDL])

    # Convert predicted values to groups
    methods = {
        'Friedewald': Friedewald,
        'Yayla': Yayla,
        'Sampson': Sampson,
        'Martin': Martin
    }

    results = {}
    for method_name, predictions in methods.items():
        pred_groups = np.array([get_ldl_group(x) for x in predictions])

        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(true_groups, pred_groups)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_groups, pred_groups)

        results[method_name] = {
            'kappa': kappa,
            'confusion_matrix': conf_matrix
        }

    return results


def plot_kappa_scores(results):
    """Plot Cohen's Kappa scores"""
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    kappa_scores = [results[method]['kappa'] for method in methods]

    bars = plt.bar(methods, kappa_scores)

    # Customize plot
    plt.title("Cohen's Kappa Scores for Different Methods", fontsize=12)
    plt.ylabel("Kappa Score")
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add interpretation guide
    interpretation_text = """
    Kappa Interpretation:
    < 0.00: Poor
    0.00-0.20: Slight
    0.21-0.40: Fair
    0.41-0.60: Moderate
    0.61-0.80: Substantial
    0.81-1.00: Almost Perfect
    """
    plt.figtext(1.02, 0.5, interpretation_text, fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('cohens_kappa_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrices(results):
    """Plot confusion matrices for all methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    group_labels = ['LDL<70', '70≤LDL<100', '100≤LDL<130',
                    '130≤LDL<160', '160≤LDL<190', 'LDL≥190']

    for idx, (method, results) in enumerate(results.items()):
        conf_matrix = results['confusion_matrix']

        # Calculate percentages
        conf_matrix_pct = (conf_matrix.T / conf_matrix.sum(axis=1)).T * 100

        sns.heatmap(conf_matrix_pct,
                    annot=True,
                    fmt='.1f',
                    cmap='Blues',
                    xticklabels=group_labels,
                    yticklabels=group_labels,
                    ax=axes[idx])

        axes[idx].set_title(f'{method} Confusion Matrix (%)')
        axes[idx].set_ylabel('True Group')
        axes[idx].set_xlabel('Predicted Group')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


# After your existing analysis, add these lines:
# Calculate classification metrics
classification_results = analyze_ldl_classification(LDL, Friedewald, Yayla, Sampson, Martin)

# Plot Kappa scores
plot_kappa_scores(classification_results)

# Plot confusion matrices
plot_confusion_matrices(classification_results)

# Print detailed results
print("\nCohen's Kappa Scores:")
for method, results in classification_results.items():
    print(f"{method}: {results['kappa']:.3f}")





def calculate_both_weighted_kappas(true_groups, pred_groups):
    """
    Calculate both linear and quadratic weighted kappa
    """
    linear_kappa = cohen_kappa_score(true_groups, pred_groups, weights='linear')
    quadratic_kappa = cohen_kappa_score(true_groups, pred_groups, weights='quadratic')
    return linear_kappa, quadratic_kappa


def analyze_ldl_classification_both_kappas(LDL, Friedewald, Sampson, Martin, Yayla):
    """Calculate both types of weighted kappa for all methods in specific order"""

    def get_ldl_group(value):
        if value < 70:
            return 0
        elif value < 100:
            return 1
        elif value < 130:
            return 2
        elif value < 160:
            return 3
        elif value < 190:
            return 4
        else:
            return 5

    # Convert values to groups
    true_groups = np.array([get_ldl_group(x) for x in LDL])

    # Define methods in desired order
    methods = {
        'Friedewald': Friedewald,
        'Sampson': Sampson,
        'Martin': Martin,
        'Yayla': Yayla
    }

    results = {}
    for method_name, predictions in methods.items():
        pred_groups = np.array([get_ldl_group(x) for x in predictions])
        linear_k = cohen_kappa_score(true_groups, pred_groups, weights='linear')
        quadratic_k = cohen_kappa_score(true_groups, pred_groups, weights='quadratic')

        results[method_name] = {
            'linear_kappa': linear_k,
            'quadratic_kappa': quadratic_k
        }

    return results


# Calculate kappa scores
kappa_results = analyze_ldl_classification_both_kappas(LDL, Friedewald, Sampson, Martin, Yayla)

# Print results
print("\nDetailed Kappa Analysis Results:")
print("-" * 50)
for method, results in kappa_results.items():
    print(f"\n{method}:")
    print(f"Linear Weighted Kappa: {results['linear_kappa']:.3f}")
    print(f"Quadratic Weighted Kappa: {results['quadratic_kappa']:.3f}")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Choose your preferred color scheme by uncommenting one of these options:

# Option 1: Orange-Green
colors = ['#ff9f43', '#10ac84']  # Warm orange and forest green

# Option 2: Blue tones
# colors = ['#2e86de', '#48dbfb']  # Deep blue and light blue

# Option 3: Orange-Blue
# colors = ['#ff9f43', '#2e86de']  # Warm orange and deep blue

bar_width = 0.35
opacity = 0.8

# Prepare data
methods = list(kappa_results.keys())
linear_kappas = [results['linear_kappa'] for results in kappa_results.values()]
quadratic_kappas = [results['quadratic_kappa'] for results in kappa_results.values()]

# Create bars
x = np.arange(len(methods))
rects1 = ax.bar(x - bar_width / 2, linear_kappas, bar_width,
                color='aquamarine', alpha=opacity, label='Linear Weighted Kappa',
                edgecolor='black', linewidth=1)
rects2 = ax.bar(x + bar_width / 2, quadratic_kappas, bar_width,
                color='tomato', alpha=opacity, label='Quadratic Weighted Kappa',
                edgecolor='black', linewidth=1)

# Customize plot
ax.set_ylabel('Kappa Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Weighted Kappa Scores',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10, fontweight='bold')

# Set y-axis limits with some padding
ax.set_ylim(0, max(max(linear_kappas), max(quadratic_kappas)) + 0.1)


# Add value labels
def add_value_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')


add_value_labels(rects1)
add_value_labels(rects2)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
ax.set_axisbelow(True)

# Customize legend
ax.legend(fontsize=10, loc='upper left')

# Set background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Adjust layout
plt.tight_layout()

# Save plot with high resolution (300 DPI for publication quality)
plt.savefig('weighted_kappa_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


def create_publication_kappa_plot(kappa_results, save_path='kappa_comparison', single_column=True):
    """Create publication-quality kappa comparison plot"""

    # Set style for publication
    set_publication_quality()

    # Set figure size based on journal column width
    if single_column:
        fig_width = 3.5  # inches - single column
    else:
        fig_width = 7.2  # inches - double column

    fig_height = fig_width * 0.75  # Maintain aspect ratio

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Prepare data
    methods = list(kappa_results.keys())
    linear_kappas = [results['linear_kappa'] for results in kappa_results.values()]
    quadratic_kappas = [results['quadratic_kappa'] for results in kappa_results.values()]

    # Plot settings
    bar_width = 0.35
    opacity = 0.8
    colors = ['#2ecc71', '#3498db']

    # Create bars
    x = np.arange(len(methods))
    rects1 = ax.bar(x - bar_width / 2, linear_kappas, bar_width,
                    color=colors[0], alpha=opacity, label='Linear Weighted',
                    edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + bar_width / 2, quadratic_kappas, bar_width,
                    color=colors[1], alpha=opacity, label='Quadratic Weighted',
                    edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_ylabel('Kappa Score')
    ax.set_xlabel('Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    # Add value labels
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7)

    add_value_labels(rects1)
    add_value_labels(rects2)

    # Grid and legend
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    # Tight layout
    plt.tight_layout()

    # Save in different formats
    # TIFF for print
    plt.savefig(f'{save_path}.tiff', dpi=300, bbox_inches='tight', format='tiff')
    # PNG for presentations/web
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', format='png')
    # EPS/PDF for vector graphics
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', format='pdf')

    plt.close()


# Use the function
create_publication_kappa_plot(kappa_results, save_path='kappa_comparison_publication', single_column=True)