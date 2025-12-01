# INJURY_COSTS.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# (1) Maps MAIS-based injury costs onto the ISS scale (econometrics, logistic regression, bounding analysis, probabilistic analysis)
# (2) Develops reduced-form ISS cost models (OLS regression, log-log regression)
# (3) Clusters ranges of MAIS/ISS values according to various data features (incidence, cost, mortality, hospitalization) (k-means clustering)

import os
import re
import math
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.metrics import r2_score
from sympy import Eq, solve, symbols
from scipy.stats import chi2, norm, pearsonr
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel

time0 = time.time()

#---------------#
#  USER INPUTS  #
#---------------#

num_init_centroids = 100        # total sets of inital random centroids (brute force methods) (k-means clustering) (int)

mais_num_clusters = [2,3]       # mais number of clusters to fit (list)
iss_num_clusters = [2,3,4,5,6]  # iss number of clusters to fit (list)

plots_for_publication = False   # True = conservative appearance / False = pretty
include_residual_plots = False  # whether to create the residual plots
export_plots = True             # both pdf and jpg export
include_fig1_in_pdf = False     # does not present any new information, so optional

#-----------#
#  PREABLE  #
#-----------#

np.random.seed(1099)
warnings.filterwarnings('ignore')

resid_plot_num = 0  # initalize to zero

assert min(mais_num_clusters) > 0 and min(iss_num_clusters) > 0

for var in [plots_for_publication, include_residual_plots, export_plots, include_fig1_in_pdf]:
    assert type(var) == bool 

# LISTS

hosp_dist_vars = ['dist_Fink_etal_2006', 'dist_wisqars_2025']
hosp_cost_vars = ['cost_Fink_etal_2006_qol', 'cost_wisqars_2025']

mais_hosp_vars = ['hosp_rate_Blin_etal_2023']
mais_dist_vars = ['dist_Copes_etal_1990', 'dist_Fink_etal_2006', 'dist_Blin_etal_2023']
mais_pdeath_vars = ['pdeath_Copes_etal_1990', 'pdeath_Genn_etal_1994', 'pdeath_Genn_Wod_2006']

mais_cost_vars = ['cost_Grah_etal_1997', 'cost_Fink_etal_2006_qol', 'cost_DOT_2021', 'cost_Blin_etal_2023']
mais_plot_cost_vars = ['cost_Grah_etal_1997', 'cost_DOT_2021', 'cost_Blin_etal_2023', 'cost_Fink_etal_2006_qol']

mais_noncost_vars = mais_dist_vars + mais_hosp_vars + mais_pdeath_vars
mais_cluster_vars = mais_noncost_vars + mais_cost_vars

mais_shares_list = ['mais' + str(c+1) + '_share' for c in range(6)]
mais_clusters_cols = ['cluster' + str(c+1) for c in range(max(mais_num_clusters))]
iss_clusters_cols = ['cluster' + str(c+1) for c in range(max(iss_num_clusters))]

iss_main_source = 'kilgo_etal_2004'
iss_source_names = ['copes_etal_1988'] + [iss_main_source]

iss_dist_vars = ['dist_' + item for item in iss_source_names]
iss_pdeath_vars = ['pdeath_' + item for item in iss_source_names]

iss_cluster_vars = iss_dist_vars + iss_pdeath_vars + mais_cost_vars

# DICTIONARIES

mais_clusters_dict = {c:'cluster' + str(c+1) for c in range(max(mais_num_clusters))}
iss_clusters_dict = {c:'cluster' + str(c+1) for c in range(max(iss_num_clusters))}

name_dict = {'cost_Grah_etal_1997':'Graham et al. (1997)',
             'cost_Fink_etal_2006_qol':'Finkelstein et al. (2006) + quality-of-life',
             'cost_DOT_2021':'U.S. Dept. of Transportation (2021, 2025)',
             'cost_Blin_etal_2023':'Blincoe et al. (2023)'}

color_dict = {'cost_Grah_etal_1997':'blue', 'cost_Fink_etal_2006_qol':'red', 'cost_DOT_2021':'darkorange', 'cost_Blin_etal_2023':'black'}

linestyle_dict = {'cost_Grah_etal_1997':'dashdot', 'cost_Fink_etal_2006_qol':'dashed', 'cost_DOT_2021':'solid', 'cost_Blin_etal_2023':'dotted'}

#-------------#
#  FUNCTIONS  #
#-------------#

def sig_figs(x, sig_figs):  # significant digit formatting
    assert sig_figs > 0, 'ERROR: number of significant digits is out-of-range'
    if x == 0:
        fmt = '{:.' + str(sig_figs - 1) + 'f}'
        return fmt.format(x)
    order_of_mag = math.floor(math.log10(abs(x)))
    round_digits = max(sig_figs - 1 - order_of_mag, 0)
    fmt = '{:,.' + str(round_digits) + 'f}'
    ratio = round(x / 10**order_of_mag, sig_figs - 1)
    trimmed_val = ratio * (10**order_of_mag)
    return fmt.format(trimmed_val)

def cost_format(df):  # currency formatting (dollar sign, commas)
    df_in = pd.DataFrame(df.copy(deep=True))
    df_out = pd.DataFrame()
    for var in df_in.columns:
        df_out[var] = df_in[var].apply(lambda x: x if str(x) in ['nan', ''] else '$0' if x == 0 else '$' + sig_figs(x,3))
    return df_out

def pct_format(df, digits):  # percent formatting (percent sign, zeros)
    assert digits >= 0, 'ERROR: number of digits is out-of-range'    
    df_in = pd.DataFrame(df.copy(deep=True))
    df_out = pd.DataFrame()
    fmt = '{:.' + str(digits) + 'f}'
    for col in df_in.columns:
        df_out[col] = round(100*df_in[col], digits)
        df_out[col] = df_out[col].apply(lambda x: '0%' if x == 0 else '100%' if x == 100 else str(fmt.format(x)) + '%')
    return df_out

def dash_format(df):  # infill missing/nan
    df_in = df.copy(deep=True)
    df_out = pd.DataFrame()
    for col in df_in.columns:
        df_out[col] = df_in[col].apply(lambda x: '-' if str(x) in ['nan', '0', '0%', ''] else str(x))
    return df_out

def pvalue_format(x):  # p-value formatting
    digits = 3
    fmt = '{:.' + str(digits) + 'f}'
    thres_low = 10**(-digits)
    thres_high = 1 - thres_low
    return 'p<' + str(thres_low) if round(x, digits) == 0 else 'p>' + str(thres_high) if round(x, digits) == 1 else 'p=' + fmt.format(x)

def r_sqrd_format(x):  # R^2 formatting
    digits = 3
    fmt = '{:.' + str(digits) + 'f}'
    thres_low = 10**(-digits)
    thres_high = 1 - thres_low
    return '<' + str(thres_low) if round(x, digits) == 0 else '>' + str(thres_high) if round(x, digits) == 1 else fmt.format(x)

def resid_plots(X, Y, title):  # residual plots
    # linear fit
    resid_fit = sm.OLS(Y, sm.add_constant(X)).fit()  # add constant instructs the regression to fit an intercept
    const = resid_fit.params[0]
    slope = resid_fit.params[1]
    pred = (slope * X) + const
    # create plot
    fig = plt.figure()
    plt.scatter(X, Y, marker='*', color='blue')
    plt.plot(X, pred, linestyle='dashed', color='black')
    plt.title(title)
    plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
    return fig

def delete_file(title):  # delete file (if exists)
    file = os.getcwd() + '\\' + title
    if os.path.exists(file):
        os.remove(file)

#--------------#
#  INPUT DATA  #
#--------------#

iss = pd.read_csv('iss_summary.csv')
mais = pd.read_csv('mais_summary.csv')
hosp = pd.read_csv('hospitalized_summary.csv')
triplets = pd.read_csv('iss_mais_triplets.csv')

#--------------#
#  MAIS / ISS  #
#--------------#

# iss theoretical bounds (by mais)
mais['iss_min_theory'] = mais['mais']**2
mais['iss_max_theory'] = 3*mais['mais']**2
mais['iss_min_theory'] = [75 if mais['mais'][i] == 6 else mais['iss_min_theory'][i] for i in mais.index]  # mais 6 / iss 75
mais['iss_max_theory'] = [75 if mais['mais'][i] == 6 else mais['iss_max_theory'][i] for i in mais.index]  # mais 6 / iss 75

# clean mais data (incidence / mortality)
for dist, source in zip(iss_dist_vars, iss_source_names):
    iss['inj_' + source] = iss['total_' + source] - iss['deaths_' + source]
    iss['dist_' + source] = iss['inj_' + source] / sum(iss['inj_' + source])
    iss['pdeath_' + source] = iss['deaths_' + source] / iss['total_' + source]

# iss rows of interest
iss_15_vicinity = iss.loc[iss['iss'].isin([14,16])].reset_index()
iss_min_max_rows = iss.loc[iss['total_' + iss_main_source].isin([min(iss['total_' + iss_main_source]), max(iss['total_' + iss_main_source])])].reset_index()

# mais bounds - theoretical (by iss)
iss_mais = pd.DataFrame(iss['iss'])
iss_mais['mais_min_theory'] = iss_mais['iss'].apply(lambda x: np.ceil(np.sqrt(x/3)))
iss_mais['mais_max_theory'] = iss_mais['iss'].apply(lambda x: np.floor(np.sqrt(x)))
iss_mais['mais_max_theory'] = iss_mais['mais_max_theory'].apply(lambda x: 5 if x > 5 else x)  # capped at mais 5
iss_mais['mais_max_theory'] = [6 if iss_mais['iss'][i] == 75 else iss_mais['mais_max_theory'][i] for i in iss_mais.index]  # mais 6 / iss 75
iss_mais['mais_theory'] = [list(np.arange(int(iss_mais['mais_min_theory'][i]), int(iss_mais['mais_max_theory'][i] + 1))) for i in iss_mais.index]

# iss-mais combos - valid only (unique by row)
iss_mais = pd.merge(iss_mais, triplets, on='iss', how='inner')
iss_mais['triplet1'] = iss_mais['triplet_1'].apply(lambda x: str(x).replace('*',''))
iss_mais['triplet2'] = iss_mais['triplet_2'].apply(lambda x: str(x).replace('*',''))
iss_mais['triplets_cnt'] = [1 if str(iss_mais['triplet_2'][i]) == 'nan' else 2 for i in iss_mais.index]
iss_mais['total_cnt'] = iss_mais['count_1'] + iss_mais['count_2']
iss_mais['total_cnt_fmt'] = iss_mais['total_cnt'].apply(lambda x: '{:,.0f}'.format(x))
iss_mais['ais1'] = iss_mais['triplet1'].apply(lambda x: re.findall(r'\d+', x) if str(x) != 'nan' else [])
iss_mais['ais2'] = iss_mais['triplet2'].apply(lambda x: re.findall(r'\d+', x) if str(x) != 'nan' else [])
iss_mais['triplet2'] = iss_mais['triplet2'].apply(lambda x: '' if str(x) == 'nan' else x)
iss_mais['ais1'] = iss_mais['ais1'].apply(lambda x: [int(item) for item in x if item != '0'])
iss_mais['ais2'] = iss_mais['ais2'].apply(lambda x: [int(item) for item in x if item != '0'])
iss_mais['mais1'] = iss_mais['ais1'].apply(lambda x: max(x))
iss_mais['mais2'] = iss_mais['ais2'].apply(lambda x: '' if len(x) == 0 else max(x))
iss_mais['mais_valid'] = [[iss_mais['mais1'][i]] if iss_mais['triplets_cnt'][i] == 1 else [iss_mais['mais1'][i]] + [iss_mais['mais2'][i]] for i in iss_mais.index]
iss_mais['mais_valid'] = iss_mais['mais_valid'].apply(lambda x: list(dict.fromkeys(x)))  # removes duplicate mais (iss 50 has two triplets but only one mais)
iss_mais['total_mais'] = iss_mais['mais_valid'].apply(lambda x: len(x))
iss_mais['wgt1'] = [1 if iss_mais['triplets_cnt'][i] == 1 else iss_mais['count_1'][i] / iss_mais['total_cnt'][i] for i in iss_mais.index]
iss_mais['wgt2'] = [0 if iss_mais['triplets_cnt'][i] == 1 else iss_mais['count_2'][i] / iss_mais['total_cnt'][i] for i in iss_mais.index]
iss_mais['mais_avg'] = [iss_mais['mais1'][i] if iss_mais['triplets_cnt'][i] == 1 else iss_mais['wgt1'][i]*iss_mais['mais1'][i] + iss_mais['wgt2'][i]*int(iss_mais['mais2'][i]) for i in iss_mais.index]
iss_mais['ln_iss'] = np.log(iss_mais['iss'])
iss_mais['ln_mais_avg'] = np.log(iss_mais['mais_avg'])

# total body regions injured
iss_mais['regions1'] = iss_mais['triplet1'].apply(lambda x: len(tuple(int(item) for item in x if str(item) not in ['0' , ' ' , ',' , '(' , ')'])))
iss_mais['regions2'] = iss_mais['triplet2'].apply(lambda x: '' if len(x) == 0 else len(tuple(int(item) for item in x if str(item) not in ['0' , ' ' , ',' , '(' , ')'])))
iss_mais['regions_min'] = [min(iss_mais['regions1'][i], int(iss_mais['regions2'][i])) if iss_mais['triplets_cnt'][i] == 2 else iss_mais['regions1'][i] for i in iss_mais.index]
iss_mais['regions_max'] = [max(iss_mais['regions1'][i], int(iss_mais['regions2'][i])) if iss_mais['triplets_cnt'][i] == 2 else iss_mais['regions1'][i] for i in iss_mais.index]
iss_mais['regions_same'] = [1 if iss_mais['regions_min'][i] == int(iss_mais['regions_max'][i]) else 0 for i in iss_mais.index]
iss_mais['body_regions_cnt'] = [[int(iss_mais['regions_min'][i])] if iss_mais['regions_same'][i] == 1 else [int(iss_mais['regions_min'][i]), int(iss_mais['regions_max'][i])] for i in iss_mais.index]

# iss 75 is essentially mais 6 only (and its weights could be set accordingly)
# iss_mais.loc[iss_mais['iss'] == 75, 'wgt1'] = 0  
# iss_mais.loc[iss_mais['iss'] == 75, 'wgt2'] = 1

#------------------#
#  MAIS-ISS PAIRS  #
#------------------#

# theoretical
iss_theory = pd.DataFrame()
iss_theory[['iss', 'mais']] = iss_mais[['iss', 'mais_theory']]
iss_theory = iss_theory.explode('mais', ignore_index=True)
iss_theory['mais'] = iss_theory['mais'].apply(lambda x: int(x))

# valid only
iss_valid = pd.DataFrame(columns=['iss', 'mais'])
for i in iss_mais.index:
    iss_val = iss_mais['iss'][i]
    triplets_cnt = iss_mais['triplets_cnt'][i]
    mais1 = iss_mais['mais1'][i]
    mais2 = iss_mais['mais2'][i]
    if (triplets_cnt == 1) or (mais1 == mais2):
        iss_valid.loc[len(iss_valid)] = [iss_val, mais1]
    else:
        iss_valid.loc[len(iss_valid)] = [iss_val, mais1]
        iss_valid.loc[len(iss_valid)] = [iss_val, int(mais2)]

# ensure is unique by row
assert len(iss_theory) == len(iss_theory.drop_duplicates())
assert len(iss_valid) == len(iss_valid.drop_duplicates())

# iss range by valid mais
iss_mais_ranges = pd.DataFrame()
iss_mais_ranges['min_iss'] = iss_valid.groupby('mais')['iss'].min()
iss_mais_ranges['max_iss'] = iss_valid.groupby('mais')['iss'].max()

#---------------#
#  MAIS SHARES  #
#---------------#

mais_shares_fmt_list = [k + '_fmt' for k in mais_shares_list]

# mais probability shares (by iss)
shares = pd.DataFrame(iss_mais[['iss', 'mais1', 'mais2', 'wgt1', 'wgt2', 'mais_valid']])
for share in mais_shares_list:
    m = mais_shares_list.index(share) + 1
    shares[share] = 0
    for i in shares.index:
        if m == shares['mais1'][i] == shares['mais2'][i]:
            shares[share][i] = shares['wgt1'][i] + shares['wgt2'][i]
        else:
            shares[share][i] = shares['wgt1'][i] if shares['mais1'][i] == m else shares['wgt2'][i] if shares['mais2'][i] == m else 0

# ensure shares sum to one (at level of precision of final table)
shares['pdf_sum'] = 0
for share in mais_shares_list:
    shares[share + '_fmt'] = 0
    shares[share + '_fmt'] = round(shares[share], 4)
    shares['pdf_sum'] += shares[share + '_fmt']
shares['pdf_sum'] = round(shares['pdf_sum'], 4)
for i in iss_mais.index:
    assert math.isclose(round(iss_mais['wgt1'][i], 2) + round(iss_mais['wgt2'][i], 2), 1), 'ERROR: shares do not sum to one'
for i in shares.index:
    assert math.isclose(shares['pdf_sum'][i], 1), 'ERROR: shares do not sum to one'

#----------#
#  COUNTS  #
#----------#

n_iss_values = len(iss)
n_mais_iss_valid = len(iss_valid)
n_mais_iss_theory = len(iss_theory)
n_iss_mult_triplets = len(iss_mais[iss_mais['triplets_cnt'] == 2])
n_iss_one_mais_valid = len(iss_mais[iss_mais['total_mais'] == 1])
n_iss_mult_mais_valid = len(iss_mais[iss_mais['total_mais'] == 2])

#-------------#
#  ISS COSTS  #
#-------------#

iss_costs = pd.DataFrame(iss['iss'])
for cost in mais_cost_vars:
    iss_costs[cost] = 0
    for m in range(6):  # loop needed because shares are rowwise
        iss_costs[cost] += mais[cost][m] * shares['mais' + str(m+1) + '_share']  # maps mais costs to iss scale

#------------------------#
#  LOGISTIC REGRESSIONS  #
#------------------------#

iss_vals_excluded = [1,2,3,75]  # iss 1-3 associated with mais 1 only (and mais 1 with iss 1-3 only) / iss 75 is essentially mais 6
iss_vals_included = [item for item in iss['iss'].unique() if item not in iss_vals_excluded]

# TRAINING DATA (mais/iss combos)

# theoretical
train1 = iss_theory[['iss', 'mais']].copy(deep=True)
train1 = train1[train1['iss'].isin(iss_vals_included)]
X_logreg_1 = train1['iss']
Y_logreg_1 = train1['mais']
n_logreg_1 = len(Y_logreg_1)

# valid only
train2 = iss_valid[['iss', 'mais']].copy(deep=True)
train2 = train2[train2['iss'].isin(iss_vals_included)]
X_logreg_2 = train2['iss']
Y_logreg_2 = train2['mais']
n_logreg_2 = len(Y_logreg_2)

# ORDINAL LOGISTIC (mais ~ iss)

olr_model_1 = OrderedModel(Y_logreg_1, X_logreg_1, distr='logit')
olr_model_2 = OrderedModel(Y_logreg_2, X_logreg_2, distr='logit')
olr_fit_1 = olr_model_1.fit(method='bfgs')
olr_fit_2 = olr_model_2.fit(method='bfgs')

print(Fore.GREEN + '\033[1m' + '\n\nORDINAL REGRESSION - theoretical (mais ~ iss)\n' + Style.RESET_ALL)
print(olr_fit_1.summary())
print('\n')
print(Fore.GREEN + '\033[1m' + 'ORDINAL REGRESSION - valid only (mais ~ iss)\n' + Style.RESET_ALL)
print(olr_fit_2.summary())
print('\n')

# MULTINOMIAL LOGISTIC (mais ~ iss)

mnl_model_1 = MNLogit(Y_logreg_1, sm.add_constant(X_logreg_1))  # add constant instructs the regression to fit an intercept
mnl_model_2 = MNLogit(Y_logreg_2, sm.add_constant(X_logreg_2))
mnl_fit_1 = mnl_model_1.fit(method='bfgs')
mnl_fit_2 = mnl_model_2.fit(method='bfgs')

print(Fore.GREEN + '\033[1m' + '\n\nMULTINOMIAL REGRESSION - theoretical (mais ~ iss)\n' + Style.RESET_ALL)
print(mnl_fit_1.summary())
print('\n')
print(Fore.GREEN + '\033[1m' + 'MULTINOMIAL REGRESSION - valid only (mais ~ iss)\n' + Style.RESET_ALL)
print(mnl_fit_2.summary())
print('\n')

# COEFFICIENTS / P-VALUES

mnl_results = pd.DataFrame()
mnl_results['model_type'] = 3*['theoretical'] + 3*['valid only']
mnl_results['level'] = 2*['mais 3', 'mais 4', 'mais 5']
mnl_results['x_var'] = 6*['iss']
mnl_results['n'] = 3*[n_logreg_1] + 3*[n_logreg_2]
mnl_results['const'] = [k for k in np.transpose(mnl_fit_1.params)['const']] + [k for k in np.transpose(mnl_fit_2.params)['const']]
mnl_results['slope'] = [k for k in np.transpose(mnl_fit_1.params)['iss']] + [k for k in np.transpose(mnl_fit_2.params)['iss']]
mnl_results['pvalue_const'] = [k for k in np.transpose(mnl_fit_1.pvalues)['const']] + [k for k in np.transpose(mnl_fit_2.pvalues)['const']]
mnl_results['pvalue_slope'] = [k for k in np.transpose(mnl_fit_1.pvalues)['iss']] + [k for k in np.transpose(mnl_fit_2.pvalues)['iss']]

mnl_results['const_fmt'] = mnl_results['const'].apply(lambda x: sig_figs(x,3))
mnl_results['slope_fmt'] = mnl_results['slope'].apply(lambda x: sig_figs(x,3))
mnl_results['pvalue_const_fmt'] = mnl_results['pvalue_const'].apply(lambda x: pvalue_format(x))
mnl_results['pvalue_slope_fmt'] = mnl_results['pvalue_slope'].apply(lambda x: pvalue_format(x))
mnl_results['const_full'] = [mnl_results['const_fmt'][i] + ' (' + mnl_results['pvalue_const_fmt'][i] + ')' for i in mnl_results.index]
mnl_results['slope_full'] = [mnl_results['slope_fmt'][i] + ' (' + mnl_results['pvalue_slope_fmt'][i] + ')' for i in mnl_results.index]

#-------------------------#
#  LIKELIHOOD RATIO TEST  #
#-------------------------#

# log-likelihoods
loglik_olr_1 = olr_fit_1.llf
loglik_olr_2 = olr_fit_2.llf
loglik_mnl_1 = mnl_fit_1.llf
loglik_mnl_2 = mnl_fit_2.llf

# format
loglik_olr_1_fmt = sig_figs(loglik_olr_1, 3)
loglik_olr_2_fmt = sig_figs(loglik_olr_2, 3)
loglik_mnl_1_fmt = sig_figs(loglik_mnl_1, 3)
loglik_mnl_2_fmt = sig_figs(loglik_mnl_2, 3)

# check parameter counts
assert len(olr_fit_1.params) == len(olr_fit_2.params)
assert len(mnl_fit_1.params) == len(mnl_fit_2.params) and len(mnl_fit_1.params.columns) == len(mnl_fit_2.params.columns)
assert len(mnl_fit_1.params.columns) == len(mnl_fit_2.params.columns)

# HYPOTHESIS TESTING
# chi-squared test (p < α -> reject H0)
# HO (null): simpler model is sufficient
# H1 (alt.): more complex model has significantly better fit

# degrees of freedom
param_cnt_olr = len(olr_fit_1.params)
param_cnt_mnl = len(mnl_fit_1.params) * len(mnl_fit_1.params.columns)
dof = param_cnt_mnl - param_cnt_olr

# chi-squared stats
lrt_stat_1 = 2*(loglik_mnl_1 - loglik_olr_1)
lrt_stat_2 = 2*(loglik_mnl_2 - loglik_olr_2)
lrt_pvalue_1 = chi2.sf(lrt_stat_1, dof)
lrt_pvalue_2 = chi2.sf(lrt_stat_2, dof)

# format
lrt_stat_1_fmt = sig_figs(lrt_stat_1, 3)
lrt_stat_2_fmt = sig_figs(lrt_stat_2, 3)
lrt_pvalue_1_fmt = sig_figs(lrt_pvalue_1, 3)
lrt_pvalue_2_fmt = sig_figs(lrt_pvalue_2, 3)

# finalize
lrt = pd.DataFrame()
lrt['model_type'] = ['theoretical', 'valid only']
lrt['y_var'] = 2*['mais']
lrt['x_var'] = 2*['iss']
lrt['n'] = [n_logreg_1, n_logreg_2]
lrt['stat'] = [lrt_stat_1, lrt_stat_2]
lrt['pvalue'] = [lrt_pvalue_1, lrt_pvalue_2]
lrt['stat_fmt'] = lrt['stat'].apply(lambda x: sig_figs(x,3))
lrt['pvalue_fmt'] = lrt['pvalue'].apply(lambda x: pvalue_format(x))
lrt['chi_sqrd_stat'] = ['χ2 = ' + lrt['stat_fmt'][i] for i in lrt.index]
lrt['pvalue_chi_sqrd'] = ['(' + lrt['pvalue_fmt'][i] + ', dof=' + str(dof) + ')' for i in lrt.index]
lrt['chi_sqrd_fmt'] = [lrt['chi_sqrd_stat'][i] + '\n' + lrt['pvalue_chi_sqrd'][i] for i in lrt.index]

chi_sqrd_1_fmt = lrt['chi_sqrd_fmt'][0]
chi_sqrd_2_fmt = lrt['chi_sqrd_fmt'][1]

#-------------------#
#  OLS REGRESSIONS  #
#-------------------#

cost_cols = ['model_type', 'y_var', 'x_var', 'n', 'const', 'pvalue_const', 'slope', 'pvalue_slope', 'r_sqrd', 'norm_corr']

ols_pwr = pd.DataFrame(columns=cost_cols)
cost_funcs = pd.DataFrame(columns=cost_cols)
reduced_form = pd.DataFrame(columns=cost_cols)

# POWER FUNCTIONS (ln-MAIS-avg ~ ln-ISS)

# SCENARIO 1 - UNCONSTRAINED

X_pwr = iss_mais['ln_iss']  # log-log model
Y_pwr = iss_mais['ln_mais_avg']

model_pwr_1 = sm.OLS(Y_pwr, sm.add_constant(X_pwr))  # add constant instructs the regression to fit an intercept
fit_pwr_1 = model_pwr_1.fit()

n_ols_pwr = len(Y_pwr)

const_pwr_1 = fit_pwr_1.params[0]
slope_pwr_1 = fit_pwr_1.params[1]

pvalue_const_pwr_1 = fit_pwr_1.pvalues[0]
pvalue_slope_pwr_1 = fit_pwr_1.pvalues[1]

r_sqrd_pwr_1 = fit_pwr_1.rsquared
Y_pred_pwr_1 = fit_pwr_1.predict()

# print summary
print(Fore.GREEN + '\033[1m' + 'OLS - power function [ln(mais-avg) ~ ln(iss)]\n' + Style.RESET_ALL)
print(fit_pwr_1.summary())
print('\n')

# check predictions
pred_errors_pwr_1 = []
for ln_iss, pred_ln_mais in zip(X_pwr, Y_pred_pwr_1):
    if np.exp(pred_ln_mais) < 1 or np.exp(pred_ln_mais) > 6:  # log-log model
        pred_errors_pwr_1.append(round(np.exp(ln_iss)))
if len(pred_errors_pwr_1) > 0:
    print(Fore.GREEN + '\033[1m' + 'NOTE: power function predictions out-of-range [ln(mais-avg) ~ ln(iss)]')
    print('for ISS:')
    print(pred_errors_pwr_1)
    print('\n' + Style.RESET_ALL)

# residuals
resids_pwr_1 = Y_pwr - Y_pred_pwr_1
resids_sorted_pwr_1 = resids_pwr_1.sort_values()
ecdf_resids_pwr_1 = [(k+1)/len(Y_pwr) for k in range(len(Y_pwr))]  # empirical cdf
norm_cdf_resids_pwr_1 = norm.cdf(resids_sorted_pwr_1, loc=np.mean(resids_pwr_1), scale=np.std(resids_pwr_1))  # normal cdf
norm_corr_pwr_1 = pearsonr(ecdf_resids_pwr_1, norm_cdf_resids_pwr_1)[0]  # normality of residuals (cumulative p-p plot correlation)
assert max(ecdf_resids_pwr_1) == 1

# residual plots
if include_residual_plots:
    resid_plot_num += 1
    fig_norm = resid_plots(norm_cdf_resids_pwr_1, ecdf_resids_pwr_1, 'Normality [ln(mais-avg) ~ ln(iss)]')
    fig_var = resid_plots(X_pwr, resids_pwr_1, 'Variance [ln(mais-avg) ~ ln(iss)]')
    exec('rnorm' + str(resid_plot_num) + ' = fig_norm')
    exec('rvar' + str(resid_plot_num) + ' = fig_var')

textcols1 = ['link (power function)', 'ln(mais-avg)', 'ln(iss)']
ols_pwr.loc[len(ols_pwr)] = textcols1 + [n_ols_pwr, const_pwr_1, pvalue_const_pwr_1, slope_pwr_1, pvalue_slope_pwr_1, r_sqrd_pwr_1, norm_corr_pwr_1]

# SCENARIO 2 - CONSTRAINED

# determine parameters (mais 1 @ iss 1 / mais 6 @ iss 75)
m, b = symbols('m b')
eq1 = Eq(np.log(1) - (m * np.log(1) + b), 0)  # log-log model
eq2 = Eq(np.log(6) - (m * np.log(75) + b), 0)
sol_dict = solve((eq1, eq2), (m, b))  # 2 equations / 2 unknowns
const_pwr_2 = float(sol_dict[b])
slope_pwr_2 = float(sol_dict[m])
Y_pred_pwr_2 = (slope_pwr_2 * X_pwr) + const_pwr_2
r_sqrd_pwr_2 = r2_score(Y_pwr, Y_pred_pwr_2)

textcols2 = ['link (power function / constrained)', 'ln(mais-avg)', 'ln(iss)']
ols_pwr.loc[len(ols_pwr)] = textcols2 + [n_ols_pwr, const_pwr_2, np.nan, slope_pwr_2, np.nan, r_sqrd_pwr_2, np.nan]

# finalize
ols_pwr['const_fmt'] = ols_pwr['const'].apply(lambda x: sig_figs(x,3))
ols_pwr['slope_fmt'] = ols_pwr['slope'].apply(lambda x: sig_figs(x,3))
ols_pwr['pvalue_const_fmt'] = ols_pwr['pvalue_const'].apply(lambda x: '' if str(x) == 'nan' else ' (' + pvalue_format(x) + ')')
ols_pwr['pvalue_slope_fmt'] = ols_pwr['pvalue_slope'].apply(lambda x: '' if str(x) == 'nan' else ' (' + pvalue_format(x) + ')')
ols_pwr['const_full'] = [ols_pwr['const_fmt'][i] + ols_pwr['pvalue_const_fmt'][i] for i in ols_pwr.index]
ols_pwr['slope_full'] = [ols_pwr['slope_fmt'][i] + ols_pwr['pvalue_slope_fmt'][i] for i in ols_pwr.index]
ols_pwr['r_sqrd_fmt'] = ols_pwr['r_sqrd'].apply(lambda x: r_sqrd_format(x))
ols_pwr['norm_corr_fmt'] = ols_pwr['norm_corr'].apply(lambda x: r_sqrd_format(x))

# COST FUNCTIONS (cost ~ iss)

# statistical models
for scenario in ['regular']:  # 'loglog', 'regular'
    for cost in mais_cost_vars:
        func_form = 'cost ~ iss' if (scenario == 'regular') else 'ln(cost+1) ~ ln(iss)'
        model_name = 'cost' if (scenario == 'regular') else 'cost (power function)'

        X_cost = iss_costs['iss']
        Y_cost = iss_costs[cost] / 10**6  # cost (in millions)
        
        if scenario == 'loglog':
            X_cost = np.log(X_cost)
            Y_cost = np.log(Y_cost + 1)  # plus one allows for zero cost values
        
        model_cost = sm.OLS(Y_cost, sm.add_constant(X_cost))  # add constant instructs the regression to fit an intercept
        fit_cost = model_cost.fit() 
    
        const_cost = fit_cost.params[0]
        slope_cost = fit_cost.params[1]
        
        pvalue_const_cost = fit_cost.pvalues[0]
        pvalue_slope_cost = fit_cost.pvalues[1]
        
        r_sqrd_cost = fit_cost.rsquared
        Y_pred_cost = fit_cost.predict()
    
        print(Fore.GREEN + '\033[1m' + 'OLS - ' + model_name + ' (' + func_form + ') - ' + cost + Style.RESET_ALL)
        print(fit_cost.summary())
        print('\n')

        # residuals
        resids_cost = Y_cost - Y_pred_cost
        resids_sorted_cost = resids_cost.sort_values()
        ecdf_resids_cost = [(k+1)/len(Y_cost) for k in range(len(Y_cost))]  # empirical cdf
        norm_cdf_resids_cost = norm.cdf(resids_sorted_cost, loc=np.mean(resids_cost), scale=np.std(resids_cost))  # normal cdf
        norm_corr_cost = pearsonr(ecdf_resids_cost, norm_cdf_resids_cost)[0]  # normality of residuals (cumulative p-p plot correlation)
        assert max(ecdf_resids_cost) == 1
                
        # check predictions
        pred_errors_cost = []
        for iss_val, pred_cost in zip(X_cost, Y_pred_cost):
            if pred_cost < 0:
                if scenario == 'regular':
                    pred_errors_cost.append(round(iss_val))
                elif scenario == 'loglog':
                    pred_errors_cost.append(round(np.exp(iss_val)))
        if len(pred_errors_cost) > 0:
            print(Fore.GREEN + '\033[1m' + 'NOTE: cost predictions out-of-range (' + func_form + ') - ' + cost)
            print('for ISS:')
            print(pred_errors_cost)
            print('\n' + Style.RESET_ALL)
    
        # residual plots
        if include_residual_plots:
            resid_plot_num += 1
            fig_norm = resid_plots(norm_cdf_resids_cost, ecdf_resids_cost, 'Normality (' + func_form + ' | ' + cost + ')')
            fig_var = resid_plots(X_cost, resids_cost, 'Variance (' + func_form + ' | ' + cost + ')')
            exec('rnorm' + str(resid_plot_num) + ' = fig_norm')
            exec('rvar' + str(resid_plot_num) + ' = fig_var')
        
        yname = cost if (scenario == 'regular') else 'ln(' + cost + ' + 1)'
        xname = 'iss' if (scenario == 'regular') else 'ln(iss)'
        
        textcols3 = [model_name, yname, xname]
        cost_funcs.loc[len(cost_funcs)] = textcols3 + [len(Y_cost), const_cost, pvalue_const_cost, slope_cost, pvalue_slope_cost, r_sqrd_cost, norm_corr_cost]

# finalize
cost_funcs['const_fmt'] = cost_funcs['const'].apply(lambda x: sig_figs(x,3))
cost_funcs['slope_fmt'] = cost_funcs['slope'].apply(lambda x: sig_figs(x,3))
cost_funcs['pvalue_const_fmt'] = cost_funcs['pvalue_const'].apply(lambda x: pvalue_format(x))
cost_funcs['pvalue_slope_fmt'] = cost_funcs['pvalue_slope'].apply(lambda x: pvalue_format(x))
cost_funcs['const_full'] = [cost_funcs['const_fmt'][i] + ' (' + cost_funcs['pvalue_const_fmt'][i] + ')' for i in cost_funcs.index]
cost_funcs['slope_full'] = [cost_funcs['slope_fmt'][i] + ' (' + cost_funcs['pvalue_slope_fmt'][i] + ')' for i in cost_funcs.index]
cost_funcs['r_sqrd_fmt'] = cost_funcs['r_sqrd'].apply(lambda x: r_sqrd_format(x))
cost_funcs['norm_corr_fmt'] = cost_funcs['norm_corr'].apply(lambda x: r_sqrd_format(x))

# REDUCED-FORM (cost ~ iss)

X_rf = iss_costs['iss']

# determine parameters
for scenario in [1,2]:
    for cost in mais_cost_vars:
        Y_rf = iss_costs[cost] / 10**6  # cost (in millions)
        # scenario 1 - zero dollars @ iss 0 / mais 6 cost @ iss 75
        if scenario == 1:
            model_type = 'cost (constrained / iss=0)'
            const_rf = 0
            slope_rf = (mais[cost][6-1] / 10**6) / 75
        # scenario 2 - mais 1 cost @ iss 1 / mais 6 cost @ iss 75
        elif scenario == 2:
            model_type = 'cost (constrained / iss=1)'
            m, b = symbols('m b')
            eq1 = Eq(mais[cost][0] / 10**6 - (m*1 + b), 0)
            eq2 = Eq(mais[cost][6-1] / 10**6 - (m*75 + b), 0)
            sol_dict = solve((eq1, eq2), (m, b))  # 2 equations / 2 unknowns
            const_rf = float(sol_dict[b])
            slope_rf = float(sol_dict[m])
        # evaluate and save
        Y_pred_rf = (slope_rf * X_rf) + const_rf
        r_sqrd_rf = r2_score(Y_rf, Y_pred_rf)
        reduced_form.loc[len(reduced_form)] = [model_type, cost, 'iss', len(Y_rf), const_rf, np.nan, slope_rf, np.nan, r_sqrd_rf, np.nan]

# finalize (pvalues and residuals analysis not given, because these models are not statistically fit)
reduced_form['const_fmt'] = reduced_form['const'].apply(lambda x: sig_figs(x,3))
reduced_form['slope_fmt'] = reduced_form['slope'].apply(lambda x: sig_figs(x,3))
reduced_form['pvalue_const_fmt'] = len(reduced_form) * ['']
reduced_form['pvalue_slope_fmt'] = len(reduced_form) * ['']
reduced_form['const_full'] = [reduced_form['const_fmt'][i] for i in reduced_form.index]
reduced_form['slope_full'] = [reduced_form['slope_fmt'][i] for i in reduced_form.index]
reduced_form['r_sqrd_fmt'] = reduced_form['r_sqrd'].apply(lambda x: r_sqrd_format(x))
reduced_form['norm_corr_fmt'] = np.nan

#-----------------#
#  COST AVERAGES  #
#-----------------#

# hospitalized
hosp_avg_costs = pd.DataFrame()
hosp_avg_costs['scale'] = ['hosp'] * len(hosp_dist_vars)
hosp_avg_costs['incidence'] = hosp_dist_vars
for cost in hosp_cost_vars:
    hosp_avg_costs[cost] = [sum(hosp[dist] * hosp[cost]) for dist in hosp_dist_vars]

# mais
mais_avg_costs = pd.DataFrame()
mais_avg_costs['scale'] = ['mais'] * len(mais_dist_vars)
mais_avg_costs['incidence'] = mais_dist_vars
for cost in mais_cost_vars:
    mais_avg_costs[cost] = [sum(mais[dist] * mais[cost]) for dist in mais_dist_vars]

# iss
iss_avg_costs = pd.DataFrame()
iss_avg_costs['scale'] = ['iss'] * len(iss_dist_vars)
iss_avg_costs['incidence'] = iss_dist_vars
for cost in mais_cost_vars:
    iss_avg_costs[cost] = [sum(iss[dist] * iss_costs[cost]) for dist in iss_dist_vars]

#---------------#
#  SCALE MEANS  #
#---------------#

scale_avg_cols = ['scale', 'incidence', 'avg']

hosp_avgs = pd.DataFrame(columns=scale_avg_cols)
mais_avgs = pd.DataFrame(columns=scale_avg_cols)
iss_avgs = pd.DataFrame(columns=scale_avg_cols)

# hospitalized / non-hospitalized
for source in hosp_dist_vars:
    hosp_avgs.loc[len(hosp_avgs)] = ['hosp', source, hosp[source][1]]  # average severity here is percent hospitalized
hosp_avgs['mais_est_1'] = ''
hosp_avgs['mais_est_2'] = ''
hosp_avgs['iss_est_1'] = ''
hosp_avgs['iss_est_2'] = ''
hosp_avgs['avg'] = pct_format(hosp_avgs['avg'], 1)

# mais
for source in mais_dist_vars:
    mais_avgs.loc[len(mais_avgs)] = ['mais', source, sum(mais[source] * mais['mais'])]
mais_avgs['mais_est_1'] = ''
mais_avgs['mais_est_2'] = ''
mais_avgs['iss_est_1'] = mais_avgs['avg'].apply(lambda x: np.exp((np.log(x) - const_pwr_1) / slope_pwr_1))  # log-log model
mais_avgs['iss_est_2'] = mais_avgs['avg'].apply(lambda x: np.exp((np.log(x) - const_pwr_2) / slope_pwr_2))

# iss
for source in iss_source_names:
    iss_avgs.loc[len(iss_avgs)] = ['iss', source, sum(iss['dist_' + source] * iss['iss'])]
iss_avgs['mais_est_1'] = iss_avgs['avg'].apply(lambda x: np.exp((slope_pwr_1 * np.log(x)) + const_pwr_1))  # log-log model
iss_avgs['mais_est_2'] = iss_avgs['avg'].apply(lambda x: np.exp((slope_pwr_2 * np.log(x)) + const_pwr_2))
iss_avgs['iss_est_1'] = ''
iss_avgs['iss_est_2'] = ''

# bring together
scale_avgs = pd.concat([hosp_avgs, mais_avgs, iss_avgs], axis=0).reset_index(drop=True)
scale_avgs['avg_fmt'] = [scale_avgs['avg'][i] if scale_avgs['scale'][i] == 'hosp' else sig_figs(scale_avgs['avg'][i], 3) for i in scale_avgs.index]
for est in ['mais_est_1', 'mais_est_2', 'iss_est_1', 'iss_est_2']:
    scale_avgs[est + '_num'] = scale_avgs[est].apply(lambda x: x if x == '' else sig_figs(x,3))
    scale_avgs[est + '_fmt'] = ['' if scale_avgs['scale'][i] == 'hosp' else scale_avgs[est + '_num'][i] for i in scale_avgs.index]

#---------------------#
#  CLUSTER FUNCTIONS  #
#---------------------#

# assign clusters (using current boundaries)
def assess_clusters(df_in, scale_var, cluster_var, num_clusters, boundaries):
    df = pd.DataFrame()
    df[[scale_var, cluster_var]] = df_in[[scale_var, cluster_var]]
    df['lock'] = 0  # locks values (so they wont change)
    # cluster assignments
    for c in range(num_clusters-1):
        df.loc[(df[scale_var] < boundaries[c]) & (df['lock'] == 0), 'cluster'] = c+1  # locate and set
        df.loc[df['cluster'] == c+1, 'lock'] = 1  # then lock
    df['lock'] = 1
    df['cluster'] = df['cluster'].apply(lambda x: x if str(x) != 'nan' else num_clusters)  # corresponds to last cluster group 
    # group-level averages
    df['grp_avg'] = df.groupby('cluster')[cluster_var].transform('mean')  # merge on cluster centroids (means)
    df['dx'] = df[cluster_var] - df['grp_avg']  # distance to cluster centroid
    df['dx_sqrd'] = df['dx']**2  # variance in position (about centroids)
    # finalize
    wcss = sum(df['dx_sqrd'])  # within-cluster sum of squares
    clusters = list(df['cluster'])  # final cluster assignments
    return wcss, clusters

# k-means clustering routine
def kmeans(df_in, scale_var, cluster_var, num_clusters):
    # initialize
    boundary_options = mais_boundary_options if (scale_var == 'mais') else iss_boundary_options  # available cluster boundaries to choose from
    scale_max = 6 if (scale_var == 'mais') else 75
    wcss_best = np.inf  # initialize to be worst possible value (b/c minimum is sought)
    
    # OPTION 1 - loop over all possibilities (brute force)
    if (scale_var == 'mais') or (num_clusters <= 4):
        combos = list(itertools.combinations(boundary_options, num_clusters-1))  # enumerates all possible combinations
        for boundaries in combos:
            wcss, clusters = assess_clusters(df_in, scale_var, cluster_var, num_clusters, boundaries)
            if wcss < wcss_best:  # compare to running global best solution
                wcss_best = wcss    
                clusters_selected = clusters

    # OPTION 2 - probabilistic selection (random-iterative)
    else:
        for k in range(num_init_centroids):
            # choose set of initial random centroids
            boundaries = list(np.random.choice(boundary_options, size=num_clusters-1, replace=False))  # choose WITHOUT replacement (no repeated boundaries)
            boundaries.sort()  # ensure they are increasing
            # optimize cluster boundaries for this set of centroids
            for c in range(num_clusters-1):
                min_thres = 1 if (c == 0) else boundaries[c-1]
                max_thres = scale_max if (c == num_clusters - 2) else boundaries[c+1]
                thresholds = [thres for thres in boundary_options if min_thres < thres < max_thres]  # all potential boundaries are MIDWAY between adjacent mais/iss
                # optimize each boundary location (1D parametric variation)
                for thres in thresholds:
                    wcss, clusters = assess_clusters(df_in, scale_var, cluster_var, num_clusters, boundaries)
                    if wcss < wcss_best:  # compare to running global best solution
                        wcss_best = wcss
                        clusters_selected = clusters
                        boundaries[c] = thres  # update cluster boundary

    return clusters_selected

# cluster ranges (mais/iss values)
def cluster_ranges(df_in, scale_var, cluster_vars, num_clusters, clusters_dict):
    index_vals = list(np.arange(0, max(num_clusters)))
    cluster_ranges = pd.DataFrame(index=index_vals)
    scale_max = 6 if (scale_var == 'mais') else 75
    for c in num_clusters:
        for var in cluster_vars:
            name = scale_var + '_c' + str(c) + '_' + var
            min_name = name + '_min'
            max_name = name + '_max'
            df = pd.DataFrame()
            df[min_name] = df_in.groupby(name)[scale_var].min()
            df[max_name] = df_in.groupby(name)[scale_var].max()
            assert min(df[min_name]) == 1 and max(df[max_name]) == scale_max  # checks boundary conditions
            df[name] = [str(df[min_name][i]) + '-' + str(df[max_name][i]) if df[min_name][i] != df[max_name][i] else str(df[min_name][i]) for i in df.index]
            df.sort_values(by=[min_name], inplace=True, ignore_index=True)
            cluster_ranges = cluster_ranges.join(df[name])
    cluster_ranges = cluster_ranges.transpose().rename(columns=clusters_dict)
    return cluster_ranges

#------------------------#
#  CLUSTERING (K-MEANS)  #
#------------------------#

# CLUSTER BOUNDARIES

# mais midpoints
mais_boundary_options = []
for i in mais.index:
    if i not in (0, len(mais)):
        new_mais = 0.5 * (mais['mais'][i-1] + mais['mais'][i])
        mais_boundary_options.append(new_mais)

# iss midpoints
iss_boundary_options = []
for i in iss.index:
    if i not in (0, len(iss)):
        new_iss = 0.5 * (iss['iss'][i-1] + iss['iss'][i])
        iss_boundary_options.append(new_iss)

# CLUSTER ASSIGNMENTS

# mais 
mais_clusters = pd.DataFrame(mais['mais'])
for c in mais_num_clusters:
    for var in mais_cluster_vars:
        mais_clusters['mais_c' + str(c) + '_' + var] = kmeans(mais, 'mais', var, c)

# iss
iss_clusters = pd.DataFrame(iss['iss'])
iss_clusters_input = pd.concat([iss[['iss'] + iss_dist_vars + iss_pdeath_vars], iss_costs[mais_cost_vars]], axis=1)
for c in iss_num_clusters:
    for var in iss_cluster_vars:
        iss_clusters['iss_c' + str(c) + '_' + var] = kmeans(iss_clusters_input, 'iss', var, c)

iss_clusters.loc[iss_clusters['iss'] == 3, 'iss_c5_cost_Fink_etal_2006_qol'] = 1  # ad hoc correction

# CLUSTER RANGES

mais_cluster_ranges = cluster_ranges(mais_clusters, 'mais', mais_cluster_vars, mais_num_clusters, mais_clusters_dict)
iss_cluster_ranges = cluster_ranges(iss_clusters, 'iss', iss_cluster_vars, iss_num_clusters, iss_clusters_dict)

# iss clusters cessation
iss_clusters_cease = pd.DataFrame(columns=['clusters', 'median_iss_ceases'])
for c in range(max(iss_num_clusters)-1):
    name = 'upper' + str(c+1)
    iss_cluster_ranges[name] = iss_cluster_ranges['cluster' + str(c+1)].apply(lambda x: '' if (str(x) == 'nan') else x if ('-' not in x) else x.split('-')[1])
    iss_cluster_ranges[name] = iss_cluster_ranges[name].apply(lambda x: x if (x != '75') else '')
    no_miss = [int(item) for item in iss_cluster_ranges[name] if item != '']
    iss_clusters_cease.loc[len(iss_clusters_cease)] = [c+1, np.median(no_miss)]

# mais clusters cessation
mais_clusters_cease = pd.DataFrame(columns=['clusters', 'median_mais_ceases'])
for c in range(max(mais_num_clusters)-1):
    name = 'upper' + str(c+1)
    mais_cluster_ranges[name] = mais_cluster_ranges['cluster' + str(c+1)].apply(lambda x: '' if (str(x) == 'nan') else x if ('-' not in x) else x.split('-')[1])
    mais_cluster_ranges[name] = mais_cluster_ranges[name].apply(lambda x: x if (x != '6') else '')
    no_miss = [int(item) for item in mais_cluster_ranges[name] if item != '']
    mais_clusters_cease.loc[len(mais_clusters_cease)] = [c+1, np.median(no_miss)]

mais_cluster_ranges = mais_cluster_ranges[mais_clusters_cols]
iss_cluster_ranges = iss_cluster_ranges[iss_clusters_cols]

#----------#
#  TABLES  #
#----------#

# hospitalized (incidence)
table1 = hosp.copy(deep=True)
table1[hosp_dist_vars] = pct_format(table1[hosp_dist_vars], 1)  
table1[hosp_cost_vars] = cost_format(table1[hosp_cost_vars])

# mais (incidence / hosp / mortality)
table2 = pd.DataFrame(mais[['mais'] + mais_noncost_vars])
table2[mais_dist_vars] = pct_format(table2[mais_dist_vars], 1)
table2[mais_hosp_vars] = round(table2[mais_hosp_vars], 3)
table2[mais_pdeath_vars] = round(table2[mais_pdeath_vars], 3)
for var in mais_hosp_vars:
    table2[var] = table2[var].apply(lambda x: 1 if x == 1 else '{:.3f}'.format(x))
for var in mais_pdeath_vars:
    table2[var] = table2[var].apply(lambda x: 1 if x == 1 else '{:.3f}'.format(x))
table2['hosp_rate_Blin_etal_2023'] = table2['hosp_rate_Blin_etal_2023'].apply(lambda x: 1 if x == 1 else x)

# mais costs
table3 = pd.DataFrame(mais[['mais'] + mais_cost_vars])
table3[mais_cost_vars] = cost_format(table3[mais_cost_vars])

# iss bifurcated (incidence / mortality)
varlist = ['iss', 'dist_copes_etal_1988', 'dist_kilgo_etal_2004', 'pdeath_copes_etal_1988', 'pdeath_kilgo_etal_2004']
table4_pre = pd.DataFrame(iss[varlist])
for var in iss_dist_vars:
    table4_pre[var] = pct_format(table4_pre[var], 2)
for var in iss_pdeath_vars:
    digits = 3
    fmt = '{:.' + str(digits) + 'f}'
    table4_pre[var] = table4_pre[var].apply(lambda x: '0' if round(x, digits) == 0 else '1' if round(x, digits) == 1 else fmt.format(x))
# single-column -> two-column
iss_breakpoint = 25
left = pd.DataFrame(table4_pre[table4_pre['iss'] <= iss_breakpoint])
right = pd.DataFrame(table4_pre[table4_pre['iss'] > iss_breakpoint]).reset_index(drop=True)
table4 = pd.DataFrame()
table4[table4_pre.columns + '_col1'] = left
table4[table4_pre.columns + '_col2'] = right
del table4_pre

# mais shares / avg mais (by iss)
varlist = ['iss', 'mais_theory', 'mais_valid', 'triplet_1', 'triplet_2', 'wgt1', 'wgt2', 'total_cnt_fmt', 'body_regions_cnt', 'mais_avg']
table5 = pd.DataFrame(iss_mais[varlist])
table5[['wgt1', 'wgt2']] = pct_format(table5[['wgt1', 'wgt2']], 2)
table5['mais_avg'] = table5['mais_avg'].apply(lambda x: sig_figs(x,3))
table5 = dash_format(table5)

# logistic regressions
LL_text_1 = 'LL-full = ' + loglik_mnl_1_fmt + '\nLL-simple = ' + loglik_olr_1_fmt
LL_text_2 = 'LL-full = ' + loglik_mnl_2_fmt + '\nLL-simple = ' + loglik_olr_2_fmt
varlist = ['model_type', 'n', 'level', 'x_var', 'const_full', 'slope_full']
table6 = pd.DataFrame(mnl_results[varlist])
table6.loc[len(table6)] = ['theoretical', n_logreg_1, 'mais', 'iss', LL_text_1, chi_sqrd_1_fmt]
table6.loc[len(table6)] = ['valid only', n_logreg_2, 'mais', 'iss', LL_text_2, chi_sqrd_2_fmt]

# linear regressions
varlist = ['model_type', 'n', 'y_var', 'x_var', 'const_full', 'slope_full', 'r_sqrd_fmt', 'norm_corr_fmt']
table7 = pd.concat([ols_pwr[varlist], cost_funcs[varlist], reduced_form[varlist]], axis=0).reset_index(drop=True)
table7 = dash_format(table7)
table7['const_full'] = table7['const_full'].apply(lambda x: '0' if x == '0.00' else x)

# mais clusters
table8 = pd.DataFrame(mais_cluster_ranges[mais_clusters_cols])
table8 = dash_format(table8)

# iss clusters
table9 = pd.DataFrame(iss_cluster_ranges[iss_clusters_cols])
table9 = dash_format(table9)

# scale avgs / analogues
varlist = ['scale', 'incidence', 'avg_fmt', 'mais_est_1_fmt', 'mais_est_2_fmt', 'iss_est_1_fmt', 'iss_est_2_fmt']
table10 = pd.DataFrame(scale_avgs[varlist])
table10 = dash_format(table10)

# average costs
varlist = mais_cost_vars + ['cost_wisqars_2025']
table11 = pd.concat([hosp_avg_costs, mais_avg_costs, iss_avg_costs], axis=0, ignore_index=True)
table11 = table11[varlist]
table11[varlist] = cost_format(table11[varlist])
table11 = dash_format(table11)

#-------------#
#  PLOT PREP  #
#-------------#

# parameters
title_size = 10
axis_labels_size = 8
axis_ticks_size = 7
legend_size = 6
point_size = 7
line_width = 1
region_alpha = 0.25
iss_increment = 5
max_cost = 14       # in millions 
cost_increment = 2  # in millions

assert 75 % iss_increment == 0
assert max_cost % cost_increment == 0

# spatial buffers
fraction = 0.02
buffer_mais = (6-1)*fraction
buffer_iss = (75)*fraction
buffer_cost = fraction * max_cost
legend_fraction = fraction / (2*fraction + 1)

y_cost_labels = ['$' + str(k) for k in np.arange(0, max_cost+1, 2)]

# plot colors
legend_color = 'whitesmoke'
plot_area_color = 'white'
background_color = 'whitesmoke' if plots_for_publication else 'lightblue'

# feasible region bounds (mais-iss)
region_bounds = pd.DataFrame()
region_bounds['iss'] = np.arange(1, 75+1)                           # includes all integers 1-75 (for smoothness)
region_bounds['mais_min_theory'] = np.sqrt(region_bounds['iss']/3)  # excludes ceiling function (to get entire curve - not just integers)
region_bounds['mais_max_theory'] = np.sqrt(region_bounds['iss'])    # excludes floor function (to get entire curve - not just integers)
region_bounds['mais_min_theory'] = region_bounds['mais_min_theory'].apply(lambda x: 1 if x < 1 else 5 if x > 5 else x)  # feasible region stops at mais 1
region_bounds['mais_max_theory'] = region_bounds['mais_max_theory'].apply(lambda x: 1 if x < 1 else 5 if x > 5 else x)  # feasible region stops at mais 5
region_bounds = region_bounds[region_bounds['iss'] <= 66]  # feasible region stops at iss 66
y_lower = region_bounds['mais_min_theory']
y_upper = region_bounds['mais_max_theory']

# feasible region textbox
bounds_title = 'Theoretical region' + '\n\nMAIS 1-5, ISS 1-66' 
bounds_body = '\nISS lower bound: MAIS^2' + '\nISS upper bound: 3•MAIS^2' 
bounds_text = bounds_title + bounds_body

# power functions parameters
pwr_const_text_1 = sig_figs(np.exp(const_pwr_1), 3)  # constant must be exponentiated (linear -> power)
pwr_const_text_2 = sig_figs(np.exp(const_pwr_2), 3)
pwr_slope_text_1 = sig_figs(slope_pwr_1, 3)
pwr_slope_text_2 = sig_figs(slope_pwr_2, 3)
pwr_rsqrd_text_1 = r_sqrd_format(r_sqrd_pwr_1)
pwr_rsqrd_text_2 = r_sqrd_format(r_sqrd_pwr_2)

# power functions textbox
pwr_func_title = 'Power function fits'
pwr_func_text_fit_1 = 'Unconstrained (ISS 2-66)\nMAIS-avg = ' + pwr_const_text_1 + '•(ISS)^' + pwr_slope_text_1 + '\nR^2=' + pwr_rsqrd_text_1 + ' (n=' + str(n_ols_pwr) + ')'
pwr_func_text_fit_2 = 'Constrained (ISS 1-75)\nMAIS-avg = ' + pwr_const_text_2 + '•(ISS)^' + pwr_slope_text_2 + '\nR^2=' + pwr_rsqrd_text_2 + ' (n=' + str(n_ols_pwr) + ')'
pwr_func_text = pwr_func_title + '\n\n' + pwr_func_text_fit_1 + '\n\n' + pwr_func_text_fit_2

#---------#
#  PLOTS  #
#---------#

# fig1 - mais costs
fig1 = plt.figure(facecolor=background_color)
plt.gca().set_facecolor(plot_area_color)
for cost in mais_plot_cost_vars:   
    plt.plot(mais['mais'], mais[cost] / 10**6, label=name_dict[cost], linestyle=linestyle_dict[cost], color=color_dict[cost], linewidth=line_width, zorder=5)  # entire curve
    plt.scatter(mais['mais'], mais[cost] / 10**6, marker='*', color='black', s=point_size, zorder=10)  # integer points only
if plots_for_publication == False:
    plt.title('MAIS Economic Costs', fontsize=title_size, fontweight='bold')
plt.xlabel('Maximum Abbreviated Injury Scale (MAIS)', fontsize=axis_labels_size)
plt.ylabel('Cost (millions) (2023$)', fontsize=axis_labels_size)
plt.xlim(1 - buffer_mais, 6 + buffer_mais)
plt.ylim(-buffer_cost, max_cost + buffer_cost)
plt.xticks(np.arange(1, 6+1), fontsize=axis_ticks_size)
plt.yticks(np.arange(0, max_cost+1, cost_increment), labels=y_cost_labels, fontsize=axis_ticks_size)
plt.legend(loc='upper left', bbox_to_anchor=(legend_fraction, 1 - legend_fraction), fontsize=legend_size, facecolor=legend_color, framealpha=1).set_zorder(20)
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
plt.show(True)

# fig2 - feasible region / power functions (mais-iss)
fig2, ax2 = plt.subplots(facecolor=background_color)
plt.gca().set_facecolor(plot_area_color)
if plots_for_publication == False:
    plt.title('MAIS-ISS Map', fontsize=title_size, fontweight='bold')
ax2.fill_between(region_bounds['iss'], y_upper, y_lower, color='blue', label='Theoretical region', alpha=region_alpha, zorder=5)
# plt.plot([66,75], [5,5], color='blue', alpha=region_alpha, linewidth=line_width, zorder=5)  # line from iss 66 to 75 (horz)
# plt.plot([75,75], [5,6], color='blue', alpha=region_alpha, linewidth=line_width, zorder=5)  # line connecting the two iss 75 points (vert)
plt.plot(iss_mais['iss'], iss_mais['mais_avg'], label='Average MAIS (n=' + str(n_ols_pwr) + ')', color='red', linestyle='dashed', linewidth=line_width, zorder=10)  # avg mais curve
plt.scatter(iss_valid['iss'], iss_valid['mais'], color='black', marker='*', label='Valid MAIS-ISS (n=' + str(n_mais_iss_valid) + ')', s=point_size, zorder=15)  # valid mais-iss only
plt.xlabel('Injury Severity Score (ISS)', fontsize=axis_labels_size)
plt.ylabel('Maximum Abbreviated Injury Scale (MAIS)', fontsize=axis_labels_size)
plt.text(75, 3, bounds_text, va='center', ha='right', fontsize=legend_size)
plt.text(75, 1, pwr_func_text, va='bottom', ha='right', fontsize=legend_size)
plt.xlim(-buffer_iss, 75 + buffer_iss)
plt.ylim(1 - buffer_mais, 6 + buffer_mais)
plt.xticks(np.arange(0, 75+1, iss_increment), fontsize=axis_ticks_size)
plt.yticks(np.arange(1, 6+1), fontsize=axis_ticks_size)
plt.legend(loc='upper left', bbox_to_anchor=(legend_fraction, 1 - legend_fraction), fontsize=legend_size, facecolor=legend_color, framealpha=1).set_zorder(20)
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
plt.show(True)

# fig3 - iss costs
fig3 = plt.figure(facecolor=background_color)
plt.gca().set_facecolor(plot_area_color)
if plots_for_publication == False:
    plt.title('ISS Economic Costs', fontsize=title_size, fontweight='bold')
for cost in mais_plot_cost_vars:
    plt.plot(iss_costs['iss'], iss_costs[cost] / 10**6, label=name_dict[cost], linestyle=linestyle_dict[cost], color=color_dict[cost], linewidth=line_width, zorder=5)
plt.xlabel('Injury Severity Score (ISS)', fontsize=axis_labels_size)
plt.ylabel('Cost (millions) (2023$)', fontsize=axis_labels_size)
plt.text(75 - buffer_iss/2, buffer_cost/2, 'n=44 per curve (one per ISS value)', va='bottom', ha='right', fontsize=legend_size)
plt.xlim(0,75)
plt.ylim(0, max_cost)
plt.xticks(np.arange(0, 75+1, iss_increment), fontsize=axis_ticks_size)
plt.yticks(np.arange(0, max_cost+1, cost_increment), labels=y_cost_labels, fontsize=axis_ticks_size)
plt.legend(loc='upper left', bbox_to_anchor=(legend_fraction, 1 - legend_fraction), fontsize=legend_size, facecolor=legend_color, framealpha=1).set_zorder(20)
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
plt.show(True)

#----------#
#  EXPORT  #
#----------#

# plots
if export_plots:
    fig_list = [fig1, fig2, fig3]
    pdf_name = 'Injury_costs_graphics.pdf'
    delete_file(pdf_name)
    pdf = PdfPages(pdf_name)
    for fig in fig_list:
        f = fig_list.index(fig) + 1
        filename = 'INJ-figure-' + str(f) + '.jpg'
        delete_file(filename)
        fig.savefig(filename, dpi=300)
        if (include_fig1_in_pdf == True) or (f != 1):
            pdf.savefig(fig)
    pdf.close()
    del pdf, fig, f

# residual plots
if export_plots and include_residual_plots:
    pdf_name = 'Injury_costs_residuals.pdf'
    delete_file(pdf_name)
    pdf = PdfPages(pdf_name)
    for r in range(resid_plot_num):
        exec('pdf.savefig(rnorm' + str(r+1) + ')')
    for r in range(resid_plot_num):
        exec('pdf.savefig(rvar' + str(r+1) + ')')
    pdf.close()
    del pdf

###

# runtime
runtime_sec = round(time.time()-time0, 2)
if runtime_sec < 60:
    print('\nruntime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec/60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\nruntime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0

