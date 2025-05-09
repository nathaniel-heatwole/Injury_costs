# INJURY_COSTS.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# available at https://github.com/nathaniel-heatwole/
# see also https://www.linkedin.com/in/nathaniel-heatwole/
# maps MAIS-based costs onto the ISS (ordinal logistic regression, Gaussian naive Bayes) and groups severity ranges using various features (k-means clustering)

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from statsmodels.miscmodels.ordinal_model import OrderedModel

time0 = time.time()
np.random.seed(1099)

#--------------#
#  PARAMETERS  #
#--------------#

mais_num_clusters = [2]
iss_num_clusters = [2, 3, 4]

hosp_dist_digits = 1
mais_dist_digits = 1
iss_dist_digits = 2

hosp_rate_digits = 3
pdeath_digits = 3
shares_digits = 1

pdeath_fmt = '{:.' + str(pdeath_digits) + 'f}'
hosp_rate_fmt = '{:.' + str(hosp_rate_digits) + 'f}'

currency_sig_digits = 3

iss_breakpoint = 25

include_plot_titles = 0  # 1 = yes / 0 = no

#---------#
#  LISTS  #
#---------#

hosp_dist_vars = ['dist_Fink_etal_2006', 'dist_wisqars_2025']
hosp_cost_vars = ['cost_Fink_etal_2006_qol', 'cost_wisqars_2025']

mais_hosp_vars = ['hosp_rate_Blin_etal_2023']
mais_dist_vars = ['dist_Copes_etal_1990', 'dist_Fink_etal_2006', 'dist_Blin_etal_2023']
mais_pdeath_vars = ['pdeath_Copes_etal_1990', 'pdeath_Genn_etal_1994', 'pdeath_Genn_Wod_2006']

mais_cost_vars = ['cost_Grah_etal_1997', 'cost_Fink_etal_2006_qol', 'cost_DOT_2021', 'cost_Blin_etal_2023']
mais_plot_cost_vars = ['cost_Grah_etal_1997', 'cost_DOT_2021', 'cost_Blin_etal_2023', 'cost_Fink_etal_2006_qol']

cost_vars_universe = mais_cost_vars + ['cost_wisqars_2025']

cost_colors_dict = {'cost_Grah_etal_1997':'black',
                    'cost_Fink_etal_2006_qol':'red',
                    'cost_DOT_2021':'darkorange',
                    'cost_Blin_etal_2023':'blue'}

cost_names_dict = {'cost_Grah_etal_1997':'Graham et al. [29,47]',
                   'cost_Fink_etal_2006_qol':'Finkelstein et al. [18] + quality-of-life [11]',
                   'cost_DOT_2021':'U.S. Dept. of Transportation [2,47]',
                   'cost_Blin_etal_2023':'Blincoe et al. [11]'}

iss_source_names = ['copes_etal_1988', 'kilgo_etal_2004']
iss_main_source = 'kilgo_etal_2004'

#-------------------#
#  GENERATED LISTS  #
#-------------------#

# iss lists
iss_dist_vars = []
iss_pdeath_vars = []
for source in iss_source_names:
    iss_dist_vars.append('dist_' + source)
    iss_pdeath_vars.append('pdeath_' + source)

# shares lists
mais_1to5_shares = []
mais_1to6_shares = []
for m in range(1, 6+1):
    if m != 6:
        mais_1to5_shares.append('mais' + str(m) + '_share')
        mais_1to6_shares.append('mais' + str(m) + '_share')
    else:
        mais_1to6_shares.append('mais' + str(m) + '_share')

# cluster variables lists
mais_noncost_vars = mais_dist_vars + mais_hosp_vars + mais_pdeath_vars
mais_cluster_vars = mais_noncost_vars + mais_cost_vars
iss_cluster_vars = iss_dist_vars + iss_pdeath_vars + mais_cost_vars

# cluster names (list, dicts)
clusters_cols = []
mais_clusters_dict = {}
iss_clusters_dict = {}
for m in range(1, max(mais_num_clusters + iss_num_clusters)+1):
    clusters_cols.append('cluster' + str(m)) 
for c in range(max(mais_num_clusters)):
    mais_clusters_dict[c] = 'cluster' + str(c+1)
for c in range(max(iss_num_clusters)):
    iss_clusters_dict[c] = 'cluster' + str(c+1)

#-------------#
#  FUNCTIONS  #
#-------------#

# format cost variables
def cost_format(df_in):
    df_in = pd.DataFrame(df_in)
    df = pd.DataFrame()
    for var in df_in.columns:
        df_in['oom'] = df_in[var].apply(lambda x: 1 if x == 0 else np.floor(math.log10(abs(x))))  # determines order of magnitude
        df_in['ratio'] = df_in[var] / 10**df_in['oom']  # condenses value
        df_in['ratio'] = round(df_in['ratio'], currency_sig_digits - 1)  # rounds condensed value
        df[var] = df_in['ratio'] * 10**df_in['oom']  # expands condensed value back out
        df[var] = round(df[var], 0)  # removes cents
        df[var] = df[var].apply(lambda x: x if str(x) == 'nan' else f'${int(x):,.0f}')  # prints in currency format
    return df

# add dashes (missing/nan)
def dash_format(df):
    df = pd.DataFrame(df)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: '-' if str(x) in ['nan', '0', '0%'] else str(x))
    return df

# format percent quantities
def percent_format(df, digits):
    df = pd.DataFrame(df)
    fmt = '{:.' + str(digits) + 'f}'
    for col in df.columns:
        df[col] = round(100*df[col], digits)
        df[col] = df[col].apply(lambda x: '0%' if x == 0 else '100%' if x == 100 else str(fmt.format(x)) + '%')
    return df

# mais bounds (by iss)
def mais_bounds(iss):
    df = pd.DataFrame()
    df['iss'] = iss
    df['mais_min'] = np.ceil(np.sqrt(iss/3))
    df['mais_max'] = np.floor(np.sqrt(iss))
    df['mais_max'] = df['mais_max'].apply(lambda x: 5 if x > 5 else x)  # cap at mais 5
    df['mais_max'] = [6 if df['iss'][i] == 75 else df['mais_max'][i] for i in df.index]  # mais 6 / iss 75
    return df[['mais_min', 'mais_max']]

# adjust mais shares (by iss)
def adjust_shares(df):
    # reset impossible mais-iss shares to zero
    for m in range(1, 5+1):
        name = 'mais' + str(m) + '_share'
        df.loc[df['mais_min'] > m, name] = 0
        df.loc[df['mais_max'] < m, name] = 0
        df.loc[df['iss'] == 75, name] = 0  # iss 75
    df['mais6_share'] = df['iss'].apply(lambda x: 1 if x == 75 else 0)  # mais 6 / iss 75
    # renormalize remaining shares to one
    df['constant'] = 1 / df[mais_1to6_shares].sum(axis=1)
    for m in range(1, 6+1):
        name = 'mais' + str(m) + '_share'
        df[name] *= df['constant']
    df.drop(columns=['constant', 'mais_min', 'mais_max'], axis=1, inplace=True)
    return df[mais_1to6_shares]

#---------#
#  INPUT  #
#---------#

hosp = pd.read_csv('hospitalized_summary.csv')
mais = pd.read_csv('mais_summary.csv')
iss = pd.read_csv('iss_summary.csv')

#------------#
#  MAIS/ISS  #
#------------#

# iss bounds (by mais)
mais['iss_min'] = mais['mais']**2
mais['iss_max'] = 3*mais['mais']**2
mais['iss_min'] = [75 if mais['mais'][i] == 6 else mais['iss_min'][i] for i in mais.index]  # mais 6 / iss 75
mais['iss_max'] = [75 if mais['mais'][i] == 6 else mais['iss_max'][i] for i in mais.index]  # mais 6 / iss 75

# mais incidence/mortality
for dist, source in zip(iss_dist_vars, iss_source_names):
    iss['inj_' + source] = iss['total_' + source] - iss['deaths_' + source]
    iss['dist_' + source] = iss['inj_' + source] / sum(iss['inj_' + source])
    iss['pdeath_' + source] = iss['deaths_' + source] / iss['total_' + source]

# mais bounds (by iss)
iss[['mais_min', 'mais_max']] = mais_bounds(iss['iss'])
iss['total_mais'] = iss['mais_max'] - iss['mais_min'] + 1
iss['mais'] = [list(np.arange(int(iss['mais_min'][i]), int(iss['mais_max'][i] + 1))) for i in iss.index]
iss['mais_str'] = [str(iss['mais'][i]) for i in iss.index]
iss['mais_str'] = iss['mais_str'].apply(lambda x: str(x).replace('[','(').replace(']',')'))

# all iss-mais combos (rowwise)
iss_exploded = iss.explode('mais', ignore_index=True)
iss_exploded['mais'] = iss_exploded['mais'].apply(lambda x: int(x))

# mais combos (by iss) and iss counts (by total mais)
mais_combos = pd.DataFrame()
mais_combos[['total_mais', 'mais_combos']] = iss.groupby('total_mais')['mais_str'].agg(set).reset_index()
mais_combos.sort_values(by=['total_mais'], inplace=True, ignore_index=True)
iss_counts = pd.DataFrame()
iss_counts['cnt'] = iss.groupby('total_mais').size()
iss_counts['pct'] = round(100 * iss_counts['cnt'] / sum(iss_counts['cnt']), 1)

# mais/iss ranges
iss_ranges = pd.DataFrame()
iss_ranges['iss_min'] = iss_exploded.groupby('mais')['iss'].min()
iss_ranges['iss_max'] = iss_exploded.groupby('mais')['iss'].max()
mais_ranges = pd.DataFrame()
mais_ranges['mais_min'] = iss_exploded.groupby('iss')['mais'].min()
mais_ranges['mais_max'] = iss_exploded.groupby('iss')['mais'].max()

# calcs
iss_unique_values = list(iss['iss'].unique())
total_iss_values = len(iss_unique_values)
unique_iss_mais_combos = len(iss_exploded)

# iss rows of interest
iss_15_vicinity = iss.loc[iss['iss'].isin([14, 16])].reset_index()
iss_min_max_rows = iss.loc[iss['total_' + iss_main_source].isin([min(iss['total_' + iss_main_source]), max(iss['total_' + iss_main_source])])].reset_index()
iss_single_mais = iss[['iss', 'total_mais']].loc[iss['total_mais'] == 1].reset_index()

#---------------#
#  SCALE MEANS  #
#---------------#

# hosp percent
hosp_pcts = pd.DataFrame(columns=['scale', 'incidence', 'avg_val', 'iss_min', 'iss_max'])
for dist in hosp_dist_vars:
    hosp_pcts.loc[len(hosp_pcts)] = ['hosp.', dist, hosp[dist][1], '-', '-']
hosp_pcts['avg_val'] = percent_format(pd.DataFrame(hosp_pcts['avg_val']), hosp_dist_digits)

# mais scale avg (by source)
mais_avgs = pd.DataFrame(columns=['scale', 'incidence', 'avg_val'])
for dist in mais_dist_vars:
    mais_avgs.loc[len(mais_avgs)] = ['mais', dist, sum(mais[dist] * mais['mais'])]
mais_avgs['mais_floor'] = np.floor(mais_avgs['avg_val'])
mais_avgs['mais_ceiling'] = np.ceil(mais_avgs['avg_val'])
mais_avgs = pd.merge(mais_avgs, mais[['mais', 'iss_min']], left_on='mais_floor', right_on='mais')
mais_avgs = pd.merge(mais_avgs, mais[['mais', 'iss_max']], left_on='mais_ceiling', right_on='mais')
mais_avgs.drop(columns=['mais_floor', 'mais_ceiling', 'mais_x', 'mais_y'], inplace=True)
mais_avgs['avg_val'] = round(mais_avgs['avg_val'], 1)
mais_avgs['avg_val'] = mais_avgs['avg_val'].apply(lambda x: '{:.1f}'.format(x))

# iss scale avg (by source)
iss_avgs = pd.DataFrame(columns=['scale', 'incidence', 'avg_val', 'iss_min', 'iss_max'])
for source in iss_source_names:
    iss_avgs.loc[len(iss_avgs)] = ['iss', source, sum(iss['dist_' + source] * iss['iss']), '-', '-']
iss_avgs['avg_val'] = round(iss_avgs['avg_val'], 1)
iss_avgs['avg_val'] = iss_avgs['avg_val'].apply(lambda x: '{:.1f}'.format(x))

scale_avgs = pd.concat([hosp_pcts, mais_avgs, iss_avgs], axis=0).reset_index(drop=True)

#----------------------#
#  TRAINING/TEST DATA  #
#----------------------#

training = iss_exploded.loc[iss_exploded['iss'] != 75]  # iss 75 separate
x_train = training['iss']
y_train = training['mais']
x_test = pd.DataFrame(iss['iss'])

num_training_pts = len(training)
num_test_pts = len(x_test)

#-----------------------#
#  LOGISTIC REGRESSION  #
#-----------------------#

# fit model
olr_model = OrderedModel(y_train, x_train, distr='logit')
olr_fit = olr_model.fit(method='bfgs')

# print summary
print(Fore.GREEN + '\033[1m' + '\n' + 'LOGISTIC REGRESSION' + '\n' + Style.RESET_ALL)
print(olr_fit.summary())

# predict shares (by iss)
shares_olr = pd.DataFrame(iss[['iss', 'mais_str', 'mais_min', 'mais_max']])
shares_olr[mais_1to5_shares] = olr_fit.predict(x_test)
shares_olr[mais_1to6_shares] = adjust_shares(shares_olr)

#---------------#
#  NAIVE BAYES  #
#---------------#

y_resized = np.resize(y_train, (len(y_train), ))

# fit model
gnb_model = GaussianNB(priors=None, var_smoothing=0)
gnb_fit = gnb_model.fit(pd.DataFrame(x_train), y_resized)

# predict shares (by iss)
shares_gnb = iss[['iss', 'mais_str', 'mais_min', 'mais_max']]
shares_gnb[mais_1to5_shares] = gnb_fit.predict_proba(x_test)
shares_gnb[mais_1to6_shares] = adjust_shares(shares_gnb)

#-------------#
#  ISS COSTS  #
#-------------#

# final shares
shares = pd.DataFrame(iss[['iss', 'mais_str', 'mais_min', 'mais_max']])
for name in mais_1to6_shares:
    shares[name] = 0.5*(shares_olr[name] + shares_gnb[name])  # average olr/gnb

# ensure shares sum to one (at level of precision of final table)
shares['pdf_sum'] = round(shares[mais_1to6_shares], mais_dist_digits).sum(axis=1)
for i in shares.index:
    assert math.isclose(shares['pdf_sum'][i], 1), 'Shares do not sum to one'
shares.drop(columns=['pdf_sum'], axis=1, inplace=True)

# mapped iss cost values
iss_costs = pd.DataFrame(iss['iss'])
for cost in mais_cost_vars:
    iss_costs[cost] = 0
    for m in range(1, 6+1): 
        iss_costs[cost] += mais[cost][m-1] * shares['mais' + str(m) + '_share']  # applies shares to mais costs (loop needed because shares are rowwise)

#------------------------#
#  CLUSTERING (K-MEANS)  #
#------------------------#

# mais cluster assignments
mais_clusters = pd.DataFrame(mais['mais'])
x_kmeans_mais = pd.DataFrame(mais['mais'])
for c in mais_num_clusters:
    for var in mais_cluster_vars:
        y_kmeans_mais = pd.DataFrame(mais[var])
        kmeans_model = KMeans(n_clusters=c, n_init='auto').fit(x_kmeans_mais, y_kmeans_mais)
        mais_clusters['mais_' + 'c' + str(c) + '_' + var] = kmeans_model.labels_ + 1

# iss cluster assignments
iss_clusters = pd.DataFrame(iss['iss'])
iss_clusters_input = pd.concat([iss[['iss'] + iss_dist_vars + iss_pdeath_vars], iss_costs[mais_cost_vars]], axis=1)
x_kmeans_iss = pd.DataFrame(iss_clusters_input['iss'])
for c in iss_num_clusters:
    for var in iss_cluster_vars:
        y_kmeans_iss = pd.DataFrame(iss_clusters_input[var])
        kmeans_model = KMeans(n_clusters=c, n_init='auto').fit(x_kmeans_iss, y_kmeans_iss) 
        iss_clusters['iss_' + 'c' + str(c) + '_' + var] = kmeans_model.labels_ + 1

# mais cluster ranges
mais_cluster_ranges = pd.DataFrame()
for c in mais_num_clusters:
    for var in mais_cluster_vars:
        name = 'mais_' + 'c' + str(c) + '_' + var
        min_name = name + '_min'
        max_name = name + '_max'
        df = pd.DataFrame()
        df[min_name] = mais_clusters.groupby(name)['mais'].min()
        df[max_name] = mais_clusters.groupby(name)['mais'].max()
        df[name] = [str(df[min_name][i]) + '-' + str(df[max_name][i]) for i in df.index]
        df.sort_values(by=[min_name], inplace=True, ignore_index=True)
        mais_cluster_ranges[name] = df[name]
mais_cluster_ranges = mais_cluster_ranges.transpose().rename(columns=mais_clusters_dict)

max_clusters_overall = max(mais_num_clusters + iss_num_clusters)
index_vals = list(np.arange(0, max_clusters_overall))

# iss cluster ranges
iss_cluster_ranges = pd.DataFrame(index=index_vals)
for c in iss_num_clusters:
    for var in iss_cluster_vars:
        name = 'iss_' + 'c' + str(c) + '_' + var
        min_name = name + '_min'
        max_name = name + '_max'
        df = pd.DataFrame()
        df[min_name] = iss_clusters.groupby(name)['iss'].min()
        df[max_name] = iss_clusters.groupby(name)['iss'].max()
        df[name] = [str(df[min_name][i]) + '-' + str(df[max_name][i]) for i in df.index]
        df.sort_values(by=[min_name], inplace=True, ignore_index=True)
        iss_cluster_ranges = iss_cluster_ranges.join(df[name])
iss_cluster_ranges = iss_cluster_ranges.transpose().rename(columns=iss_clusters_dict)

#-----------------#
#  AVERAGE COSTS  #
#-----------------#

# hospitalized
hosp_avg_costs = pd.DataFrame()
hosp_avg_costs['scale'] = len(hosp_dist_vars) * ['hosp.']
hosp_avg_costs['incidence'] = hosp_dist_vars
for cost in hosp_cost_vars:
    avg_values = []
    for dist in hosp_dist_vars:
        avg_values.append(sum(hosp[dist] * hosp[cost]))
    hosp_avg_costs[cost] = avg_values

# mais
mais_avg_costs = pd.DataFrame()
mais_avg_costs['scale'] = len(mais_dist_vars) * ['mais']
mais_avg_costs['incidence'] = mais_dist_vars
for cost in mais_cost_vars:
    avg_values = []
    for dist in mais_dist_vars:
        avg_values.append(sum(mais[dist] * mais[cost]))
    mais_avg_costs[cost] = avg_values

# iss
iss_avg_costs = pd.DataFrame()
iss_avg_costs['scale'] = len(iss_dist_vars) * ['iss']
iss_avg_costs['incidence'] = iss_dist_vars
for cost in mais_cost_vars:
    avg_values = []
    for dist in iss_dist_vars:
        avg_values.append(sum(iss[dist] * iss_costs[cost]))
    iss_avg_costs[cost] = avg_values

# use shares to calc average mais (at each iss)
shares['avg_mais'] = 0
for m in range(1, 6+1): 
    shares['avg_mais'] += m * shares['mais' + str(m) + '_share']  # applies shares to determine average mais (loop needed because shares are rowwise)
shares['avg_mais'] = shares['avg_mais'].apply(lambda x: '{:.2f}'.format(round(x, 2)))

#----------#
#  TABLES  #
#----------#

# hospitalized (incidence/costs)
table0 = hosp.copy(deep=True)
table0[hosp_dist_vars] = percent_format(table0[hosp_dist_vars], hosp_dist_digits)  
table0[hosp_cost_vars] = cost_format(table0[hosp_cost_vars])

# mais (incidence/hosp/mortality)
table1 = mais[['mais'] + mais_noncost_vars]
table1[mais_dist_vars] = percent_format(table1[mais_dist_vars], mais_dist_digits)
table1[mais_hosp_vars] = round(table1[mais_hosp_vars], hosp_rate_digits)
table1[mais_pdeath_vars] = round(table1[mais_pdeath_vars], pdeath_digits)
for var in mais_hosp_vars:
    table1[var] = table1[var].apply(lambda x: 1 if x == 1 else hosp_rate_fmt.format(x))
for var in mais_pdeath_vars:
    table1[var] = table1[var].apply(lambda x: 1 if x == 1 else pdeath_fmt.format(x))
table1['hosp_rate_Blin_etal_2023'] = table1['hosp_rate_Blin_etal_2023'].apply(lambda x: 1 if x == 1 else x)

# mais costs
table2 = mais[['mais', 'iss_min', 'iss_max'] + mais_cost_vars]
table2[mais_cost_vars] = cost_format(table2[mais_cost_vars])  # formats cost variables

# iss (incidence/mortality)
table3 = pd.DataFrame()
table3_prelim = iss[['iss', 'dist_copes_etal_1988', 'dist_kilgo_etal_2004', 'pdeath_copes_etal_1988', 'pdeath_kilgo_etal_2004']]
for var in iss_dist_vars:
    table3_prelim[var] = percent_format(table3_prelim[var], iss_dist_digits)
for var in iss_pdeath_vars:
    table3_prelim[var] = round(table3_prelim[var], pdeath_digits)
    table3_prelim[var] = [pdeath_fmt.format(table3_prelim[var][i]) for i in table3_prelim.index]
# single-column table -> two-column
left = pd.DataFrame(table3_prelim[table3_prelim['iss'] <= iss_breakpoint])
right = pd.DataFrame(table3_prelim[table3_prelim['iss'] > iss_breakpoint]).reset_index(drop=True)
table3[table3_prelim.columns + '_col1'] = left
table3[table3_prelim.columns + '_col2'] = right
del table3_prelim

# mais shares (by iss)
table4 = shares[['iss', 'mais_str', 'avg_mais']]
table4[mais_1to6_shares] = percent_format(shares[mais_1to6_shares], shares_digits)
table4[mais_1to6_shares] = dash_format(table4[mais_1to6_shares])

# clusters (mais/iss ranges)
table5 = pd.concat([mais_cluster_ranges, iss_cluster_ranges], axis=0)
table5[clusters_cols] = dash_format(table5[clusters_cols])

# average costs (all scales)
table6 = pd.concat([hosp_avg_costs, mais_avg_costs, iss_avg_costs], axis=0, ignore_index=True)
table6[cost_vars_universe] = cost_format(table6[cost_vars_universe])
table6[cost_vars_universe] = dash_format(table6[cost_vars_universe])
table6 = table6.join(scale_avgs['avg_val'])
table6 = table6[['scale', 'incidence', 'avg_val'] + cost_vars_universe]

#---------#
#  PLOTS  #
#---------#

# plot parameters
title_size = 11
axis_labels_size = 9
axis_ticks_size = 8
legend_size = 8
point_size = 8
line_width = 1.25
horz_buffer_mais = 0
vert_buffer_mais = 0.15
horz_buffer_iss = 1.5
region_alpha = 0.25
max_cost = 14  # millions
max_iss = 75

# cost labels (y-axis)
y_cost_labels = list(np.arange(0, max_cost+1, 2))
for i in range(len(y_cost_labels)):
    y_cost_labels[i] = '$' + str(y_cost_labels[i])

# mais-iss feasible region curves (bounds are rearranged version of mais bounds for iss)
iss_bounds = pd.DataFrame()
iss_bounds['iss'] = np.arange(1, max_iss+1)          # integers 1-75 (for smoothness)
iss_bounds['mais_min'] = np.sqrt(iss_bounds['iss']/3)  # excludes ceiling function (for curve, not just integers)
iss_bounds['mais_max'] = np.sqrt(iss_bounds['iss'])    # excludes floor function (for curve, not just integers)
iss_bounds['mais_min'] = iss_bounds['mais_min'].apply(lambda x: 1 if x < 1 else 5 if x > 5 else x)  # ensures region does not go below mais 1
iss_bounds['mais_max'] = iss_bounds['mais_max'].apply(lambda x: 1 if x < 1 else 5 if x > 5 else x)  # ensures region does not go above mais 5

# mais-iss feasible region bounds
region_bounds = iss_bounds[iss_bounds['iss'] <= 66]  # region does not extend beyond iss 66
y_lower = region_bounds['mais_min']
y_upper = region_bounds['mais_max']

# mais costs plot
fig1 = plt.figure()
for cost in mais_plot_cost_vars:   
    plt.plot(mais['mais'], mais[cost] / 10**6, label=cost_names_dict[cost], color=cost_colors_dict[cost], linewidth=line_width, zorder=5)  # entire curve
    plt.scatter(mais['mais'], mais[cost] / 10**6, marker='*', color='black', s=point_size, zorder=10)    # integer points only
if include_plot_titles == 1:
    plt.title('MAIS Economic Costs', fontsize=title_size, fontweight='bold')
plt.xlabel('Maximum Abbreviated Injury Scale (MAIS)', fontsize=axis_labels_size)
plt.ylabel('Cost (million 2023$)', fontsize=axis_labels_size)
plt.xlim(1 - horz_buffer_mais, 6 + horz_buffer_mais)
plt.ylim(0, max_cost)
plt.xticks(np.arange(1, 6+1), fontsize=axis_ticks_size)
plt.yticks(np.arange(0, max_cost+1, 2), labels=y_cost_labels, fontsize=axis_ticks_size)
plt.legend(loc='upper left', fontsize=legend_size, facecolor='white', framealpha=1)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

# feasible region plot (mais-iss)
fig2, ax = plt.subplots()
if include_plot_titles == 1:
    plt.title('MAIS-ISS Map', fontsize=title_size, fontweight='bold')
ax.fill_between(region_bounds['iss'], y_upper, y_lower, color='blue', label='Feasible region', alpha=region_alpha, zorder=5)
plt.plot([66, 75], [5, 5], color='blue', alpha=region_alpha, linewidth=line_width, zorder=10)  # line from iss 66 to 75 (horz)
plt.plot([75, 75], [5, 6], color='blue', alpha=region_alpha, linewidth=line_width, zorder=10)  # line connecting two iss 75 points (vert)
plt.scatter(iss_exploded['iss'], iss_exploded['mais'], color='black', marker='*', label='Valid MAIS-ISS', s=point_size, zorder=15)  # valid mais-iss points
plt.xlabel('Injury Severity Score (ISS)', fontsize=axis_labels_size)
plt.ylabel('Maximum Abbreviated Injury Scale (MAIS)', fontsize=axis_labels_size)
plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.0325), fontsize=legend_size, facecolor='white', framealpha=1)
plt.xlim(0 - horz_buffer_iss, max_iss + horz_buffer_iss)
plt.ylim(1 - vert_buffer_mais, 6 + vert_buffer_mais)
plt.xticks(np.arange(0, max_iss + horz_buffer_iss + 1, 5), fontsize=axis_ticks_size)
plt.yticks(np.arange(1, 6+1), fontsize=axis_ticks_size)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

# iss costs plot
fig3 = plt.figure()
for cost in mais_plot_cost_vars:
    plt.plot(iss_costs['iss'], iss_costs[cost] / 10**6, label=cost_names_dict[cost], color=cost_colors_dict[cost], linewidth=line_width, zorder=5)
if include_plot_titles == 1:
    plt.title('MAIS Economic Costs Mapped to ISS', fontsize=title_size, fontweight='bold')
plt.xlabel('Injury Severity Score (ISS)', fontsize=axis_labels_size)
plt.ylabel('Cost (million 2023$)', fontsize=axis_labels_size)
plt.xlim(0, max_iss)
plt.ylim(0, max_cost)
plt.xticks(np.arange(0, max_iss+1, 5), fontsize=axis_ticks_size)
plt.yticks(np.arange(0, max_cost+1, 2), labels=y_cost_labels, fontsize=axis_ticks_size)
plt.legend(loc='upper left', fontsize=legend_size, facecolor='white', framealpha=1)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

# export figures (png)
f = 1
for fig in [fig1, fig2, fig3]:
    fig.savefig('figure-' + str(f) + '.png', dpi=300)
    f += 1
del fig, f


###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0
