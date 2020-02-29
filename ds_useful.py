import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import mean_absolute_error

# Include the below code in jupyter notebooks to link people to this file for reference
# [My Useful Data Science Functions](https://github.com/cobyoram/python-for-data-scientists/blob/master/ds_useful.py)

# ----- FUNCTION FOR DISTRIBUTIONS ---------------|
def auto_subplots(df, **kwargs):
    '''
    This function creates a series of sublots to display the distribution for all continuous variables in a DataFrame
    It operates like a FacetGrid, but it's a little more customized.

    The dimensions of the subplot (no_subplots X no_subplots) are calculated automatically by how many continuous variables are found.
    You can use a number of kwargs to customize the output of the function. Those include:
    kwargs:
        limitx  -   Limits the number of subplots along the x-axis, and calculates the no_subplots on y axis from given limit
        kind    -   Allows you to specify which type of distribution grid you'd like to use. Values include
            -   hist: creates a matplotlib.pyplot histogram
            -   boxplot: creates a seaborn boxplot
                -   whis: adjust the boxplot whisker bounds
            -   swarm: create a one-dimensional seaborn swarmplot

    Returns None

    '''
    EACH_SIZE = 3
    # WSPACE = .3
    # HSPACE = .7
    DEFAULT_BOXPLOT_WHIS = 1.5

    columns = df.select_dtypes(include='number').columns
    len_cols = len(columns)

    if kwargs.get('limitx'):
        limitx = kwargs.get('limitx')
        count_dimensions = tuple([int(len_cols/limitx) + 1, limitx])

    else:
        try_num = len_cols
        while True:
            sq = math.sqrt(try_num)
            if sq == int(sq):
                break
            try_num += 1
        count_dimensions = tuple([sq, sq])

    dimensions = tuple([count_dimensions[0] * EACH_SIZE, count_dimensions[1] * EACH_SIZE])
    plt.figure(figsize=dimensions)

    for i, col in enumerate(columns, 1):
        plt.subplot(count_dimensions[0], count_dimensions[1], i)
        if kwargs.get('kind'):
            kind = kwargs.get('kind')
            selection = ['hist', 'boxplot', 'swarm']
            if kind == 'hist':
                plt.hist(df[col])
            elif kind == 'boxplot':
                whis = DEFAULT_BOXPLOT_WHIS
                if kwargs.get('whis'):
                    whis = kwargs.get('whis')
                sns.boxplot(df[col], whis=whis)
            elif kind == 'swarm':
                sns.swarmplot(y=col, data=df)
            else:
                print('Kind: {} is currently unavailable. For now enjoy our limited selection of: {}'.format(kind, selection))
        else:
            plt.hist(df[col])
        plt.title(col)

    # plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
    plt.tight_layout()
    plt.show()

def make_subplots(df, plotfunc=None, func_args=[], func_kwargs={}, limitx=8, each_size=3, **kwargs):
    '''
    Makes a subplot, filled with a given plotting function
    '''
    columns = df.columns
    len_cols = len(columns)

    try_num = len_cols
    while True:
        sq = math.sqrt(try_num)
        if sq == int(sq):
            break
        try_num += 1
    count_dimensions = tuple([sq, sq])

    if count_dimensions[0] > limitx:
        count_dimensions = tuple([int(len_cols/limitx) + 1, limitx])

    dimensions = tuple([count_dimensions[1] * each_size, count_dimensions[0] * each_size])
    plt.figure(figsize=dimensions)

    for i, col in enumerate(columns, 1):
        plt.subplot(count_dimensions[0], count_dimensions[1], i)
        plotfunc(df, col, *func_args, **func_kwargs)
        plt.title(col)

    plt.tight_layout()
    plt.show()
# ------- END OF DISTRIBUTION SELF_MADE FUNCS ----------------------|

# ----- FUNCTION FOR GENERAL MISSING VALUES ---------------|
def missingness_summary(df, **kwargs):
    '''
    This function creates a series representing what percentage of data is null for each column of a dataframe

    You can use a number of kwargs to customize the function. Those include:
    kwargs:
        print_log   -   [True, False]: If true, will print the output before returning the Series (default False)
        sort        -   ['asc', 'desc']: Allows you to sort the data by ascending or descending (default 'desc')
    
    Returns Series with index = column names and value = percentage of nulls in column

    '''
    s = df.isna().sum()*100/len(df)

    sort = 'desc'
    if kwargs.get('sort'):
        sort = kwargs.get('sort')
    if sort == 'asc':
        s.sort_values(ascending=True, inplace=True)
    elif sort == 'desc':
        s.sort_values(ascending=False, inplace=True)

    print_log = False
    if kwargs.get('print_log'):
        print_log = kwargs.get('print_log')
    if print_log == True:
        print(s)

    return s
# ------- END OF MISSING SELF_MADE FUNCS ----------------------|

# ----- FUNCTIONS FOR GENERAL OUTLIER HANDLING --------------|
def get_minmax_with_threshold(s, threshold=1.5, range_type='iqr'):
    if range_type == 'iqr':
        q75, q25 = np.percentile(s, [75,25])
        ranged = q75 - q25
    elif range_type == 'std':
        ranged = s.std()

    min_val = q25 - (ranged*threshold)
    max_val = q75 + (ranged*threshold)
    
    return min_val, max_val
    
def get_outliers(s, threshold=1.5, range_type='iqr'):
    min_val, max_val = get_minmax_with_threshold(s, threshold, range_type=range_type)
    return s.loc[(s > max_val) | (s < min_val)]

def outliers_summary(df, threshold=1.5, range_type='iqr', **kwargs):  
    '''
    This function creates a series representing what percentage of data are outliers for each column of a dataframe

    You can use a number of kwargs to customize the function. Those include:
    kwargs:
        print_log   -   [True, False]: If true, will print the output before returning the Series (default False)
        sort        -   ['asc', 'desc']: Allows you to sort the data by ascending or descending (default 'desc')
    
    Returns Series with index = column names and value = percentage of outliers in column

    '''
    s = pd.Series([get_outliers(df[col], threshold, range_type=range_type).count() *100 / len(df[col])
                   for col in df.select_dtypes(include='number').columns],
                 index=df.select_dtypes(include='number').columns)
    
    sort = 'desc'
    if kwargs.get('sort'):
        sort = kwargs.get('sort')
    if sort == 'asc':
        s.sort_values(ascending=True, inplace=True)
    elif sort == 'desc':
        s.sort_values(ascending=False, inplace=True)

    print_log = False
    if kwargs.get('print_log'):
        print_log = kwargs.get('print_log')
    if print_log == True:
        print(s)
        
    return s

def get_percentiles(df, column_name, threshold=1.5, range_type='iqr'):
    min_val, max_val = get_minmax_with_threshold(df[column_name], threshold, range_type=range_type)
    
    max_percentile = df.loc[df[column_name] >= max_val, column_name].count() / len(df[column_name])
    min_percentile = df.loc[df[column_name] <= min_val, column_name].count() / len(df[column_name])
    
    return min_percentile, max_percentile

def drop_outliers(df, threshold=1.5, range_type='iqr'):
    drop_inds = set()
    for col in outliers_summary(df, threshold, range_type=range_type).index:
        outliers = get_outliers(df[col], threshold, range_type=range_type).index
        drop_inds = set(list(drop_inds) + list(outliers))
    df.drop(index=drop_inds, inplace=True)
    return df
# ------- END OF OUTLIER SELF_MADE FUNCS ----------------------|

# ----- FUNCTIONS FOR ANALYZING MODELS --------------|
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
# ------- END OF MODEL ANALYSIS SELF_MADE FUNCS ----------------------|

def repeats_summary(df, sort='desc', print_log=False, value_agg='none', value=0):
    repeats_percents = []
    print_value = []
    for col in df.columns:
        if value_agg == 'none':
            value = value
            print_value = []
        elif value_agg == 'mode':
            value = df[col].mode().iloc[0]
        elif value_agg == 'mean':
            value = df[col].mean().iloc[0]
        elif value_agg == 'median':
            value = df[col].median().iloc[0]
        elif value_agg == 'max':
            value = df[col].max().iloc[0]
        elif value_agg == 'min':
            value = df[col].min().iloc[0]
        else: raise ValueError('Wrong entry for \'value_agg\'. Will accept \'mode\', \'mean\', \'median\', \'max\', \'min\'')

        repeats_percents.append(len(df.loc[df[col] == value])*100/len(df))
        print_value.append(value)

    S = pd.Series(repeats_percents, index=df.columns)
    if sort == 'desc':
        S = S.sort_values(ascending=False)
    elif sort == 'asc':
        S = S.sort_values(ascending=True)
    else: raise ValueError('Wrong entry for \'sort\'. Will accept \'asc\' or \'desc\'')

    if print_log:
        print(f'Repeated values: {print_value}\n{S}')

    return S

def drop_null_rows(df):
    for col in df.columns:
        df.drop(df.loc[df[col].isnull()].index, axis=0, inplace=True)
    return df

def remove_correlated_features(dataset, target, threshold):
    col_corr = set()
    corr_matrix = dataset.drop(target, axis=1).corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    # print(f'Deleted {colname} from dataset.')
                    del dataset[colname]
    return dataset

# Function is outdated
def similar_variables(df, target, similarity_threshold=.9, print_log=False):
    corr = df.corr()

    feature_corr = corr.drop(target, axis=1).drop(target, axis=0)
    target_corr = corr.loc[target]

    similar_pairs = []
    for col in feature_corr.columns:
        for index in feature_corr.index:
            if np.abs(feature_corr.loc[index, col]) > similarity_threshold and index != col and [col, index] not in similar_pairs:
                similar_pairs.append([index, col])

    # Then we'll find the variable in each similar pair that is less correlated (which we'll drop later)
    drop_variables = []
    corr_to_target = []

    for pair in similar_pairs:
        if target_corr[pair[0]] < target_corr[pair[1]]:
            drop_variables.append(pair[0])
            corr_to_target.append(target_corr[pair[0]])
        else:
            drop_variables.append(pair[1])
            corr_to_target.append(target_corr[pair[0]])

    S = pd.Series(corr_to_target, index=drop_variables)
    
    if print_log:
        print(S)
    
    return S

def get_significant_category_columns(df, target, sig=True):
    sig_cols = set()
    for col in df.select_dtypes('object').columns:
        ucats = []
        for ucat in df[col].unique():
            ucats.append(df.loc[df[col] == ucat, target])
        anova = stats.f_oneway(*ucats)
        if anova.pvalue < .05 and sig:
            sig_cols.update([col])
        if anova.pvalue >= .05 and not sig:
            sig_cols.update([col])
    
    return sig_cols

def sort_by_correlation(df, target_name, sort='desc', abs=True):
    corr_df = df.corr()
    if sort=='asc':
        ascending=True
    elif sort == 'desc':
        ascending=False
    corr = corr_df[target_name].copy()
    if abs:
        corr = np.abs(corr)
    corr.sort_values(ascending=ascending, inplace=True)
    return corr

def sort_by_categorical_var(df, target_name, sort='desc'):
    cat_cols = df.select_dtypes('object').columns
    stds = []
    for col in cat_cols:
        cats = df[col].unique()
        cat_means = [df.loc[df[col]==cat,target_name].mean() for cat in cats]
        col_std = np.std(cat_means)
        stds.append(col_std)
    s = pd.Series(stds, index=cat_cols)
    if sort=='asc':
        ascending = True
    elif sort=='desc':
        ascending=False
    s.sort_values(ascending = ascending, inplace=True)
    return s

def print_evaluation_metrics(true, predicted):
    print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(true, predicted)))
    print("Mean squared error of the prediction is: {}".format(mse(true, predicted)))
    print("Root mean squared error of the prediction is: {}".format(rmse(true, predicted)))
    print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((true - predicted) / true)) * 100))

# def auto_subplots(df, **kwargs):
#     '''
#     This function creates a series of sublots to display the distribution for all continuous variables in a DataFrame
#     It operates like a FacetGrid, but it's a little more customized.

#     The dimensions of the subplot (no_subplots X no_subplots) are calculated automatically by how many continuous variables are found.
#     You can use a number of kwargs to customize the output of the function. Those include:
#     kwargs:
#         limitx  -   Limits the number of subplots along the x-axis, and calculates the no_subplots on y axis from given limit
#         kind    -   Allows you to specify which type of distribution grid you'd like to use. Values include
#             -   hist: creates a matplotlib.pyplot histogram
#             -   boxplot: creates a seaborn boxplot
#                 -   whis: adjust the boxplot whisker bounds
#             -   swarm: create a one-dimensional seaborn swarmplot

#     Returns None

#     '''
#     EACH_SIZE = 3
#     WSPACE = .3
#     HSPACE = .7
#     DEFAULT_BOXPLOT_WHIS = 1.5

#     columns = df.select_dtypes(include='number').columns
#     len_cols = len(columns)

#     categories = df.select_dtypes(include='object').columns
#     len_cats = len(categories)

#     if kwargs.get('limitx'):
#         limitx = kwargs.get('limitx')
#         count_dimensions = tuple([limitx, int(len_cols/limitx + 1)])
#     else:
#         try_num = len_cols
#         while True:
#             sq = math.sqrt(try_num)
#             if sq == int(sq):
#                 break
#             try_num += 1
#         count_dimensions = tuple([sq, sq])

#     dimensions = tuple([count_dimensions[0] * EACH_SIZE, count_dimensions[1] * EACH_SIZE])
#     plt.figure(figsize=dimensions)

#     for category in categories:
#         for sub_cat in df[category].unique():
#             if not kwargs.get('categorical'):
#                 sub_cat = df[category].unique()

#             plot_df = df.loc[df[category].isin(pd.Series(sub_cat))]

#             for i, col in enumerate(columns, 1):
#                 plt.subplot(count_dimensions[0], count_dimensions[1], i)
#                 if kwargs.get('kind'):
#                     kind = kwargs.get('kind')
#                     selection = ['hist', 'boxplot', 'swarm']
#                     if kind == 'hist':
#                         plt.hist(plot_df[col])
#                     elif kind == 'boxplot':
#                         whis = DEFAULT_BOXPLOT_WHIS
#                         if kwargs.get('whis'):
#                             whis = kwargs.get('whis')
#                         sns.boxplot(plot_df[col], whis=whis)
#                     elif kind == 'swarm':
#                         sns.swarmplot(y=col, data=plot_df)
#                     else:
#                         print('Kind: {} is currently unavailable. For now enjoy our limited selection of: {}'.format(kind, selection))
#                 else:
#                     plt.hist(plot_df[col])
#             plt.title(col)
#         if not kwargs.get('categorical'):
#             break

#     # plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
#     plt.tight_layout()
#     plt.show()

# auto_subplots(life_df, categorical=True)
