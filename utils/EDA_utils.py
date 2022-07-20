import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


def set_default_plot_properties():
    plt.style.use('mpl20')
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['figure.figsize'] = 15, 5


def sns_set_size(height, width):
    sns.set(rc={'figure.figsize':(width, height)})


def plot_generic(x, y, xlabel, ylabel, plot_title):
    plt.figure(figsize=(20,10))
    plt.scatter(x, y)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(plot_title, fontsize=20)
    plt.show()


def plot_column_grouped_by_race(data, plot_column):
    fig = plt.figure()
    gs0 = matplotlib.gridspec.GridSpec(ncols=3, nrows=3, figure=fig, wspace=0.5, hspace=0.5)
    race_names = ['White alone', 'Black or African American alone', 'American Indian alone',
                  'Alaska Native alone', 'American Indian', 'Asian alone',
                  'Hawaiian and Pacific Islander', 'Some Other Race alone', 'Two or More Races']

    for i in range(len(race_names)):
        n_row, n_col = divmod(i, 3)
        ax = fig.add_subplot(gs0[n_row, n_col])
        ax.set_title(race_names[i])
        sns.distplot(data[data['RAC1P'] == i + 1][plot_column], ax=ax)

    plt.show()


def null_scenario_analysis(data, corrupted_data, target_col, condition_col, special_values,
                           print_plots_grouped_by_race=True):
    """
    Display plots to compare a real dataset without nulls and a corrupted dataset with nulls, created
     based on a special null scenario
    """
    # Count of nulls
    print(f'Count of nulls for {target_col} column: {corrupted_data[target_col].isnull().sum()}')
    print(f'Count of all records in {target_col} column: {data[target_col].count()}')
    print(f'Count of records in {condition_col} column in the defined condition: '
          f'{data[data[condition_col].isin(special_values)][target_col].count()}\n\n')

    # Print density plots for the target column before and after the corruption
    plt.figure()
    sns.displot(data[target_col]).set(title=f'Density of {target_col} Column Before Corruption')
    plt.show()

    plt.figure()
    sns.displot(corrupted_data[target_col]).set(title=f'Density of {target_col} Column After Corruption')
    plt.show()

    if print_plots_grouped_by_race:
        # Print density plots for AGEP split by race before and after the corruption
        plot_column = 'AGEP'
        print(f'Plot {plot_column} column Split by Race [Before Corruption]')
        plot_column_grouped_by_race(data, plot_column)
        print(f'\n\n\nPlot {plot_column} column Split by Race [After Corruption]')
        plot_column_grouped_by_race(corrupted_data[~corrupted_data[target_col].isnull()], plot_column)


def imputed_nulls_analysis(real_data, imputed_data, corrupted_data, target_col):
    """
    Display side-by-side plots to compare a real dataset without nulls, a corrupted dataset with nulls, created
     based on a special null scenario, and a dataset without nulls, which were imputed with one of imputation techniques
    """
    print(f"Number of nulls in {target_col} column in the corrupted dataframe: ", corrupted_data[target_col].isnull().sum())
    print(f"Number of nulls in {target_col} column in the imputed dataframe: ", imputed_data[target_col].isnull().sum())

    # Print density plots for the target column for corrupted and imputed dataframes
    f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    sns.despine(left=True)

    sns.histplot(data=corrupted_data[target_col], kde=True, color='b', ax=ax[0])
    ax[0].set(title=f'Density of Corrupted {target_col} Column')
    sns.histplot(data=imputed_data[target_col], kde=True, color='m', ax=ax[1])
    ax[1].set(title=f'Density of Imputed {target_col} Column')
    sns.histplot(data=real_data[target_col], kde=True, color='g', ax=ax[2])
    ax[2].set(title=f'Density of Real {target_col} Column')
    plt.show()
