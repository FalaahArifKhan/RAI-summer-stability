import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('mpl20')
matplotlib.rcParams['figure.dpi'] = 100
sns.set(rc={'figure.figsize':(12, 12)})


def set_default_plot_properties():
    plt.style.use('mpl20')
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['figure.figsize'] = 15, 5


def sns_set_size(height, width):
    sns.set(rc={'figure.figsize':(width, height)})


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


def null_scenario_analysis(data, corrupted_data, target_col, condition_col, special_values):
    # Count of nulls
    print(f'Count of nulls for {target_col} column: {corrupted_data[target_col].isnull().sum()}')
    print(f'Count of all records in {target_col} column: {data[target_col].count()}')
    print(f'Count of records in {condition_col} column in the defined condition: {data[data[condition_col].isin(special_values)][target_col].count()}\n\n')

    # Print density plots for the target column before and after the corruption
    plt.figure()
    sns.displot(data[target_col]).set(title=f'Density of {target_col} Column Before Corruption')
    plt.show()

    plt.figure()
    sns.displot(corrupted_data[target_col]).set(title=f'Density of {target_col} Column After Corruption')
    plt.show()

    # Print density plots for AGEP split by race before and after the corruption
    plot_column = 'AGEP'
    print(f'Plot {plot_column} column Split by Race [Before Corruption]')
    plot_column_grouped_by_race(data, plot_column)
    print(f'\n\n\nPlot {plot_column} column Split by Race [After Corruption]')
    plot_column_grouped_by_race(corrupted_data[~corrupted_data[target_col].isnull()], plot_column)