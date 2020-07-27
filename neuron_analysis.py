import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
import matplotlib.pyplot as plt


def get_contrast_modulation(spikes, contrast):
    """
    Computes how much each neuron's firing rate is modulated by contrast. Because firing rates are not normally
    distributed, perform Kruskal-Wallis test instead of ANOVA. Aside the p-value from Kruskal-Wallis, we also compute
    the fraction of variance explained by contrast conditions 'eta2' from the test statistic 'H'.

    Args:
        spikes (np.array): spike data with shape [neurons, trials, time bins]
        contrast (np.array): contralateral contrast (0 - 1) for every trial

    Returns:
        results (np.array): test results (p-value of Kruskal-Wallis test (col 1) and fraction of variance
                            explained by contrast condition (col 2)) with shape [neurons, results]

    Hendrik (22.07.2020)
    """

    # Get mean firing rate in Hz of all neurons for each trial across available time bins (10 ms/bin)
    mean_fr = np.sum(spikes, axis=2)/(spikes.shape[2]/100)

    # For every neuron, sort trials into 4 groups by contrast condition and perform Kruskal-Wallis test.
    # Then, calculate explained variance eta2 from test statistic 'H'.
    test_stats = np.zeros((mean_fr.shape[0], 2))
    for neuron in range(mean_fr.shape[0]):

        # neuron=132
        # Sort trials into one of 4 np.arrays in the list 'fr_per_cont' according to contrast condition
        fr_per_cont = []
        cont_label = []
        for c in np.unique(contrast):
            curr_mask = contrast == c
            fr_per_cont.append(mean_fr[neuron, curr_mask])
            cont_label.extend([f'{int(c*100)}%']*sum(curr_mask))

        df = pd.DataFrame(data={'fr': np.hstack(fr_per_cont), 'contrast': cont_label})

        # plt.figure()
        # plt.title(f'Neuron {neuron}')
        # sns.barplot(data=df, x='contrast', y='fr')
        # sns.swarmplot(data=df, x='contrast', y='fr', color='0', alpha=0.35)

        # Perform Kruskal-Wallis test and use 'H' statistic to compute fraction of variance explained
        try:
            H, p = kruskal(*fr_per_cont)
            eta2 = (H - len(fr_per_cont) + 1) / (mean_fr.shape[1] - len(fr_per_cont))
            test_stats[neuron] = np.array([p, eta2])
        # If a neuron does not fire once, Kruskal-Wallis fails. Enter dummy values.
        except ValueError:
            test_stats[neuron] = np.array([1, 0])


    # Return test statistics for every neuron:
    # Kruskal-Wallis p-value (1st column) and fraction of explained variance (2nd column)
    return test_stats


        # _, bins = np.histogram(df2["fr"])
        # g = sns.FacetGrid(df2, hue="contrast")
        # g = (g.map(sns.distplot, "fr", bins=bins, hist=False).add_legend())
    #
    #
    #
    # sns.distplot(df2['fr'])


def population_vector_correlation(spikes, contrast, plot=False, fig=None):

    # Get avg firing rate for each trial
    mean_fr = np.sum(spikes, axis=2)/(spikes.shape[2]/100)

    # Average firing rates across contrast condition trials
    contrasts = np.unique(contrast)
    activity_matrix = np.zeros((len(contrasts), mean_fr.shape[0]))
    for idx, cont in enumerate(contrasts):
        mask = contrast == cont
        activity_matrix[idx] = np.mean(mean_fr[:, mask], axis=1)

    y, std = pvc_curve(activity_matrix, plot=False)

    if plot:
        if fig is None:
            fig = plt.figure()
        out = plot_pvc_curve(y, std, fig=fig)

    return y, std


def pvc_curve(activity_matrix, plot=True):
    """Calculate the mean pvc curve

        Parameters
        ----------
        activity_matrix : 2D array containing (float, dim1 = bins, dim2 = neurons)
        plot: bool, optional
        max_delta_bins: int, optional
            max difference in bin distance

       Returns
       -------
       curve_yvals:
           array of mean pvc curve (idx = delta_bin)
    """
    max_delta_bins = activity_matrix.shape[0]-1
    num_bins = np.size(activity_matrix,0)
    num_neurons = np.size(activity_matrix,1)
    curve_yvals = np.empty(max_delta_bins + 1)
    curve_stdev = np.empty(max_delta_bins + 1)
    for delta_bin in range(max_delta_bins + 1):
        pvc_vals = []
        for offset in range(num_bins - delta_bin):
            idx_x = offset
            idx_y = offset + delta_bin
            pvc_xy_num = pvc_xy_den_term1 = pvc_xy_den_term2 = 0
            for neuron in range(num_neurons):
                pvc_xy_num += activity_matrix[idx_x][neuron] * activity_matrix[idx_y][neuron]
                pvc_xy_den_term1 += activity_matrix[idx_x][neuron]*activity_matrix[idx_x][neuron]
                pvc_xy_den_term2 += activity_matrix[idx_y][neuron]*activity_matrix[idx_y][neuron]
            pvc_xy = pvc_xy_num / (np.sqrt(pvc_xy_den_term1*pvc_xy_den_term2))
            pvc_vals.append(pvc_xy)
        mean_pvc_delta_bin = np.mean(pvc_vals)
        stdev_delta_bin = np.std(pvc_vals)
        curve_yvals[delta_bin] = mean_pvc_delta_bin
        curve_stdev[delta_bin] = stdev_delta_bin

    if plot:
        plot_pvc_curve(curve_yvals, curve_stdev, show=False)

    return curve_yvals, curve_stdev


def plot_pvc_curve(y_vals, session_stdev, show=False, fig=None):
    """Plots the pvc curve

        Parameters
        ----------
        y_vals : array-like
            data points of the pvc curve (idx = bin distance)
        bin_size : bool, optional
        show : bool, optional

       Returns
       -------
       fig: figure object
           a figure object of the pvc curve
    """
    if fig is None:
        fig = plt.figure()
    x_axis = np.arange(4)
    line, _, _ = plt.errorbar(x_axis, y_vals, session_stdev, figure=fig)
    plt.fill_between(x_axis, y_vals - session_stdev, y_vals + session_stdev, alpha=0.3)
    plt.xticks(x_axis, labels=[0, 25, 50, 100])
    # plt.ylim(bottom=0.5);
    plt.ylabel('Mean PVC')
    plt.xlim(left=0); plt.xlabel('Contrast [%]')
    if show:
        plt.show(block=True)
    return fig, line