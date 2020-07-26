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