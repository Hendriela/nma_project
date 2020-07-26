import numpy as np

def ztransform_pupil_size(pupil, nsds_crit=4, outlier_reject=False):
    '''
    This function computes the z-transformed mean pupil sizes.

      Args:
        pupil (np.array): pupil data of one session of shape (ntrials, ntime_bins).
        nsds_crit (int): number of standard deviations from the mean that is considered as criterion to classify outliers.
        outlier_reject (boolean): indicates whether to compute z scores for all trials or excluding outlier trials, default is False.

      Returns:
        z_scores (np.array): z-transformed mean pupil sizes across time_bins, shape is (ntrials,).
        idx_out (list): *optional output* if outlier_reject=True a list of rejected outlier trial indices is returned in addition to z_scores.

        (Anne, 23.07.2020)
    '''

    # check for negative pupil sizes
    if np.any(pupil.min(axis=1) < 0):
        idx_neg = np.where(pupil.min(axis=1) < 0)[0]
        print(f'Warning: Negative pupil size value: trial(s) {idx_neg}!')

    # compute mean pupil size per trial
    means_per_trl = np.array([pupil[trl, :].mean(axis=0) for trl in range(pupil.shape[0])])

    # raise ValueError if any mean value is negative
    if np.any(means_per_trl) < 0:
        idx_err = np.where(means_per_trl < 0)[0]
        raise ValueError(f'Negative mean value(s): trial(s) {idx_err}')

    # check for outliers in mean values
    outlier_crit = means_per_trl.mean() + nsds_crit * means_per_trl.std()

    if np.any(means_per_trl) > outlier_crit:
        idx_out = np.where(means_per_trl > outlier_crit)[0]
        print(f'Warning: Potential outlier: trial(s) {idx_out}!')

    # compute z scores including all trials
    z_scores = np.array([(val - means_per_trl.mean()) / means_per_trl.std() for val in means_per_trl])

    if outlier_reject:
        # reject outlier trials
        # return indices of rejected trials + z-scores (length ntrials, nan for all rejected trials)

        z_scores = []
        idx_out = np.where(means_per_trl > outlier_crit)[0]
        out_mask = np.array([0 if idx in idx_out else 1 for idx in range(len(means_per_trl))], dtype=bool)

        for idx, val in enumerate(means_per_trl):

            if idx in idx_out:
                z_scores.append(np.nan)
            else:
                z_scores.append((val - means_per_trl[out_mask].mean()) / means_per_trl[out_mask].std())

        z_scores = np.array(z_scores)

        return z_scores, idx_out

    return z_scores


def get_pupil_percentiles(pupil, percentiles, nsds_crit=4, outlier_reject=True):
    """
    Get indices of trials with a small and large pupil size. Thresholds are set by percentiles. The output masks
    can be used to filter for pupil size via indexing (e.g. spikes[:, low_perc_mask] to get spikes of trials with
    a small pupil size.

      Args:
        pupil (np.array): pupil data of one session of shape (n_trials, n_time_bins).
        percentiles (tuple): lower and upper percentile that serves as thresholds for small and large pupil size.
        nsds_crit (int): number of standard deviations from the mean above which trials are classified as outliers.
        outlier_reject (boolean): indicates whether to exclude outlier trials while computing z scores (default False).

      Returns:
        low_perc_mask (np.array): boolean mask of shape (n_trials,) that is True for trials with a small pupil size
        high_perc_mask (np.array): boolean mask of shape (n_trials,) that is True for trials with a large pupil size

    (Hendrik, 26.07.2020)
    """
    # Get Z-scored mean pupil sizes from Anna's function
    if outlier_reject:
        pup_z = ztransform_pupil_size(pupil, nsds_crit, outlier_reject)[0]
    else:
        pup_z = ztransform_pupil_size(pupil, nsds_crit, outlier_reject)

    # Create masks that are TRUE for trials that are below/above the percentiles chosen
    low_perc_mask = pup_z < np.nanpercentile(pup_z, percentiles[0])
    high_perc_mask = pup_z > np.nanpercentile(pup_z, percentiles[1])

    # Return the masks that can be used to filter trials via indexing
    return low_perc_mask, high_perc_mask


def split_by_pupil_size(pupil_mask, test_set=20):
    """
    Selects a testing set from a subset of trials marked by pupil_size_mask (i.e. trials with small or large
    pupil sizes).

      Args:
        pupil_mask (np.array): boolean mask indicating the trial subset from which to draw test set trials.
        test_set (int): size of the test set as percentage of dataset size.

      Returns:
        test_mask (np.array): boolean mask of shape (n_trials,) that is True for trials selected for the testing set.

    (Hendrik, 26.07.2020)
    """
    # Initialize boolean array
    test_mask = np.zeros(len(pupil_mask), dtype=bool)
    # Set array to True at randomly chosen indices of trials with small/large pupil size (as marked by pupil_mask)
    test_mask[np.random.choice(np.where(pupil_mask)[0], int(len(pupil_mask)*(test_set/100)), replace=False)] = True

    return test_mask
