import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import neuron_analysis as ana

#%% Data I/O
def download_data(target=r'C:\Users\Hendrik\PycharmProjects\nma_project\data'):
    fname = []
    for j in range(3):
        fname.append(os.path.join(target, 'steinmetz_part%d.npz' % j))
    url = ["https://osf.io/agvxh/download", "https://osf.io/uv3mw/download", "https://osf.io/ehmw2/download"]

    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            try:
                r = requests.get(url[j])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[j], "wb") as fid:
                        fid.write(r.content)


def load_data(path=r'C:\Users\Hendrik\PycharmProjects\nma_project\data'):
    alldat = np.array([])
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            alldat = np.hstack((alldat, np.load(os.path.join(dirpath, file), allow_pickle=True)['dat']))
    return alldat


def filter_data(data, session, areas, time_bins=(50, 150), inverse=False):
    """
    Filter the Steinmetz dataset by session, brain area and time bin and return the corresponding
    neuronal spikes, pupil area and contrast.

    Args:
        data (np.array): np.array containing all session dicts (alldat)
        session (int): index of preferred session
        areas (str, list of str or None): list of names of preferred brain areas. Can be 'all_vis' to get all
                                          visual areas. If None, all areas are returned.
        time_bins (tuple): first and last time bin of neural data. If None, all time bins are returned.
                           Defaults to (50, 150), the first second after stimulus onset
        inverse (bool): bool flag whether the area filters should be inverted. This results in all regions
                        EXCEPT the ones listed in 'areas' being selected.

    Returns:
        spikes (np.array): binned spikes of filtered neurons and time bins with shape [neurons, trials, time bins]
        pupil_area (np.array): pupil area [cm²?] with shape [trials, time bins (unfiltered)]
        contrast (np.array): contrast of the right stimulus for each trial

    Hendrik (22.07.2020)
    """

    # Select desired session
    pref_session = data[session]

    #### Find neuron indices of correct brain regions ####
    # Type check (transform string into single-entry list if necessary)
    if areas is None:
        pass
    elif type(areas) == str:
        areas = [areas]
    elif type(areas) != list:
        raise TypeError('"Areas" argument has to be either a string or list of strings')

    # Initialize mask array
    area_mask = np.zeros(pref_session['spks'].shape[0], dtype=bool)

    # Iterate through all areas and set 'True' at indices of a desired brain area
    if areas is not None:
        for area in areas:
            if area == 'all_vis':
                # Special case: if all visual areas should be selected, look for substring via list comprehension
                area_mask = np.array([1 if 'VIS' in x else 0 for x in pref_session['brain_area']], dtype=bool)
            else:
                area_mask = np.logical_or(area_mask, pref_session['brain_area'] == area)
    else:
        area_mask = np.ones(pref_session['spks'].shape[0], dtype=bool)

    if inverse:
        area_mask = np.invert(area_mask)

    # Get indices of desired time bins through slicing
    if time_bins is not None:
        time_mask = np.zeros(pref_session['spks'].shape[2], dtype=bool)
        time_mask[time_bins[0]:time_bins[1]] = True
    else:
        time_mask = np.ones(pref_session['spks'].shape[2], dtype=bool)

    # Filter data by the masks created above
    spikes = pref_session['spks'][area_mask]               # Select only neurons from the desired brain regions
    spikes = spikes[:, :, time_mask]                       # Select only desired time bins
    pupil_area = pref_session['pupil'][0][:, time_mask]    # Select pupil area (indices 1 and 2 are X/Y positions)
    contrast = pref_session['contrast_right']              # Select contrast value on contralateral hemifield (right)

    return spikes, pupil_area, contrast

#%%

def survey_contrast_modulation(data, thresh_p=0.05, thresh_var=0.05):

    df_list = []
    for sess_idx, session in enumerate(data):
        for region in np.unique(session['brain_area']):
            spikes, pupil_area, contrast = filter_data(data, session=sess_idx, areas=str(region), inverse=False)
            results = ana.get_contrast_modulation(spikes, contrast)
            df_list.append(pd.DataFrame(data={'frac_p': [(sum(results[:, 0] < thresh_p)/len(results))*100],
                                              'frac_var': [(sum(results[:, 1] > thresh_var)/len(results))*100],
                                              'area': str(region)}))
    df = pd.concat(df_list)

    pd_df = df.sort_values('frac_p', ascending=False).reset_index()

    sns.set_style('whitegrid')
    sns.set_context('notebook')
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    out = sns.barplot(data=pd_df, x='area', y='frac_p', ax=ax[0])
    out = sns.barplot(data=pd_df, x='area', y='frac_var', ax=ax[1])
    for item in ax[1].get_xticklabels():
        item.set_rotation(45)
    ax[0].set_ylabel('Contrast-modulated cells\nby p-value [%]', fontsize=15)
    ax[1].set_ylabel('Contrast-modulated cells\nby explained variance [%]', fontsize=15)
    ax[0].set_xlabel('')
#
# print(f'{(sum(results[:, 1]>0.14)/len(results))*100}% of neurons are contrast modulated')