import data_manager as dm
import neuron_analysis as ana
import pupil_data as pup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#%%
data = dm.load_data()
spikes, pupil, contrast = dm.filter_data(data, session=21, areas=['all_vis'], time_bins=(50, 150))
results = ana.get_contrast_modulation(spikes, contrast)

neuron_id = 8

sns.barplot(data=df2, x='contrast', y='fr')

dm.survey_contrast_modulation(data, thresh_p=0.05, thresh_var=0.05)

#%% pupil size

# Look at a few random pupil traces
spikes, pupil, contrast = dm.filter_data(data, session=21, areas=['all_vis'], time_bins=(25, 45))
n_rows = 4
n_cols = 6
rand_ind = np.sort(np.random.choice(len(pupil), size=n_rows*n_cols, replace=False))
fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(16, 9))
count = 0
for row in range(n_rows):
    for col in range(n_cols):
        ax[row, col].plot(pupil[rand_ind[count]])
        ax[row, col].set_title(f'Trial {rand_ind[count]+1}, {int(contrast[rand_ind[count]]*100)}%')
        ax[row, col].axvline(50, color='r')
        count += 1
plt.tight_layout()

# Look at pre-stimulus pupil dynamics in more detail
performance = np.delete(data[21]['feedback_type'], [0, 129], axis=0)
cont = np.delete(contrast, [0,129], axis=0)
pupil_filt = pupil[1:]
pupil_filt = np.delete(pupil_filt, 129, axis=0)
pupil_z = pup.ztransform_pupil_size(pupil_filt)
pup_dat = np.hstack((np.mean(pupil_filt, axis=1), pupil_z))
label = ['pupil']*len(pupil_z) + ['pupil_z']*len(pupil_z)
label2 = [0]*len(pupil_z) + [1]*len(pupil_z)
df = pd.DataFrame({'pupil': pup_dat, 'type': label, 'label': label2, 'correct': np.hstack((performance, performance))})

sns.set_context('talk')
fig, ax = plt.subplots(1, 2, figsize=(15,8))
sns.swarmplot(y='pupil', x='type', hue='correct', data=df[df['type']=='pupil'], ax=ax[0], alpha=0.7)
sns.boxplot(y='pupil', x='type', data=df[df['type']=='pupil'], ax=ax[0], showfliers=False, color='gray', whis=[20, 80])
sns.swarmplot(y='pupil', x='type', hue='correct', data=df[df['type']=='pupil_z'], ax=ax[1], alpha=0.7)
sns.boxplot(y='pupil', x='type', data=df[df['type']=='pupil_z'], ax=ax[1], showfliers=False, color='gray', whis=[20, 80])

# split into 20th, 80th and middle percentile
pupil_low = pupil_z[pupil_z < np.percentile(pupil_z, 20)]
pupil_mid = pupil_z[(pupil_z > np.percentile(pupil_z, 40)) & (pupil_z < np.percentile(pupil_z, 60))]
pupil_high = pupil_z[pupil_z > np.percentile(pupil_z, 80)]

cont_low = cont[pupil_z < np.percentile(pupil_z, 20)]
cont_mid = cont[(pupil_z > np.percentile(pupil_z, 40)) & (pupil_z < np.percentile(pupil_z, 60))]
cont_high = cont[pupil_z > np.percentile(pupil_z, 80)]

perc_labels = np.hstack([['20th']*len(pupil_low), ['40th']*len(pupil_mid), ['80th']*len(pupil_high)])

df2 = pd.DataFrame({'pupil': np.hstack([pupil_low, pupil_mid, pupil_high]),
                    'contrast': np.hstack([cont_low, cont_mid, cont_high])*100,
                    'percentiles': perc_labels})

plt.figure(); sns.swarmplot(x='contrast', y='pupil', hue='percentiles', data=df2)

# split into 40th, and 60th percentile
pupil_low = pupil_z[pupil_z < np.percentile(pupil_z, 30)]
pupil_high = pupil_z[pupil_z > np.percentile(pupil_z, 70)]

cont_low = cont[pupil_z < np.percentile(pupil_z, 30)]*100
cont_high = cont[pupil_z > np.percentile(pupil_z, 70)]*100

perc_labels = np.hstack([['30th']*len(pupil_low), ['70th']*len(pupil_high)])

df3 = pd.DataFrame({'pupil': np.hstack([pupil_low, pupil_high]),
                    'contrast': np.hstack([cont_low, cont_high]),
                    'percentiles': perc_labels})

plt.figure(); sns.swarmplot(x='contrast', y='pupil', hue='percentiles', data=df3)

#%% PVC

# Plot one matrix
spikes, pupil, contrast = dm.filter_data(data, session=21, areas=['all_vis'], time_bins=(50, 75))
act_mat = ana.population_vector_matrix(spikes, contrast)
mat = ana.pvc_matrix(act_mat)
ana.plot_pvc_matrix(mat)

# Try out different time bins
start_bin = 50
end_bin = 150
bin_size = 25
bins = np.arange(start_bin, end_bin+1, bin_size)

fig = plt.figure()
ax = plt.gca()
lines = []
labels = []
for i in range(len(bins)):
    if i == 0:
        spikes, pupil, contrast = dm.filter_data(data, session=21, areas=['all_vis'], time_bins=(min(bins), max(bins)))
        labels.append('total')
    else:
        spikes, pupil, contrast = dm.filter_data(data, session=21, areas=['all_vis'], time_bins=(bins[i-1], bins[i]))
        labels.append((bins[i - 1], bins[i]))
    y, std = ana.population_vector_correlation(spikes, contrast)
    line = ana.plot_pvc_curve(y, std, ax=ax)
    lines.append(line)

plt.legend(lines, labels)


# Try out different time bins across different regions and sessions
start_bin = 50
end_bin = 150
bin_size = 25
bins = np.arange(start_bin, end_bin+1, bin_size)

regions = ['VISp', 'MOp', 'LD', 'RT', 'ZI']
fig, ax = plt.subplots(2, len(regions), sharey='row')

for idx, reg in enumerate(regions):
    lines = []
    labels = []
    for i in range(len(bins)):
        y_sess = []
        y_std = []
        mats = []
        for sess in range(len(data)):
            if i == 0:
                spikes, pupil, contrast = dm.filter_data(data, session=sess, areas=[reg], time_bins=(min(bins), max(bins)))
                if len(spikes) == 0:
                    continue
            else:
                spikes, pupil, contrast = dm.filter_data(data, session=sess, areas=[reg], time_bins=(bins[i-1], bins[i]))
                if len(spikes) == 0:
                    continue

            # Potential: only take contrast modulated neurons (>5% variance)
            # cont_modulation = ana.get_contrast_modulation(spikes, contrast)
            # cont_mask = cont_modulation[:, 1] > 0.05
            # if sum(cont_mask) == 0:
            #     continue
            # spikes = spikes[cont_mask]
            y, std = ana.population_vector_correlation(spikes, contrast)
            y_sess.append(y)
            y_std.append(std)
            mats.append(ana.population_vector_matrix(spikes, contrast))

        y = np.mean(np.stack(y_sess, axis=0), axis=0)
        std = np.std(np.stack(y_std, axis=0), axis=0)
        line = ana.plot_pvc_curve(y, std, ax=ax[0, idx])

        if (reg=='VISp' and i == 1) or (reg=='MOp' and i == 2) or (reg=='LD' and i == 1) or (reg=='RT' and i == 1) or (reg=='ZI' and i == 3):
            mat = ana.pvc_matrix(np.hstack(mats))
            ana.plot_pvc_matrix(mat, ax=ax[1, idx])
            ax[1, idx].title.set_text(f'CorrMat for timebin {(bins[i-1], bins[i])}')

        lines.append(line)
        if i == 0:
            labels.append('total')
        else:
            labels.append((bins[i - 1], bins[i]))
        ax[0, idx].title.set_text(f'{reg} ({len(y_sess)} sess)')

    if idx == len(regions)-1:
        ax[0, idx].legend(lines, labels)
