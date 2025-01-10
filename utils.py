import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy.stats import entropy


def real_results(df_train, folder_name):
    folder = os.path.join('Results', folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    cats_1 = ['No PV EV Spring', 'No PV EV Summer', 'No PV EV Fall', 'No PV EV Winter',
             'PV Spring', 'PV Summer', 'PV Fall', 'PV Winter',
             'EV_lv1 Spring', 'EV_lv1 Summer', 'EV_lv1 Fall', 'EV_lv1 Winter',
             'EV_lv2 Spring', 'EV_lv2 Summer', 'EV_lv2 Fall', 'EV_lv2 Winter',
             'EV_both Spring', 'EV_both Summer', 'EV_both Fall', 'EV_both Winter',
             'PV+EV_lv1 Spring', 'PV+EV_lv1 Summer', 'PV+EV_lv1 Fall', 'PV+EV_lv1 Winter',
             'PV+EV_lv2 Spring', 'PV+EV_lv2 Summer', 'PV+EV_lv2 Fall', 'PV+EV_lv2 Winter',
             'PV+EV_both Spring', 'PV+EV_both Summer', 'PV+EV_both Fall', 'PV+EV_both Winter']
    cats_2 = ['Weekday', 'Weekend']
    cats_ = []
    for a in cats_2:
        for b in cats_1:
            cat = f'{a} - {b}'
            cats_.append(cat)

    for cat_num in range(len(cats_)):
        cat = cats_[cat_num]
        df_temp = df_train[df_train['Label'] == cat_num].iloc[:, :-2]
        df_temp.to_csv(os.path.join(folder, cat + '_real.csv'), index=False, header=True)


def generate_results(gan, num_samples, folder_name, min_value, scale):
    folder = os.path.join('Results', folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    cats_1 = ['No PV EV Spring', 'No PV EV Summer', 'No PV EV Fall', 'No PV EV Winter',
              'PV Spring', 'PV Summer', 'PV Fall', 'PV Winter',
              'EV_lv1 Spring', 'EV_lv1 Summer', 'EV_lv1 Fall', 'EV_lv1 Winter',
              'EV_lv2 Spring', 'EV_lv2 Summer', 'EV_lv2 Fall', 'EV_lv2 Winter',
              'EV_both Spring', 'EV_both Summer', 'EV_both Fall', 'EV_both Winter',
              'PV+EV_lv1 Spring', 'PV+EV_lv1 Summer', 'PV+EV_lv1 Fall', 'PV+EV_lv1 Winter',
              'PV+EV_lv2 Spring', 'PV+EV_lv2 Summer', 'PV+EV_lv2 Fall', 'PV+EV_lv2 Winter',
              'PV+EV_both Spring', 'PV+EV_both Summer', 'PV+EV_both Fall', 'PV+EV_both Winter']
    cats_2 = ['Weekday', 'Weekend']
    cats_ = []
    for a in cats_2:
        for b in cats_1:
            cat = f'{a} - {b}'
            cats_.append(cat)

    for cat_num in range(len(cats_)):
        cat = cats_[cat_num]
        df_temp = pd.DataFrame(gan.generate(num_samples, cat_num).detach().numpy())
        df_temp = df_temp*scale + min_value
        df_temp.to_csv(os.path.join(folder, cat + '_gen.csv'), index=False, header=True)

# Plots
def stat_df_dict(folder_name):
    folder = os.path.join('Results', folder_name)

    df_gens = {}
    df_reals = {}

    cats_1 = ['No PV EV Spring', 'No PV EV Summer', 'No PV EV Fall', 'No PV EV Winter',
              'PV Spring', 'PV Summer', 'PV Fall', 'PV Winter',
              'EV_lv1 Spring', 'EV_lv1 Summer', 'EV_lv1 Fall', 'EV_lv1 Winter',
              'EV_lv2 Spring', 'EV_lv2 Summer', 'EV_lv2 Fall', 'EV_lv2 Winter',
              'EV_both Spring', 'EV_both Summer', 'EV_both Fall', 'EV_both Winter',
              'PV+EV_lv1 Spring', 'PV+EV_lv1 Summer', 'PV+EV_lv1 Fall', 'PV+EV_lv1 Winter',
              'PV+EV_lv2 Spring', 'PV+EV_lv2 Summer', 'PV+EV_lv2 Fall', 'PV+EV_lv2 Winter',
              'PV+EV_both Spring', 'PV+EV_both Summer', 'PV+EV_both Fall', 'PV+EV_both Winter']
    cats_2 = ['Weekday', 'Weekend']
    cats_ = []
    for a in cats_2:
        for b in cats_1:
            cat = f'{a} - {b}'
            cats_.append(cat)

    for label in cats_:
        df_gen = pd.read_csv(os.path.join(folder, f'{label}_gen.csv'))
        df_real = pd.read_csv(os.path.join(folder, f'{label}_real.csv'))
        df_gens[label] = df_gen
        df_reals[label] = df_real

    return df_gens, df_reals


def compute_fid(features_real, features_generated):
    features_real = features_real.reshape(-1, 1)
    features_generated = features_generated.reshape(-1, 1)
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_generated, axis=0)
    sigma_real = np.cov(features_real.T)
    sigma_gen = np.cov(features_generated.T)
    sigma_real = np.atleast_2d(sigma_real)
    sigma_gen = np.atleast_2d(sigma_gen)
    diff = mu_real - mu_gen
    try:
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean))
    except ValueError:
        fid = float(diff.dot(diff) + np.abs(np.sqrt(sigma_real) - np.sqrt(sigma_gen))[0, 0] ** 2)
    return fid


def stats_plot(folder_name):
    matplotlib.use('TkAgg')
    plt.rcParams['figure.figsize'] = [18, 9]

    df_gens, df_reals = stat_df_dict(folder_name)
    folder = os.path.join('Results', folder_name)

    generate_stats_df = {}
    real_stats_df = {}

    cats_1 = ['No PV EV Spring', 'No PV EV Summer', 'No PV EV Fall', 'No PV EV Winter',
              'PV Spring', 'PV Summer', 'PV Fall', 'PV Winter',
              'EV_lv1 Spring', 'EV_lv1 Summer', 'EV_lv1 Fall', 'EV_lv1 Winter',
              'EV_lv2 Spring', 'EV_lv2 Summer', 'EV_lv2 Fall', 'EV_lv2 Winter',
              'EV_both Spring', 'EV_both Summer', 'EV_both Fall', 'EV_both Winter',
              'PV+EV_lv1 Spring', 'PV+EV_lv1 Summer', 'PV+EV_lv1 Fall', 'PV+EV_lv1 Winter',
              'PV+EV_lv2 Spring', 'PV+EV_lv2 Summer', 'PV+EV_lv2 Fall', 'PV+EV_lv2 Winter',
              'PV+EV_both Spring', 'PV+EV_both Summer', 'PV+EV_both Fall', 'PV+EV_both Winter']
    cats_2 = ['Weekday', 'Weekend']
    cats_ = []
    for a in cats_2:
        for b in cats_1:
            cat = f'{a} - {b}'
            cats_.append(cat)

    for label in cats_:

        generate_peak = []
        generate_base = []
        generate_peakduration = []
        generate_rise = []
        generate_fall = []
        real_peak = []
        real_base = []
        real_peakduration = []
        real_rise = []
        real_fall = []

        real_data = df_reals[label]
        generate_data = df_gens[label]
        for generate_i in range(len(generate_data)):
            generate_peak_i = generate_data.iloc[generate_i].sort_values().iloc[22]
            generate_base_i = generate_data.iloc[generate_i].sort_values().iloc[1]
            generate_mean = (generate_peak_i + generate_base_i) / 2
            generate_peakduration_i = len(
                generate_data.iloc[generate_i][generate_data.iloc[generate_i] > generate_mean])
            generate_rise_i = abs(int(abs(generate_data.iloc[generate_i] - generate_peak_i).idxmax()) -
                                  int(abs(generate_data.iloc[generate_i] - generate_peak_i).idxmin()))
            generate_fall_i = abs(int(abs(generate_data.iloc[generate_i] - generate_base_i).idxmax()) -
                                  int(abs(generate_data.iloc[generate_i] - generate_base_i).idxmin()))
            generate_peak.append(generate_peak_i)
            generate_base.append(generate_base_i)
            generate_peakduration.append(generate_peakduration_i)
            generate_rise.append(generate_rise_i)
            generate_fall.append(generate_fall_i)

        for real_i in range(len(real_data)):
            real_peak_i = real_data.iloc[real_i].sort_values().iloc[22]
            real_base_i = real_data.iloc[real_i].sort_values().iloc[1]
            real_mean = (real_peak_i + real_base_i) / 2
            real_peakduration_i = len(real_data.iloc[real_i][real_data.iloc[real_i] > real_mean])
            real_rise_i = abs(int(abs(real_data.iloc[real_i] - real_peak_i).idxmax()) -
                              int(abs(real_data.iloc[real_i] - real_peak_i).idxmin()))
            real_fall_i = abs(int(abs(real_data.iloc[real_i] - real_base_i).idxmax()) -
                              int(abs(real_data.iloc[real_i] - real_base_i).idxmin()))
            real_peak.append(real_peak_i)
            real_base.append(real_base_i)
            real_peakduration.append(real_peakduration_i)
            real_rise.append(real_rise_i)
            real_fall.append(real_fall_i)

        generate_data['Peak_load'] = generate_peak
        generate_data['Base_load'] = generate_base
        generate_data['Peak_duration'] = generate_peakduration
        generate_data['Rise'] = generate_rise
        generate_data['Fall'] = generate_fall

        real_data['Peak_load'] = real_peak
        real_data['Base_load'] = real_base
        real_data['Peak_duration'] = real_peakduration
        real_data['Rise'] = real_rise
        real_data['Fall'] = real_fall

        generate_stats_df[label] = generate_data
        real_stats_df[label] = real_data

    generate_stat_df = pd.DataFrame()
    real_stat_df = pd.DataFrame()
    dif_stat_df = pd.DataFrame()
    for label in cats_:

        generate_stat = generate_stats_df[label].iloc[:, -5:]
        real_stat = real_stats_df[label].iloc[:, -5:]
        generate_stat_mean = generate_stat.mean()
        generate_stat_std = generate_stat.std()
        real_stat_mean = real_stat.mean()
        real_stat_std = real_stat.std()
        dif_mean = generate_stat_mean - real_stat_mean
        dif_std = generate_stat_std - real_stat_std
        generate_stat_df[f'{label}_mean'] = generate_stat_mean
        generate_stat_df[f'{label}_std'] = generate_stat_std
        real_stat_df[f'{label}_mean'] = real_stat_mean
        real_stat_df[f'{label}_std'] = real_stat_std
        dif_stat_df[f'{label}_mean'] = dif_mean
        dif_stat_df[f'{label}_std'] = dif_std

    temp_df = dif_stat_df.copy()
    temp_dif_df = pd.DataFrame()
    for label in cats_:

        temp_mean = temp_df[f'{label}_mean'].tolist()
        temp_std = temp_df[f'{label}_std'].tolist()
        temp_stat = temp_mean + temp_std
        temp_dif_df[label] = temp_stat
    temp_dif_df.index = ['PeakLoad_mean', 'BaseLoad_mean', 'PeakDuration_mean', 'RiseTime_mean', 'FallTime_mean',
                         'PeakLoad_std', 'BaseLoad_std', 'PeakDuration_std', 'RiseTime_std', 'FallTime_std']

    temp_dif_df.to_csv(os.path.join(folder, 'stat_dif_plot.csv'), index=True, header=True)

    t1 = pd.read_csv(os.path.join(folder, 'stat_dif_plot.csv'))
    t1.rename(columns={'Unnamed: 0': 'Key parameters'}, inplace=True)
    t1_mean = t1.iloc[:5, :]
    t1_std = t1.iloc[5:, :]
    t1_new = pd.DataFrame()
    for row_idx in range(len(t1_mean)):
        mean_temp = t1_mean.iloc[row_idx]
        std_temp = t1_std.iloc[row_idx]
        t1_new = pd.concat([t1_new, mean_temp, std_temp], axis=1)
    t1_new = t1_new.T
    t1_new.index = t1_new.iloc[:, 0]
    t1_new = t1_new.iloc[:, 1:]
    t1_new = t1_new.astype('float64')

    # Plot stats difference
    sns.set(font_scale=1.4)
    t1_new_weekday = t1_new.iloc[:, :32]
    t1_new_weekend = t1_new.iloc[:, 32:]
    t1_new_weekday.columns = cats_1
    t1_new_weekend.columns = cats_1

    cm = sns.heatmap(t1_new, cmap='vlag', vmin=-2, vmax=2)
    plt.tight_layout()
    # plt.show()
    cm.figure.savefig(os.path.join(folder, 'temp_plot.png'))
    plt.close()

    cm = sns.heatmap(t1_new_weekday, cmap='vlag', vmin=-1.5, vmax=1.5)
    plt.tight_layout()
    # plt.show()
    cm.figure.savefig(os.path.join(folder, 'temp_plot_weekday.png'))
    plt.close()

    cm = sns.heatmap(t1_new_weekend, cmap='vlag', vmin=-1.5, vmax=1.5)
    plt.tight_layout()
    # plt.show()
    cm.figure.savefig(os.path.join(folder, 'temp_plot_weekend.png'))
    plt.close()

    # Plot KLD
    kl_stat_df = pd.DataFrame(index=['PeakLoad', 'BaseLoad', 'PeakDuration', 'RiseTime', 'FallTime'])
    for label in cats_:
        generate_file = generate_stats_df[label]
        real_file = real_stats_df[label]
        generate_stat = generate_file.iloc[:, -5:]
        real_stat = real_file.iloc[:, -5:]
        # Normalize generate_stat and real_stat (based on real_stat)
        generate_stat_normalized = generate_stat.div(real_stat.max(axis=0), axis=1)
        real_stat_normalized = real_stat.div(real_stat.max(axis=0), axis=1)
        kld_values = []
        for column in generate_stat.columns:
            kld = rel_entr(real_stat_normalized[column].sort_values(), generate_stat_normalized[column].sort_values()).dropna()
            kld.replace([np.inf, -np.inf], np.nan, inplace=True)
            kld.dropna(inplace=True)
            kld_values.append(abs(kld.mean()))
        kl_stat_df[label] = kld_values

    sns.set(font_scale=1.4)
    cm = sns.heatmap(kl_stat_df, cmap='Blues', vmin=0, vmax=1)
    plt.tight_layout()
    cm.figure.savefig(os.path.join(folder, 'temp_KLD.png'))
    plt.close()

    kl_stat_df_weekday = kl_stat_df.iloc[:, :32]
    kl_stat_df_weekend = kl_stat_df.iloc[:, 32:]
    kl_stat_df_weekday.columns = cats_1
    kl_stat_df_weekend.columns = cats_1

    cm = sns.heatmap(kl_stat_df_weekday, cmap='Blues', vmin=0, vmax=1)
    plt.tight_layout()
    cm.figure.savefig(os.path.join(folder, 'temp_KLD_weekday.png'))
    plt.close()

    cm = sns.heatmap(kl_stat_df_weekend, cmap='Blues', vmin=0, vmax=1)
    plt.tight_layout()
    cm.figure.savefig(os.path.join(folder, 'temp_KLD_weekend.png'))
    plt.close()

    # Plot generated results and real results
    # Separate weekday and weekend
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [16, 24]
    # Weekday
    figure, ax = plt.subplots(8, 4, sharex=True, sharey=True)
    for label_num in range(len(cats_[:32])):
        label = cats_[label_num]
        generate_loads = df_gens[label].iloc[:, :-5]
        real_loads = df_reals[label].iloc[:, :-5]

        row_idx = label_num // 4
        col_idx = label_num % 4

        ax[row_idx, col_idx].plot(real_loads.mean(), color='#F24405', lw=5, label='Real')
        ax[row_idx, col_idx].plot(generate_loads.mean(), color='#008F8C', lw=5, label='Generated')
        if col_idx == 0:
            ax[row_idx, col_idx].set_ylabel(f'{label[10:-7]}\n\nLoad (kWh)', fontsize=14)
        ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=14)

    ax[0, 0].set_title('Spring', fontsize=16)
    ax[0, 1].set_title('Summer', fontsize=16)
    ax[0, 2].set_title('Fall', fontsize=16)
    ax[0, 3].set_title('Winter', fontsize=16)
    ax[7, 0].set_xticks(np.arange(0, 25, 3), ['0', '3', '6', '9', '12', '15', '18', '21', '24'])
    ax[7, 0].set_xlabel('Time of day', fontsize=14)
    ax[7, 1].set_xlabel('Time of day', fontsize=14)
    ax[7, 2].set_xlabel('Time of day', fontsize=14)
    ax[7, 3].set_xlabel('Time of day', fontsize=14)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    # figure.show()
    figure.savefig(os.path.join(folder, 'temp_Loads_weekday.png'))
    plt.close(figure)

    # Weekend
    figure, ax = plt.subplots(8, 4, sharex=True, sharey=True)
    cats_weekend = cats_[32:]
    for label_num in range(len(cats_weekend)):
        label = cats_weekend[label_num]
        generate_loads = df_gens[label].iloc[:, :-5]
        real_loads = df_reals[label].iloc[:, :-5]

        row_idx = label_num // 4
        col_idx = label_num % 4

        ax[row_idx, col_idx].plot(real_loads.mean(), color='#F24405', lw=5, label='Real')
        ax[row_idx, col_idx].plot(generate_loads.mean(), color='#008F8C', lw=5, label='Generated')
        if col_idx == 0:
            ax[row_idx, col_idx].set_ylabel(f'{label[10:-7]}\n\nLoad (kWh)', fontsize=14)
        ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=14)
    ax[0, 0].set_title('Spring', fontsize=16)
    ax[0, 1].set_title('Summer', fontsize=16)
    ax[0, 2].set_title('Fall', fontsize=16)
    ax[0, 3].set_title('Winter', fontsize=16)
    ax[7, 0].set_xticks(np.arange(0, 25, 3), ['0', '3', '6', '9', '12', '15', '18', '21', '24'])
    ax[7, 0].set_xlabel('Time of day', fontsize=14)
    ax[7, 1].set_xlabel('Time of day', fontsize=14)
    ax[7, 2].set_xlabel('Time of day', fontsize=14)
    ax[7, 3].set_xlabel('Time of day', fontsize=14)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    # figure.show()
    figure.savefig(os.path.join(folder, 'temp_Loads_weekend.png'))
    plt.close(figure)

    # Generate paper plots
    plt.rcParams['figure.figsize'] = [18, 9]
    cats_peak = ['Weekday - PV+EV_both Summer', 'Weekday - No PV EV Fall', 'Weekend - EV_lv2 Summer', 'Weekend - EV_both Winter']
    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
    for i in range(len(cats_peak)):
        label = cats_peak[i]
        generate_file = generate_stats_df[label]
        real_file = real_stats_df[label]
        generate_stat = generate_file.iloc[:, -5:]
        real_stat = real_file.iloc[:, -5:]
        peak_real = real_stat['Peak_load']
        peak_gen = generate_stat['Peak_load']

        row_idx = i // 2
        col_idx = i % 2

        bins = np.histogram(np.hstack((peak_real, peak_gen)), bins=15)[1]
        sns.set_style('white')
        sns.histplot(peak_real, color='#F24405', label='Real', stat='percent', bins=bins, kde=True, line_kws={'lw': 5}, ax=ax[row_idx, col_idx])
        sns.histplot(peak_gen, color='#008F8C', label='Generated', stat='percent', bins=bins, kde=True, line_kws={'lw': 5}, ax=ax[row_idx, col_idx])
        ax[row_idx, col_idx].set_title(f'{label}', fontsize=18)
        ax[row_idx, col_idx].set_xlabel('Peak load', fontsize=16)
        if col_idx == 0:
            ax[row_idx, col_idx].set_ylabel('Frequency (%)', fontsize=16)

        ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=16)

        plt.grid(False)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    # fig.show()
    fig.savefig(os.path.join(folder, 'temp_peak_example.png'))
    plt.close(fig)

    cats_base = ['Weekday - EV_lv1 Spring', 'Weekday - PV+EV_lv1 Spring', 'Weekend - EV_lv2 Summer', 'Weekend - PV+EV_both Fall']
    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
    for i in range(len(cats_base)):
        label = cats_base[i]
        generate_file = generate_stats_df[label]
        real_file = real_stats_df[label]
        generate_stat = generate_file.iloc[:, -5:]
        real_stat = real_file.iloc[:, -5:]
        base_real = real_stat['Base_load']
        base_gen = generate_stat['Base_load']

        row_idx = i // 2
        col_idx = i % 2

        bins_ = np.histogram(np.hstack((base_real, base_gen)), bins=15)[1]
        sns.set_style('white')
        sns.histplot(base_real, color='#F24405', label='Real', stat='percent', bins=bins_, kde=True, line_kws={'lw': 5}, ax=ax[row_idx, col_idx])
        sns.histplot(base_gen, color='#008F8C', label='Generated', stat='percent', bins=bins_, kde=True, line_kws={'lw': 5}, ax=ax[row_idx, col_idx])
        ax[row_idx, col_idx].set_title(f'{label}', fontsize=18)
        ax[row_idx, col_idx].set_xlabel('Base load', fontsize=16)
        if col_idx == 0:
            ax[row_idx, col_idx].set_ylabel('Frequency (%)', fontsize=16)
        ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=16)
        plt.grid(False)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    # fig.show()
    fig.savefig(os.path.join(folder, 'temp_base_example.png'))
    plt.close(fig)

    # Calculate FID scores
    fid_df = pd.DataFrame(index=['PeakLoad', 'BaseLoad', 'PeakDuration', 'RiseTime', 'FallTime'])
    for label in cats_:
        generate_file = generate_stats_df[label]
        real_file = real_stats_df[label]
        generate_stat = generate_file.iloc[:, -5:]
        real_stat = real_file.iloc[:, -5:]

        fid_values = []
        for col in generate_stat.columns:
            fid_col = compute_fid(real_stat[col].to_numpy(), generate_stat[col].to_numpy())
            fid_values.append(fid_col)
        fid_df[label] = fid_values

    sns.set(font_scale=1.4)
    cm1 = sns.heatmap(fid_df, cmap='cubehelix_r', vmin=0, vmax=10, linewidths=1.2)
    plt.tight_layout()
    # plt.show()
    cm1.figure.savefig(os.path.join(folder, 'temp_FID.png'))
    plt.close()

    fid_stat_df_weekday = fid_df.iloc[:, :32]
    fid_stat_df_weekend = fid_df.iloc[:, 32:]
    fid_stat_df_weekday.columns = cats_1
    fid_stat_df_weekend.columns = cats_1

    cm1 = sns.heatmap(fid_stat_df_weekday, cmap='cubehelix_r', vmin=0, vmax=10, linewidths=1.2)
    plt.tight_layout()
    cm1.figure.savefig(os.path.join(folder, 'temp_FID_weekday.png'))
    plt.close()

    cm1 = sns.heatmap(fid_stat_df_weekend, cmap='cubehelix_r', vmin=0, vmax=10, linewidths=1.2)
    plt.tight_layout()
    cm1.figure.savefig(os.path.join(folder, 'temp_FID_weekend.png'))
    plt.close()
