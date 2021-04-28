import xarray as xr
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import argparse
import preprocess

# %%

units = {'beta_raw': '$\\log (m^{-1} sr^{-1})$', 'v_raw': '$m s^{-1}$',
         'v_raw_averaged': '$m s^{-1}$',
         'beta_averaged': '$\\log (m^{-1} sr^{-1})$',
         'v_error': '$m s^{-1}$'}


def classification_algorithm(file, out_directory, diagnostic=False, xr_data=False):
    Path(out_directory).mkdir(parents=True, exist_ok=True)
    df = xr.open_dataset(file)
    df = df.where(df.range > 90, drop=True)
    df = preprocess.bleed_through(df)

    df['beta_raw'] = df['beta_raw'].where(df['co_signal'] >
                                          (1 + df.attrs['background_snr_sd']))

    classifier = np.zeros(df['beta_raw'].shape, dtype=int)

    log_beta = np.log10(df['beta_raw'])

    if xr_data is True:
        with open('ref_XR2.npy', 'rb') as f:
            ref_XR = np.load(f)
        log_beta[:, :50] = log_beta[:, :50] - ref_XR

    # Aerosol
    aerosol = log_beta < -5.5

    # Small size median filter to remove noise
    aerosol_smoothed = median_filter(aerosol, size=11)
    # Remove thin bridges, better for the clustering
    aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

    classifier[aerosol_smoothed] = 10

    for var in ['beta_raw', 'v_raw', 'depo_bleed']:
        df[var] = df[var].where(df['co_signal'] > (1 + 3*df.attrs['background_snr_sd']))
    log_beta = np.log10(df['beta_raw'])

    if xr_data is True:
        log_beta[:, :50] = log_beta[:, :50] - ref_XR

    range_flat = np.tile(df['range'],
                         df['beta_raw'].shape[0])
    # Liquid
    liquid = log_beta > -5.5

    # maximum filter to increase the size of liquid region
    liquid_max = maximum_filter(liquid, size=5)
    # Median filter to remove background noise
    liquid_smoothed = median_filter(liquid_max, size=13)

    classifier[liquid_smoothed] = 30

    # updraft - indication of aerosol zone
    updraft = df['v_raw'] > 1
    updraft_smooth = median_filter(updraft, size=3)
    updraft_max = maximum_filter(updraft_smooth, size=91)

    # Fill the gap in aerosol zone
    updraft_median = median_filter(updraft_max, size=31)

    # precipitation < -1 (center of precipitation)
    precipitation_1 = (log_beta > -7) & (df['v_raw'] < -1)

    precipitation_1_median = median_filter(precipitation_1, size=9)

    # Only select precipitation outside of aerosol zone
    precipitation_1_ne = precipitation_1_median * ~updraft_median
    precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                                  size=3)
    precipitation = precipitation_1_median_smooth

    # precipitation < -0.5 (include all precipitation)
    precipitation_1_low = (log_beta > -7) & (df['v_raw'] < -0.5)

    # Avoid ebola infection surrounding updraft
    # Useful to contain error during ebola precipitation
    updraft_ebola = df['v_raw'] > 0.2
    updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

    # Ebola precipitation
    for _ in range(1500):
        prep_1_max = maximum_filter(precipitation, size=3)
        prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
        precipitation_ = precipitation_1_low * prep_1_max
        if np.sum(precipitation) == np.sum(precipitation_):
            break
        precipitation = precipitation_

    classifier[precipitation] = 20

    # Remove all aerosol above cloud or precipitation
    mask_aerosol0 = classifier == 10
    for i in np.array([20, 30]):
        if i == 20:
            mask = classifier == i
        else:
            mask = log_beta > -5
            mask = maximum_filter(mask, size=5)
            mask = median_filter(mask, size=13)
        mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
        mask_col = np.nanargmax(mask[mask_row, :], axis=1)
        for row, col in zip(mask_row, mask_col):
            mask[row, col:] = True
        mask_undefined = mask * mask_aerosol0
        classifier[mask_undefined] = i

    if (classifier == 10).any():
        classifier_ = classifier.ravel()
        time_dbscan = np.repeat(np.arange(df['time'].size),
                                df['beta_raw'].shape[1])
        height_dbscan = np.tile(np.arange(df['range'].size),
                                df['beta_raw'].shape[0])

        time_dbscan = time_dbscan[classifier_ == 10].reshape(-1, 1)
        height_dbscan = height_dbscan[classifier_ == 10].reshape(-1, 1)
        X = np.hstack([time_dbscan, height_dbscan])
        db = DBSCAN(eps=3, min_samples=25, n_jobs=-1).fit(X)

        v_dbscan = df['v_raw'].values.ravel()[classifier_ == 10]
        range_dbscan = range_flat[classifier_ == 10]

        v_dict = {}
        r_dict = {}
        for i in np.unique(db.labels_):
            v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
            r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

        lab = db.labels_.copy()
        for key, val in v_dict.items():
            if key == -1:
                lab[db.labels_ == key] = 40
            elif (val < -0.5):
                lab[db.labels_ == key] = 20
            elif r_dict[key] == min(df['range']):
                lab[db.labels_ == key] = 10
            elif (val > -0.2):
                lab[db.labels_ == key] = 10
            else:
                lab[db.labels_ == key] = 40

        classifier[classifier == 10] = lab

    # Separate ground rain
    if (classifier == 20).any():
        classifier_ = classifier.ravel()
        time_dbscan = np.repeat(np.arange(df['time'].size),
                                df['beta_raw'].shape[1])
        height_dbscan = np.tile(np.arange(df['range'].size),
                                df['beta_raw'].shape[0])

        time_dbscan = time_dbscan[classifier_ == 20].reshape(-1, 1)
        height_dbscan = height_dbscan[classifier_ == 20].reshape(-1, 1)
        X = np.hstack([time_dbscan, height_dbscan])
        db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

        range_dbscan = range_flat[classifier_ == 20]

        r_dict = {}
        for i in np.unique(db.labels_):
            r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

        lab = db.labels_.copy()
        for key, val in r_dict.items():
            if r_dict[key] == min(df['range']):
                lab[db.labels_ == key] = 20
            else:
                lab[db.labels_ == key] = 30

        classifier[classifier == 20] = lab

    cmap = mpl.colors.ListedColormap(
        ['white', '#2ca02c', 'blue', 'red', 'gray'])
    boundaries = [0, 10, 20, 30, 40, 50]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    decimal_time = df['time'].dt.hour + \
        df['time'].dt.minute / 60 + df['time'].dt.second/3600

    if diagnostic is True:
        fig, axes = plt.subplots(6, 2, sharex=True, sharey=True,
                                 figsize=(16, 9))
        for val, ax, cmap_ in zip([aerosol, aerosol_smoothed,
                                   liquid_smoothed, precipitation_1_median,
                                   updraft_median,
                                   precipitation_1_median_smooth, precipitation_1_low,
                                   updraft_ebola_max, precipitation],
                                  axes.flatten()[2:-1],
                                  [['white', '#2ca02c'], ['white', '#2ca02c'],
                                   ['white', 'red'], ['white', 'blue'],
                                   ['white', '#D2691E'],
                                   ['white', 'blue'], ['white', 'blue'],
                                   ['white', '#D2691E'], ['white', 'blue']]):
            ax.pcolormesh(decimal_time, df['range'],
                          val.T, cmap=mpl.colors.ListedColormap(cmap_))
        axes.flatten()[-1].pcolormesh(decimal_time, df['range'],
                                      classifier.T,
                                      cmap=cmap, norm=norm)
        axes[0, 0].pcolormesh(decimal_time, df['range'],
                              np.log10(df['beta_raw']).T,
                              cmap='jet', vmin=-8, vmax=-4)
        axes[0, 1].pcolormesh(decimal_time, df['range'],
                              df['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
        fig.tight_layout()
        fig.savefig(out_directory + '/' + df.attrs['file_name'] + '_diagnostic_plot.png',
                    dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(4, 1, figsize=(6, 8))
    ax1, ax3, ax5, ax7 = ax.ravel()
    p1 = ax1.pcolormesh(decimal_time, df['range'],
                        np.log10(df['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
    p2 = ax3.pcolormesh(decimal_time, df['range'],
                        df['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
    p3 = ax5.pcolormesh(decimal_time, df['range'],
                        df['depo_bleed'].T, cmap='jet', vmin=0, vmax=0.5)
    p4 = ax7.pcolormesh(decimal_time, df['range'],
                        classifier.T,
                        cmap=cmap, norm=norm)
    for ax in [ax1, ax3, ax5, ax7]:
        ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
        ax.set_ylabel('Range [km, a.g.l]')

    cbar = fig.colorbar(p1, ax=ax1)
    cbar.ax.set_ylabel('Beta [' + units.get('beta_raw', None) + ']', rotation=90)
    # cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p2, ax=ax3)
    cbar.ax.set_ylabel('Velocity [' + units.get('v_raw', None) + ']', rotation=90)
    # cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p3, ax=ax5)
    cbar.ax.set_ylabel('Depolarization ratio')
    # cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Precipitation', 'Clouds', 'Undefined'])
    ax7.set_xlabel('Time [UTC - hour]')

    fig.tight_layout()
    fig.savefig(out_directory + '/' + df.attrs['file_name'] + '_classified.png',
                dpi=150, bbox_inches='tight')
    plt.close('all')
    df['classified'] = (['time', 'range'], classifier)

    df.attrs['classified'] = 'Clasification algorithm by Vietle \
                                at github.com/vietle94/halo-lidar'
    df.attrs['bleed_corrected'] = 'Bleed through corrected for \
                                depolarization ratio, see Vietle thesis'

    df['depo_bleed'].attrs = {'units': ' ',
                              'long_name': 'Depolarization ratio \
                              (bleed through corrected)',
                              'comments': 'Bleed through corrected'}

    df['depo_bleed_sd'].attrs = {'units': ' ',
                                 'long_name': 'Standard deviation of depolarization \
                              ratio (bleed through corrected)',
                                 'comments': 'Bleed through corrected'}
    df['classified'].attrs = {'units': ' ',
                              'long_name': 'Classified mask',
                              'comments': '0: Background, 10: Aerosol, \
                           20: Precipitation, 30: Clouds, 40: Undefined'}

    df.to_netcdf(out_directory + '/' + df.attrs['file_name'] +
                 '_classified.nc', format='NETCDF3_CLASSIC')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description for arguments")
    parser.add_argument("file_location", help="File location", type=str)
    parser.add_argument("out_directory", help="Output directory", type=str)
    parser.add_argument("-xr", "--XRdevice", help="If data is from Uto-32XR",
                        action='store_true')
    parser.add_argument("-d", "--diagnostic", help="Output a diagnostic image \
                        for classification algorithm",
                        action='store_true')
    argument = parser.parse_args()
    classification_algorithm(argument.file_location, argument.out_directory,
                             diagnostic=argument.diagnostic,
                             xr_data=argument.XRdevice)
