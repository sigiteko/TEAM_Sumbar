import argparse
import os
import numpy as np
import json
import pickle
import sklearn.metrics as metrics

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import seaborn as sns
from geopy.distance import geodesic
import h5py
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
#from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import models
import loader
import util
import plots
from models import EnsembleEvaluateModel
from util import generator_from_config
import json

import time as tm
import psutil
import pynvml


#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.compat.v1.Session(config=config))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


EARTH_RADIUS = 6371

sns.set(font_scale=1.5)
sns.set_style('ticks')

import os

# Daftar kandidat kolom untuk masing-masing koordinat
candidates = {
    'LAT': ['Latitude(°)', 'Latitude'],
    'LON': ['Longitude(°)', 'Longitude'],
    'DEPTH': ['JMA_Depth(km)', 'Depth(km)', 'Depth/Km', 'Depth']
}

def select_columns(df, candidates):
    """Memilih kolom yang tersedia berdasarkan kandidat yang diberikan."""
    selected_columns = {}
    for key, options in candidates.items():
        for col in options:
            if col in df.columns:
                selected_columns[key] = col
                break
    return selected_columns


def predict_at_time(model, time, data, event_metadata, batch_size, config, sampling_rate=100, pga=False,
                    use_multiprocessing=True, no_event_token=False, dataset_id=None):
    generator = generator_from_config(config, data, event_metadata, time, batch_size, sampling_rate, pga, dataset_id=dataset_id)



    if pga:
        # Assume pga output is at index 2
        workers = 1
        if use_multiprocessing:
            workers = 10
        predictions = model.predict_generator(generator, workers=workers, use_multiprocessing=use_multiprocessing)
        #predictions = model.predict(generator)        
        if no_event_token:
            mag_pred = []
            loc_pred = []
            predictions = [predictions]
        else:
            mag_pred = predictions[0][generator.reverse_index[:-1]]
            loc_pred = predictions[1][generator.reverse_index[:-1]]

        pga_pred = []
        pga_pred_idx = 2 - 2 * no_event_token
        for i, (start, end) in enumerate(zip(generator.reverse_index[:-1], generator.reverse_index[1:])):
            sample_pga_pred = predictions[pga_pred_idx][start:end].reshape((-1,) + predictions[pga_pred_idx].shape[-2:])
            sample_pga_pred = sample_pga_pred[:len(generator.pga[i])]
            pga_pred += [sample_pga_pred]

        return mag_pred, loc_pred, pga_pred
    else:
        #pred = model.predict_generator(generator, workers=10, use_multiprocessing=True)
        pred = model.predict(generator)
        if no_event_token:
            return ([], []) + pred
        else:
            return pred


def calc_mag_stats(mag_pred, event_metadata, key):
    mean_mag = np.sum(mag_pred[:, :, 0] * mag_pred[:, :, 1], axis=1)
    true_mag = event_metadata[key].values
    r2 = metrics.r2_score(true_mag, mean_mag)
    rmse = np.sqrt(metrics.mean_squared_error(true_mag, mean_mag))
    mae = metrics.mean_absolute_error(true_mag, mean_mag)
    return r2, rmse, mae


def calc_pga_stats(pga_pred, pga_true):
    if len(pga_pred) == 0:
        return np.nan, np.nan, np.nan
    else:
        pga_pred = np.concatenate(pga_pred, axis=0)
        mean_pga = np.sum(pga_pred[:, :, 0] * pga_pred[:, :, 1], axis=1)
        pga_true = np.concatenate(pga_true, axis=0)
        #mask = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
        mask_valid = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
        mask_nonzero = pga_true != 0
        mask = np.logical_and(mask_valid, mask_nonzero)
        pga_true = pga_true[mask]
        mean_pga = mean_pga[mask]
        r2 = metrics.r2_score(pga_true, mean_pga)
        rmse = np.sqrt(metrics.mean_squared_error(pga_true, mean_pga))
        mae = metrics.mean_absolute_error(pga_true, mean_pga)
        return r2, rmse, mae


def calc_loc_stats(loc_pred, event_metadata, pos_offset):
    coord_keys = util.detect_location_keys(event_metadata.columns)
    true_coords = event_metadata[coord_keys].values
    mean_coords = np.sum(loc_pred[:, :, :1] * loc_pred[:, :, 1:4], axis=1)

    mean_coords *= 100
    mean_coords[:, :2] /= util.D2KM
    mean_coords[:, 0] += pos_offset[0]
    mean_coords[:, 1] += pos_offset[1]

    dist_epi = np.zeros(len(mean_coords))
    dist_hypo = np.zeros(len(mean_coords))
    for i, (pred_coord, true_coord) in enumerate(zip(mean_coords, true_coords)):
        dist_epi[i] = geodesic(pred_coord[:2], true_coord[:2]).km
        dist_hypo[i] = np.sqrt(dist_epi[i] ** 2 + (pred_coord[2] - true_coord[2]) ** 2)

    rmse_epi = np.sqrt(np.mean(dist_epi ** 2))
    mae_epi = np.mean(np.abs(dist_epi))

    rmse_hypo = np.sqrt(np.mean(dist_hypo ** 2))
    mae_hypo = np.mean(dist_hypo)
    return rmse_hypo, mae_hypo, rmse_epi, mae_epi


def generate_true_pred_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
#    _, cbar = plots.true_predicted(true_values, pred_values, time=time, agg='mean', quantile=True, ax=ax)
    _, cbar = plots.true_predicted(true_values, pred_values, agg='mean', quantile=True, ax=ax)
    #cax = fig.colorbar(cbar)
    #cax.set_label('Quantile')
    #ax.set_title(f"True vs Prediction Plot for Time {time} s")
    ax.set_title(f"No Upsampling", fontsize=30)
    fig.savefig(os.path.join(path, f'truepred_{suffix}{time}.png'), bbox_inches='tight',dpi=400)
    plt.close(fig)


def generate_calibration_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    plots.calibration_plot(pred_values, true_values, ax=ax)
    ax.set_xlabel('<-- Overestimate       Underestimate -->')
    ax.set_title(f"Calibration Plot for Time {time} s")
    fig.savefig(os.path.join(path, f'quantiles_{suffix}{time}.png'), bbox_inches='tight')
    plt.close(fig)


def calculate_warning_times(config, model, data, event_metadata, batch_size, sampling_rate=100,
                            times=np.arange(0.5, 25, 0.2), alpha=(0.3, 0.4, 0.5, 0.6, 0.7), use_multiprocessing=True,
                            no_event_token=False, dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']

    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None

    alpha = np.array(alpha)

    if isinstance(training_params['data_path'], list):
        if dataset_id is not None:
            training_params['data_path'] = training_params['data_path'][dataset_id]
        else:
            training_params['data_path'] = training_params['data_path'][0]

    f = h5py.File(training_params['data_path'], 'r')
    g_data = f['data']
    #thresholds = f['metadata']['pga_thresholds'].value
    #time_before = f['metadata']['time_before'].value
    thresholds = f['metadata']['pga_thresholds'][()]
    time_before = f['metadata']['time_before'][()]


    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')

    if 'KiK_File' in event_metadata.columns:
        event_key = 'KiK_File'
    else:
        event_key = '#EventID'

    full_predictions = []
    coord_keys = util.detect_location_keys(event_metadata.columns)

    for i, _ in tqdm(enumerate(event_metadata.iterrows()), total=len(event_metadata)):
        event = event_metadata.iloc[i]
        event_metadata_tmp = event_metadata.iloc[i:i+1]
        data_tmp = {key: val[i:i+1] for key, val in data.items()}
        generator_params['translate'] = False
        generator = util.PreloadedEventGenerator(data=data_tmp,
                                                 event_metadata=event_metadata_tmp,
                                                 coords_target=True,
                                                 cutout=(0, 3000),
                                                 pga_targets=n_pga_targets,
                                                 max_stations=max_stations,
                                                 sampling_rate=sampling_rate,
                                                 select_first=True,
                                                 shuffle=False,
                                                 pga_mode=True,
                                                 **generator_params)

        cutout_generator = util.CutoutGenerator(generator, times, sampling_rate=sampling_rate)

        # Assume PGA output at index 2
        workers = 1
        if use_multiprocessing:
            workers = 10
        predictions = model.predict_generator(cutout_generator, workers=workers, use_multiprocessing=use_multiprocessing)
        if no_event_token:
            pga_pred = predictions
        else:
            pga_pred = predictions[2]
        pga_pred = pga_pred.reshape((len(times), -1) + pga_pred.shape[2:])
        pga_pred = pga_pred[:, :len(generator.pga[0])]  # Remove padding stations

        pga_times_pre = np.zeros((pga_pred.shape[1], thresholds.shape[0], alpha.shape[0]), dtype=int)

        for j, log_level in enumerate(np.log10(thresholds * 9.81)):
            prob = np.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                axis=-1)
            prob = prob.reshape(prob.shape + (1,))
            exceedance = prob > alpha  # Shape: times, stations, 1
            exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode='constant')
            pga_times_pre[:, j] = np.argmax(exceedance, axis=0)

        pga_times_pre -= 1
        pga_times_pred = np.zeros_like(pga_times_pre, dtype=float)
        pga_times_pred[pga_times_pre == -1] = np.nan
        pga_times_pred[pga_times_pre > -1] = times[pga_times_pre[pga_times_pre > -1]]

        g_event = g_data[str(event[event_key])]
        #pga_times_true_pre = g_event['pga_times'].value
        pga_times_true_pre = g_event['pga_times'][()]

        pga_times_true = np.zeros_like(pga_times_true_pre, dtype=float)
        pga_times_true[pga_times_true_pre == 0] = np.nan
        pga_times_true[pga_times_true_pre != 0] = pga_times_true_pre[pga_times_true_pre != 0] / sampling_rate - time_before

        #coords = g_event['coords'].value
        coords = g_event['coords'][()]                    # Array

        coords_event = event[coord_keys]

        dist = np.zeros(coords.shape[0])
        for j, station_coords in enumerate(coords):
            dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
        #dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)  # Epi- to hypocentral distance
        dist = np.sqrt(dist ** 2 + coords_event.iloc[2] ** 2)  # Explicitly access the 3rd element

        full_predictions += [(pga_times_pred, pga_times_true, dist)]

    return full_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    #parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')
    #default_time = ','.join([f"{x:.1f}" for x in np.arange(0.5, 39.0, 0.5)])
    #parser.add_argument('--times', type=str, default=default_time)        
    parser.add_argument('--times', type=str, default='0.5,1,2,5,10,15,22,39')
    #parser.add_argument('--times', type=str, default='22')    
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)  # Has only performance implications
    parser.add_argument('--test', action='store_true')  # Evaluate on test set
    parser.add_argument('--pga', action='store_true')  # Evaluate PGA
    parser.add_argument('--n_pga_targets', type=int)  # Overwrite number of PGA targets
    parser.add_argument('--head_times', action='store_true')  # Evaluate warning times
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    default_alpha = ','.join([f"{x:.2f}" for x in np.arange(0.05, 1.00, 0.05)])
    parser.add_argument('--alpha', type=str, default=default_alpha)
    #parser.add_argument('--alpha', type=str, default='0.05,0,1,0.3,0.4,0.5,0.6,0.7,0.8,0.9')  # Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--dataset_id', type=int)  # ID of dataset to evaluate on, in case of joint training
    parser.add_argument('--wait_file', type=str)  # Wait for this file to exist before starting evaluation
    parser.add_argument('--ensemble_member', action='store_true')  # Task to evaluate is an ensemble member
                                                                   # (not the full ensembel)
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    # A combination of tensorflow multiprocessing for generators and pandas dataframes causes the code to deadlock
    # sometimes. This flag provides a workaround.
    parser.add_argument('--no_multiprocessing', action='store_true')
    args = parser.parse_args()
      
    
    if args.wait_file is not None:
        util.wait_for_file(args.wait_file)

    if args.test:
        # raise ValueError('Do you really want to look at the test set?')
        print('WARNING: Test set')

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    if (args.dataset_id is None) and (isinstance(training_params['data_path'], list) and
                                      len(training_params['data_path']) > 1):
        raise ValueError('dataset_id needs to be set for experiments with multiple input data sets.')
    if (args.dataset_id is not None) and not (isinstance(training_params['data_path'], list) and
                                              len(training_params['data_path']) > 1):
        raise ValueError('dataset_id may only be set for experiments with multiple input data sets.')

    if args.dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[args.dataset_id]
        data_path = training_params['data_path'][args.dataset_id]
        n_datasets = len(training_params['data_path'])
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]
        data_path = training_params['data_path']
        n_datasets = 1
    key = generator_params.get('key', 'MA')
    pos_offset = generator_params.get('pos_offset', (-21, -69))
    pga_key = generator_params.get('pga_key', 'pga')
    no_event_token = config['model_params'].get('no_event_token', False)

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.dataset_id is not None:
        suffix += f'_{args.dataset_id}'

    if args.test:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test')
        test_set = True
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'dev')
        test_set = False

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    shuffle_train_dev = generator_params.get('shuffle_train_dev', False)
    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(data_path,
                                                        parts=(False, not test_set, test_set),
                                                        shuffle_train_dev=shuffle_train_dev,
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate)

    #if test_set:
    #    eventid_output_path = os.path.join(output_dir, 'test_eventids.csv')
    #    if '#EventID' in event_metadata.columns:
    #        event_metadata.sample(frac=1, random_state=42)[['#EventID']].to_csv(eventid_output_path, index=False)
    #        print(f"✅ Daftar EventID (acak) disimpan ke: {eventid_output_path}")
    #    else:
    #        print("❌ Kolom '#EventID' tidak ditemukan dalam event_metadata.")                                                        

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if pga_key in data:
        pga_true = data[pga_key]
    else:
        pga_true = None

    if 'max_stations' not in config['model_params']:
        config['model_params']['max_stations'] = data['waveforms'].shape[1]
    if args.max_stations is not None:
        config['model_params']['max_stations'] = args.max_stations

    if args.n_pga_targets is not None:
        if config['model_params'].get('n_pga_targets', 0) > 0:
            print('Overwriting number of PGA targets')
            config['model_params']['n_pga_targets'] = args.n_pga_targets
        else:
            print('PGA flag is set, but model does not support PGA')

    ensemble = config.get('ensemble', 1)
    if ensemble > 1 and not args.ensemble_member:
        model = EnsembleEvaluateModel(config, loss_limit=args.loss_limit)
        model.load_weights(args.experiment_path)
    else:
        if 'n_datasets' in config['model_params']:
            del config['model_params']['n_datasets']
        _, model = models.build_transformer_model(**config['model_params'], trace_length=data['waveforms'][0].shape[1],
                                                  n_datasets=n_datasets)

        if args.weight_file is not None:
            weight_file = os.path.join(args.experiment_path, args.weight_file)
        else:
            weight_file = sorted([x for x in os.listdir(args.experiment_path) if x[:5] == 'event'])[-1]
            weight_file = os.path.join(args.experiment_path, weight_file)

        model.load_weights(weight_file)

    mag_stats = []
    loc_stats = []
    pga_stats = []
    mag_pred_full = []
    loc_pred_full = []
    pga_pred_full = []
    
    #print("dataset_id:", args.dataset_id)
    rmse_pga_list = []
    mae_pga_list = []
    
    # Daftar event ID yang ingin dievaluasi
    selected_event_ids = {
    "2022-02-25T013928_6.2",
    "2023-04-24T200054_6.9",
    "2024-02-23T074628_5.3"
    }
    summary_rows = []

    
    for time in times:
        print(f'Evaluating at time: {time} s')
        pred = predict_at_time(model, time, data, event_metadata,
                               config=config,
                               batch_size=args.batch_size,
                               pga=args.pga,
                               use_multiprocessing=not args.no_multiprocessing,
                               no_event_token=no_event_token,
                               sampling_rate=metadata['sampling_rate'],
                               dataset_id=args.dataset_id)
        

        mag_pred = pred[0]
        loc_pred = pred[1]
        

            
                                                
        if args.pga:
            pga_pred = pred[2]
        else:
            pga_pred = []


        if args.pga:
            r2_pga, rmse_pga, mae_pga = calc_pga_stats(pga_pred, pga_true)
            pga_stats.append((r2_pga, rmse_pga, mae_pga))
            print(f"PGA: R2={r2_pga:.3f}, RMSE={rmse_pga:.3f}, MAE={rmse_pga:.3f}")     
        else:
            # Pilih kolom terbaik yang tersedia di event_metadata
            selected_cols = select_columns(event_metadata, candidates)
            pred_coords = np.sum(loc_pred[:, :, :1] * loc_pred[:, :, 1:4], axis=1)                          
            # Pastikan semua kolom ditemukan
            if set(selected_cols.keys()) == {'LAT', 'LON', 'DEPTH'}:
                true_coords = event_metadata[[selected_cols['LAT'], selected_cols['LON'], selected_cols['DEPTH']]].values
                epicenter_coords = event_metadata[[selected_cols['LON'], selected_cols['LAT']]].values  # Format (Lon, Lat)
                
                # Plot koordinat prediksi vs observasi
                plots.plot_pred_true_coordinates(pred_coords, 
                                                 true_coords, 
                                                 pos_offset, 
                                                 save_path=os.path.join(output_dir, f'coord_pred_true_{time}.png'))
                
                # Plot perbandingan episenter
                plots.plot_epicenter_comparison(pred_coords, 
                                                epicenter_coords, 
                                                pos_offset, 
                                                time=time, 
                                                save_path=os.path.join(output_dir, f'epicenter_comparison_{time}.png'))
            else:
                print("Error: Tidak semua kolom yang diperlukan ditemukan dalam event_metadata.")
     
        
        mag_pred_full += [mag_pred]
        loc_pred_full += [loc_pred]
        pga_pred_full += [pga_pred]


        if not no_event_token:
            mag_stats += [calc_mag_stats(mag_pred, event_metadata, key)]
            loc_stats += [calc_loc_stats(loc_pred, event_metadata, pos_offset)]
            pga_stats += [calc_pga_stats(pga_pred, pga_true)]

            # Evaluasi statistik
            r2, rmse_mag, mae_mag = calc_mag_stats(mag_pred, event_metadata, key)
            rmse_hypo, mae_hypo, rmse_epi, mae_epi = calc_loc_stats(loc_pred, event_metadata, pos_offset)
           

            generate_true_pred_plot(mag_pred, event_metadata[key].values, time, output_dir)
            generate_calibration_plot(mag_pred, event_metadata[key].values, time, output_dir)

        if args.pga:
            pga_pred_reshaped = np.concatenate(pga_pred, axis=0)
            pga_true_reshaped = np.concatenate(pga_true, axis=0)
            #mask = ~np.logical_or(np.isnan(pga_true_reshaped), np.isinf(pga_true_reshaped))
            mask_valid = ~np.logical_or(np.isnan(pga_true_reshaped), np.isinf(pga_true_reshaped))
            mask_nonzero = pga_true_reshaped != 0
            mask = np.logical_and(mask_valid, mask_nonzero)
            pga_true_reshaped = pga_true_reshaped[mask]
            pga_pred_reshaped = pga_pred_reshaped[mask]
            generate_true_pred_plot(pga_pred_reshaped, pga_true_reshaped, time, output_dir, suffix='pga')
            generate_calibration_plot(pga_pred_reshaped, pga_true_reshaped, time, output_dir, suffix='pga')                 
        
    results = {'times': times,
               'mag_stats': np.array(mag_stats).tolist(),
               'loc_stats': np.array(loc_stats).tolist(),
               'pga_stats': np.array(pga_stats).tolist()}
    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    if args.head_times:
        times_pga = np.arange(args.blind_time, 25, 0.2)
        alpha = [float(x) for x in args.alpha.split(',')]
        warning_time_information = calculate_warning_times(config, model, data, event_metadata,
                                                           times=times_pga,
                                                           alpha=alpha,
                                                           batch_size=args.batch_size,
                                                           use_multiprocessing=not args.no_multiprocessing,
                                                           no_event_token=no_event_token,
                                                           dataset_id=args.dataset_id)
    else:
        warning_time_information = None
        alpha = None

    # Tambahkan plot warning time jika informasi tersedia
    if warning_time_information is not None:
        save_warning_plot_path = os.path.join(output_dir, 'warning_time_plot.png')
        plots.plot_warning_times(warning_time_information, alpha, times_pga, save_path=save_warning_plot_path)

    
    mag_pred_full = np.array(mag_pred_full)
    loc_pred_full = np.array(loc_pred_full)
    with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as pred_file:
        pickle.dump((times, mag_pred_full, loc_pred_full, pga_pred_full, warning_time_information, alpha), pred_file)

    
    try:
        pga_pred_reshaped
    except NameError:
        stats_json_path = os.path.join(output_dir, 'stats.json')
        save_plot_path = os.path.join(output_dir, 'rmse_hypo_epic_vs_time.png')
        plots.plot_loc_rmse_vs_time(stats_json_path, save_path=save_plot_path)
    
        save_mae_plot_path = os.path.join(output_dir, 'mae_hypo_epic_vs_time.png')
        plots.plot_loc_mae_vs_time(stats_json_path, save_path=save_mae_plot_path)
    
        save_plot_path = os.path.join(output_dir, 'rmse_mag_vs_time.png')
        plots.plot_mag_rmse_vs_time(stats_json_path, save_plot_path)
    
        save_mae_plot_path = os.path.join(output_dir, 'mae_mag_vs_time.png')
        plots.plot_mag_mae_vs_time(stats_json_path, save_mae_plot_path)  # Jika pga_pred_reshaped tidak terdefinisi, lanjutkan tanpa melakukan apa-apa


# Simpan seluruh ringkasan event yang diminta ke CSV
#if summary_rows:
#    summary_df = pd.DataFrame(summary_rows)
#    summary_df.to_csv(os.path.join(output_dir, "event_eval_summary.csv"), index=False)
            
