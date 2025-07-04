import numpy as np
import pandas as pd
import h5py


class TrainDevTestSplitter:
    @staticmethod
    def run_method(event_metadata, name, shuffle_train_dev, parts):
        mask = np.zeros(len(event_metadata), dtype=bool)

        split_methods = {'test_2016': TrainDevTestSplitter.test_2016,
                         'test_2011': TrainDevTestSplitter.test_2011,
                         'no_test': TrainDevTestSplitter.no_test,
                         'test_sumbar': TrainDevTestSplitter.test_sumbar,
                         'test_10': TrainDevTestSplitter.test_10,
                         'test_15': TrainDevTestSplitter.test_15,
                         'test_20': TrainDevTestSplitter.test_20,
                         'test_25': TrainDevTestSplitter.test_25,
                         'test_30': TrainDevTestSplitter.test_30,}


        if name is None or name == '':
            test_set = TrainDevTestSplitter.default(event_metadata)
        elif name in split_methods:
            test_set = split_methods[name](event_metadata)
        else:
            raise ValueError(f'Unknown split function: {name}')

        if name == 'test_10':
            # Kita ingin total: 80% train, 10% dev, 10% test
            #   => test_set = 10% data 
            #   => sisanya = 90% data => di-split train=80% & dev=10% (dari total)
            #   => dari 90% itu, train = 80/90 ~ 0.888... 
            b1 = int((0.80 / 0.90) * np.sum(~test_set))

        elif name == 'test_15':
            # Kita ingin total: 70% train, 10% dev, 20% test
            #   => test_set = 20% data
            #   => sisanya = 80% data => di-split train=70% & dev=10% (dari total)
            #   => dari 80% itu, train = 70/80 = 0.875
            b1 = int((0.75 / 0.85) * np.sum(~test_set))

        elif name == 'test_20':
            # Kita ingin total: 70% train, 10% dev, 20% test
            #   => test_set = 20% data
            #   => sisanya = 80% data => di-split train=70% & dev=10% (dari total)
            #   => dari 80% itu, train = 70/80 = 0.875
            b1 = int((0.75 / 0.80) * np.sum(~test_set))

        elif name == 'test_25':
            # Kita ingin total: 70% train, 10% dev, 20% test
            #   => test_set = 20% data
            #   => sisanya = 80% data => di-split train=70% & dev=10% (dari total)
            #   => dari 80% itu, train = 70/80 = 0.875
            b1 = int((0.65 / 0.75) * np.sum(~test_set))
            
        elif name == 'test_30':
            # Kita ingin total: 70% train, 10% dev, 20% test
            #   => test_set = 20% data
            #   => sisanya = 80% data => di-split train=70% & dev=10% (dari total)
            #   => dari 80% itu, train = 70/80 = 0.875
            b1 = int((0.60 / 0.70) * np.sum(~test_set))            
            
        elif name is None or name == '':
            # Fallback default-nya 60/10/30 (train/dev/test)
            # b2 = int(0.7 * len(event_metadata)) # 70% train+dev, 30% test
            # kemudian di-split 60/70 (0.6/0.7 ~ 0.857) => 60% train, 10% dev
            b1 = int(0.6 / 0.7 * np.sum(~test_set))
        else:
            # Jika ada lainnya (misal test_2016, dsb) yang default 70/30
            b1 = int(0.65 / 0.7 * np.sum(~test_set))

        #b1 = int(0.6 / 0.7 * np.sum(~test_set))
        train_set = np.zeros(np.sum(~test_set), dtype=bool)
        train_set[:b1] = True

        if shuffle_train_dev:
            np.random.seed(len(event_metadata))  # The same length of data always gets split the same way
            np.random.shuffle(train_set)

        if parts[0] and parts[1]:
            mask[~test_set] = True
        elif parts[0]:
            mask[~test_set] = train_set
        elif parts[1]:
            mask[~test_set] = ~train_set
        if parts[2]:
            mask[test_set] = True

        return mask

    @staticmethod
    def default(event_metadata):
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.7 * len(event_metadata))
        test_set[b2:] = True
        return test_set

    @staticmethod
    def test_2016(event_metadata):
        test_set = np.array([x[:4] == '2016' for x in event_metadata['Time']])
        return test_set

    @staticmethod
    def test_2011(event_metadata):
        test_set = np.array([x[:4] == '2011' for x in event_metadata['Origin_Time(JST)']])
        return test_set

    @staticmethod
    def no_test(event_metadata):
        return np.zeros(len(event_metadata), dtype=bool)

    @staticmethod
    def test_sumbar(event_metadata):
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.7 * len(event_metadata))  # 90% pertama bukan test
        test_set[b2:] = True                # 10% terakhir -> test
        
        
        ## Memastikan event besar tidak masuk test set
        #if 'Magnitude' in event_metadata.columns:
        #    large_eqs = event_metadata['Magnitude'] >= 6.0
        #    test_set[large_eqs.values] = False

        # Menambahkan event dengan #EventID tertentu ke dalam test set
        specific_events = {"2023-04-24T200054_6.9","2022-02-25T013928_6.2","2024-02-23T074628_5.3","2024-01-27T165058_4.6"}#"2023-04-24T200054_6.9"
        test_set |= event_metadata['#EventID'].isin(specific_events)
        
        # Tambahkan 20 event kecil secara acak ke test set (misal < 4.0)
        #small_eq_mask = (event_metadata['Magnitude'] < 4.0) & (~test_set)
        #small_eq_indices = event_metadata[small_eq_mask].sample(n=10, random_state=42).index
        #test_set[small_eq_indices] = True
              
        return test_set
        
        
    @staticmethod
    def test_10(event_metadata):
        # 90% train+dev, 10% test
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.9 * len(event_metadata))  # 90% pertama bukan test
        test_set[b2:] = True                # 10% terakhir -> test
        
        ## Memastikan event besar tidak masuk test set
        if 'M_J' in event_metadata.columns:
            large_eqs = event_metadata['M_J'] >= 6.0
            test_set[large_eqs.values] = False                
        
        return test_set

    @staticmethod
    def test_15(event_metadata):
        # 90% train+dev, 10% test
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.85 * len(event_metadata))  # 90% pertama bukan test
        test_set[b2:] = True                # 10% terakhir -> test
        
        ## Memastikan event besar tidak masuk test set
        if 'M_J' in event_metadata.columns:
            large_eqs = event_metadata['M_J'] >= 6.0
            test_set[large_eqs.values] = False              
        
        return test_set

    @staticmethod
    def test_20(event_metadata):
        # 80% train+dev, 20% test
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.8 * len(event_metadata))  # 80% pertama bukan test
        test_set[b2:] = True                # 20% terakhir -> test

        ## Memastikan event besar tidak masuk test set
        if 'M_J' in event_metadata.columns:
            large_eqs = event_metadata['M_J'] >= 6.0
            test_set[large_eqs.values] = False             
        
        return test_set

    @staticmethod
    def test_25(event_metadata):
        # 90% train+dev, 10% test
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.75 * len(event_metadata))  # 90% pertama bukan test
        test_set[b2:] = True                # 10% terakhir -> test
        
        ## Memastikan event besar tidak masuk test set
        if 'M_J' in event_metadata.columns:
            large_eqs = event_metadata['M_J'] >= 6.5
            test_set[large_eqs.values] = False         
        
        return test_set

    @staticmethod
    def test_30(event_metadata):
        # 90% train+dev, 10% test
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.70 * len(event_metadata))  # 90% pertama bukan test
        test_set[b2:] = True                # 10% terakhir -> test
        
        ## Memastikan event besar tidak masuk test set
        #if 'M_J' in event_metadata.columns:
        #    large_eqs = event_metadata['M_J'] >= 6.5
        #    test_set[large_eqs.values] = False      
        
        return test_set

def load_events(data_paths, limit=None, parts=None, shuffle_train_dev=False, custom_split=None, data_keys=None,
                overwrite_sampling_rate=None, min_mag=None, mag_key=None, decimate_events=None):
    if min_mag is not None and mag_key is None:
        raise ValueError('mag_key needs to be set to enforce magnitude threshold')
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if len(data_paths) > 1:
        raise NotImplementedError('Loading partitioned data is currently not supported')
    data_path = data_paths[0]

    event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
# 🔽 Filter global berdasarkan tahun 2006–2018 dari KiK_File
    #if 'KiK_File' in event_metadata.columns:
     #   tahun = event_metadata['KiK_File'].str[:4].astype(int)
     #   event_metadata = event_metadata[tahun.between(2004, 2018)]

    if min_mag is not None:
        event_metadata = event_metadata[event_metadata[mag_key] >= min_mag]

    for event_key in ['KiK_File', '#EventID', 'EVENT']:
        if event_key in event_metadata.columns:
            break


    if limit:
        event_metadata = event_metadata.iloc[:limit]
    if parts:
        mask = TrainDevTestSplitter.run_method(event_metadata, custom_split, shuffle_train_dev, parts=parts)
        event_metadata = event_metadata[mask]

    if decimate_events is not None:
        event_metadata = event_metadata.iloc[::decimate_events]

    metadata = {}
    data = {}

    with h5py.File(data_path, 'r') as f:
        for key in f['metadata'].keys():
            if key == 'event_metadata':
                continue
            metadata[key] = f['metadata'][key][()]

        if overwrite_sampling_rate is not None:
            if metadata['sampling_rate'] % overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({metadata["sampling_rate"]})')
            decimate = metadata['sampling_rate'] // overwrite_sampling_rate
            metadata['sampling_rate'] = overwrite_sampling_rate
        else:
            decimate = 1

        skipped = 0
        contained = []
        for _, event in event_metadata.iterrows():
            event_name = str(event[event_key])
            if event_name not in f['data']:
                print(f"Skipped event: {event_name}")
                skipped += 1
                contained += [False]
                continue
            contained += [True]
            g_event = f['data'][event_name]
            for key in g_event:
                if data_keys is not None and key not in data_keys:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    #data[key] += [g_event[key][:, ::decimate, :]]
                    #print(f"Shape of g_event[{key}]:{g_event[key].shape}")
                    dataset = g_event[key]
                    if dataset.ndim != 3:
                        print(f"Skipping event {event_name} due to invalid waveforms shape: {dataset.shape}")
                        contained[-1] = False  # tandai event ini untuk di-skip
                        break  # keluar dari loop key untuk event ini
                    data[key] += [dataset[:, ::decimate, :]]
                else:
                    data[key] += [g_event[key][()]]
                if key == 'p_picks':
                    data[key][-1] //= decimate

        data_length = None
        for val in data.values():
            if data_length is None:
                data_length = len(val)
            assert len(val) == data_length

        if len(contained) < len(event_metadata):
            contained += [True for _ in range(len(event_metadata) - len(contained))]
        event_metadata = event_metadata[contained]
        if skipped > 0:
            print(f'Skipped {skipped} events')

    return event_metadata, data, metadata

