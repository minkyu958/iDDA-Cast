import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


def _reduced_window_end(border1, border2, seq_len, pred_len, percent, split_name):
    """Keep the earliest `percent` of VALID forecasting windows within a split.

    A sample requires `seq_len` input points and `pred_len` future points.
    Validation/test keep their existing history overlap with the train/val split.
    The split start and feature scaling statistics remain unchanged.
    """
    if not (1 <= percent <= 100):
        raise ValueError(f"percent must be in [1, 100], got {percent}")

    available_windows = border2 - border1 - seq_len - pred_len + 1
    if available_windows <= 0:
        raise ValueError(
            f"No forecasting windows in {split_name}: "
            f"rows={border2 - border1}, seq_len={seq_len}, pred_len={pred_len}"
        )

    if percent == 100:
        return border2
    kept_windows = max(1, available_windows * percent // 100)
    return border1 + seq_len + pred_len - 1 + kept_windows


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, channel_independent=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independent = channel_independent

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Reduced-data Setting: keep the first percent of valid windows
        # independently within Train, Validation, and Test.
        border2 = _reduced_window_end(
            border1, border2, self.seq_len, self.pred_len,
            self.percent, ['train', 'val', 'test'][self.set_type]
        )

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        if self.channel_independent:
            # Original Time-LLM behavior:
            # each variable is treated as a separate univariate sample.
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
        else:
            # TimesNet behavior:
            # one sample contains all variables simultaneously.
            feat_id = None
            s_begin = index

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.channel_independent:
            return base_len * self.enc_in
        return base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None, channel_independent=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independent = channel_independent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Reduced-data Setting: keep the first percent of valid windows
        # independently within Train, Validation, and Test.
        border2 = _reduced_window_end(
            border1, border2, self.seq_len, self.pred_len,
            self.percent, ['train', 'val', 'test'][self.set_type]
        )

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.channel_independent:
            # Original Time-LLM behavior:
            # each variable is treated as a separate univariate sample.
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
        else:
            # TimesNet behavior:
            # one sample contains all variables simultaneously.
            feat_id = None
            s_begin = index

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.channel_independent:
            return base_len * self.enc_in
        return base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, channel_independent=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independent = channel_independent
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def _prepare_dataframe(self, df_raw):
        # No changes to Exchange/Weather/Traffic/ECL or other Custom datasets.
        return df_raw

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # Dataset_Custom keeps the original behavior. Dataset_Wind only
        # normalizes supported Wind CSV layouts before the existing split.
        df_raw = self._prepare_dataframe(df_raw)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (0.6 if self.data_path == 'AQShunyi.csv' else 0.7))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Reduced-data Setting: keep the first percent of valid windows
        # independently within Train, Validation, and Test.
        border2 = _reduced_window_end(
            border1, border2, self.seq_len, self.pred_len,
            self.percent, ['train', 'val', 'test'][self.set_type]
        )

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # Wind is 15-minute data; other Custom datasets keep the exact
            # four original time markers (month/day/weekday/hour).
            if getattr(self, '_include_quarter_hour_mark', False):
                df_stamp['minute'] = df_stamp.date.apply(
                    lambda row: row.minute // 15, 1
                )
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.channel_independent:
            # Original Time-LLM behavior:
            # each variable is treated as a separate univariate sample.
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
        else:
            # TimesNet behavior:
            # one sample contains all variables simultaneously.
            feat_id = None
            s_begin = index

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.channel_independent:
            return base_len * self.enc_in
        return base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Wind(Dataset_Custom):
    """Wind CSV adapter, preserving Dataset_Custom splits and sampling.

    Accepts wide files (date + numeric variables), or the TFB-style long
    format (date, data, cols). Timestamp aliases are normalized in memory;
    the CSV on disk is not changed.
    """

    _include_quarter_hour_mark = True

    def _prepare_dataframe(self, df_raw):
        df_raw = df_raw.copy()
        df_raw.columns = [str(col).strip() for col in df_raw.columns]

        # Accept common timestamp headers without modifying the input file.
        if 'date' not in df_raw.columns:
            date_column = next(
                (col for col in df_raw.columns
                 if col.lower() in ('date', 'datetime', 'timestamp', 'time', 'datatime')),
                None
            )
            if date_column is None:
                raise ValueError(
                    'Wind CSV needs a date/time column such as '
                    "'date', 'timestamp', or 'DATATIME'. "
                    f'Found: {list(df_raw.columns)}'
                )
            df_raw = df_raw.rename(columns={date_column: 'date'})

        # Some time-series benchmarks distribute one row per timestamp/channel.
        if {'data', 'cols'}.issubset(df_raw.columns):
            extra_columns = set(df_raw.columns) - {'date', 'data', 'cols'}
            if extra_columns:
                raise ValueError(
                    'Wind long-format CSV must contain only date, data, cols; '
                    f'extra columns: {sorted(extra_columns)}'
                )
            if df_raw.duplicated(['date', 'cols']).any():
                raise ValueError(
                    'Wind long-format CSV has duplicate (date, cols) entries. '
                    'Deduplicate or aggregate them before training.'
                )
            df_raw['date'] = pd.to_datetime(df_raw['date'], errors='raise')
            df_raw = df_raw.pivot(index='date', columns='cols', values='data')
            df_raw = df_raw.reset_index()
            df_raw.columns.name = None

        try:
            df_raw['date'] = pd.to_datetime(df_raw['date'], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(
                'Wind timestamps could not be parsed as calendar dates. '
                'Convert the date column to a datetime string before training.'
            ) from exc
        if df_raw['date'].isna().any():
            raise ValueError('Wind CSV contains missing timestamps.')
        if not df_raw['date'].is_monotonic_increasing:
            raise ValueError(
                'Wind timestamps must be in chronological order. '
                'Sort the CSV by date before training.'
            )
        if df_raw['date'].duplicated().any():
            raise ValueError(
                'Wind CSV contains duplicate timestamps. '
                'Deduplicate or aggregate them before training.'
            )

        value_columns = [col for col in df_raw.columns if col != 'date']
        if not value_columns:
            raise ValueError('Wind CSV has no forecasting variables.')

        # The original Dataset_Custom requires the target column to exist.
        # For multivariate prediction (M/MS), the last numerical channel is
        # used as the default target when the CSV does not have an OT column.
        # All channels are still forecast with --features M.
        if self.target not in value_columns:
            if self.target == 'OT' and self.features in ('M', 'MS'):
                self.target = value_columns[-1]
                print(
                    f"[Data] Wind: no 'OT' column; selecting last variable "
                    f"'{self.target}' as the target for column ordering. "
                    'All channels are retained for multivariate forecasting.',
                    flush=True
                )
            else:
                raise ValueError(
                    f"Wind target '{self.target}' is missing. "
                    f'Available variable columns: {value_columns}. '
                    'Set --target to an existing variable.'
                )

        try:
            df_raw[value_columns] = df_raw[value_columns].apply(pd.to_numeric, errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(
                'Wind value columns must be numeric. '
                'If using long format, convert the data field to numbers.'
            ) from exc
        if not np.isfinite(df_raw[value_columns].to_numpy(dtype=float)).all():
            raise ValueError(
                'Wind CSV contains missing or non-finite values. '
                'Clean/impute them before running iDDA-Cast.'
            )
        return df_raw


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly', percent=100):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.percent = percent
        if not (1 <= self.percent <= 100):
            raise ValueError(f"percent must be in [1, 100], got {self.percent}")
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        group_mask = dataset.groups == self.seasonal_patterns
        self.ids = np.asarray(dataset.ids[group_mask])
        self.timeseries = [v[~np.isnan(v)] for v in dataset.values[group_mask]]

        # M4 contains independent series, rather than one chronological CSV.
        # Keep the first `percent` of series sorted by stable series ID.
        # Sorting makes train and test choose the same IDs even if they are
        # stored in a different order. percent=100 preserves original order.
        if self.percent < 100 and len(self.ids) > 0:
            n_keep = max(1, len(self.ids) * self.percent // 100)
            keep = np.argsort(self.ids.astype(str), kind='stable')[:n_keep]
            self.ids = self.ids[keep]
            self.timeseries = [self.timeseries[int(j)] for j in keep]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts_last_window):] = ts_last_window
            insample_mask[i, -len(ts_last_window):] = 1.0
        return insample, insample_mask


