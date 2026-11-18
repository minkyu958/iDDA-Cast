from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Wind, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'AQShunyi': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Wind': Dataset_Wind,
    'm4': Dataset_M4,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    # Reduced-data setting: only TRAIN uses --percent.
    # Validation and Test always use the full original split (100%)
    # so that different training-data ratios are evaluated fairly.
    percent = args.percent if flag == 'train' else 100
    # Time-LLM processes variables independently; TimesNet uses all channels.
    # An explicit args.channel_independent setting takes priority if present.
    channel_independent = getattr(args, 'channel_independent', None)
    if channel_independent is None:
        channel_independent = getattr(args, 'model', '') != 'TimesNet'

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            percent=percent
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            channel_independent=channel_independent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    print(
        f"[Data] {args.data} | {flag}: "
        f"percent={percent}% | samples={len(data_set)} | batches={len(data_loader)}",
        flush=True,
    )
    return data_set, data_loader
