from data_provider.data_loader import Dataset_ETT_hour
# from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour
}

def data_provider(args, flag):
    tsData = data_dict[args.data]  # 这里还没有实例化一个Dataset
    timeenc = 0 if args.embed != 'timeF' else 1  # ???

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'Anomaly_detection' or args.task_name == 'Classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bs = 1 for eval
        freq = args.freq  # ???
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == 'Anomaly_detection':
        pass
    if args.task_name == 'Classification':
        pass
    else:
        if args.data == 'm4':
            drop_last = False

        # 在这里实例化一个Dataset类
        data_set = tsData(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            season_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))

        # 实例化一个DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader  # is tensor?
