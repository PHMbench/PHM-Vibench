import os
from types import SimpleNamespace
from src.data_factory import build_data


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    data_dir = os.path.join(base_dir, 'data')
    args_data = SimpleNamespace(
        data_dir=data_dir,
        metadata_file='metadata_dummy.csv'
    )
    metadata, data_dict = build_data(args_data)
    print('元数据:')
    for k, v in metadata.items():
        print(f'ID: {k}, Name: {v.get("Name")}, File: {v.get("File")}', flush=True)
    print('\ndata_dict:')
    for k, arr in data_dict.items():
        print(f'ID: {k}, shape: {arr.shape}, dtype: {arr.dtype}', flush=True)


if __name__ == '__main__':
    main()