import numpy as np

from .utils import fix_byte_order, load_data


def read(file_path, *args):
    """Read one RM_020_DIRG MATLAB recording.

    Args:
        file_path (str): Path to the source recording.

    Returns:
        numpy.ndarray: Recording with shape ``length x channel``.
    """

    data = load_data(file_path, file_type="mat")
    file_name = file_path.split('/')[-1].split('.')[0]
    data = data[file_name]
    # data = pd.DataFrame(data)
    # data.columns = ['A1_x', 'A1_y', 'A1_z', 'A2_x', 'A2_y', 'A2_z']
    
    # Normalize byte order while retaining the reader's float64 contract.
    data = fix_byte_order(data)
    data = data.astype(np.float64)
    
    # 确保是二维数组
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data


if __name__ == "__main__":
    from utils import test_reader

    test_reader(
        metadata_path=(
            "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/"
            "data/metadata_5_data.csv"
        ),
        data_dir="/home/user/data/PHMbenchdata/raw/",
        name="RM_020_DIRG",
        output_dir=(
            "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/"
            "src/data_factory/reader/output"
        ),
        read=read,
    )
