from ..Default_dataset import Default_dataset


class set_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)