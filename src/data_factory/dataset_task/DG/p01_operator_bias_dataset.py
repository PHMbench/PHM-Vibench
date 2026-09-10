"""Ordinary DG windows with explicitly named per-record physical metadata."""
from ..Default_dataset import Default_dataset


class set_dataset(Default_dataset):
    def __init__(self,data,metadata,args_data,args_task,mode='train'):
        super().__init__(data,metadata,args_data,args_task,mode)
        if self.split_strategy!='grouped_metadata':
            raise ValueError('P01 requires grouped_metadata split before windowing')
        row=metadata[self.key]
        columns={
            'unit_id':args_data.p01_unit_field,
            'sample_rate_hz':args_data.p01_fs_field,
            'rotation_speed_rpm':args_data.p01_rpm_field,
        }
        self.physical={key:row[column] for key,column in columns.items()}
        self.physical['unit_id']=str(self.physical['unit_id'])

    def __getitem__(self,idx):
        item=super().__getitem__(idx)
        return dict(item,**self.physical)
