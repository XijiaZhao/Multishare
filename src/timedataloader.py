import h5py 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class TimeSeriesDataset(Dataset):
    def __init__(self, hdf5_file_path, pick_one=True, partial=False, partial_type='first', N=None):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            pick_one (bool): If True, randomly pick one sample with the same schedule.
                             If False, include all samples with the same schedule.
            partial (bool): If True, select a partial dataset based on partial_type and N.
            partial_type (str): 'first' or 'last', indicating whether to select the first N or last N schedules.
            N (int): Number of schedules to select when partial is True.
        """
        super(TimeSeriesDataset, self).__init__()
        self.hdf5_file_path = hdf5_file_path
        self.pick_one = pick_one
        self.partial = partial
        self.partial_type = partial_type
        self.N = N
        self.data_items = []
        self.vspotid_list = []
        self.headers = None
        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            # Loop over
            for dataset_name in hdf5_file.keys():
                if (int(dataset_name[0]) == 1) and (dataset_name[1] != '4'):
                    dataset = hdf5_file[dataset_name]
                    data = dataset[()]  # Load the data
                    vspotid = dataset.attrs['vspotid']

                    # Can change the schedule to select the stackup here
                    schedule = vspotid[13:] if vspotid != 'c' else 'CCCC'
                    if self.headers is None:
                        self.headers = dataset.attrs['headers']

                    attributes = dict(dataset.attrs)
                    self.data_items.append({
                        'data': data,
                        'vspotid': vspotid,
                        'schedule': schedule,
                        'attributes': attributes  # Add all attributes
                    })
                    self.vspotid_list.append(vspotid)
            print('Data loading complete.')

        self.schedule_to_indices = {} # Index mapping from schedule to indices
        for idx, item in enumerate(self.data_items):
            schedule = item['schedule']
            if schedule not in self.schedule_to_indices:
                self.schedule_to_indices[schedule] = []
            self.schedule_to_indices[schedule].append(idx)

        # Apply partial selection if required
        if self.partial and self.N is not None:
            #sorted list of unique schedules
            schedules_list = sorted(self.schedule_to_indices.keys())
            if self.partial_type == 'first':
                selected_schedules = schedules_list[:self.N]
            elif self.partial_type == 'last':
                selected_schedules = schedules_list[-self.N:]
            else:
                raise ValueError("partial_type must be 'first' or 'last'")

            # Filter data_items to include by selected schedules
            new_data_items = [item for item in self.data_items if item['schedule'] in selected_schedules]
            self.data_items = new_data_items

            # Rebuild schedule_to_indices based on new data_items
            self.schedule_to_indices = {}
            for idx, item in enumerate(self.data_items):
                schedule = item['schedule']
                if schedule not in self.schedule_to_indices:
                    self.schedule_to_indices[schedule] = []
                self.schedule_to_indices[schedule].append(idx)

        self.indices = list(range(len(self.data_items)))# Indices for __getitem__
        self._calculate_min_max_values()# Min-max normalization

    def _calculate_min_max_values(self):
        # Calculate min and max across all datasets and channels
        all_data = np.concatenate([item['data'] for item in self.data_items], axis=0)
        self.min_values = np.min(all_data, axis=0)
        self.max_values = np.max(all_data, axis=0)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        target_item = self.data_items[idx]
        target_data = target_item['data']
        vspotid = target_item['vspotid']
        schedule = target_item['schedule']

        # Normalize the target data
        target_data = (target_data - self.min_values) / (self.max_values - self.min_values + 1e-8)
        
        same_schedule_indices = self.schedule_to_indices[schedule]# Find indices of samples with the same schedule

        # Remove the current index to avoid pairing the sample with itself
        same_schedule_indices = [i for i in same_schedule_indices if i != idx]

        if not same_schedule_indices:
            # If no other samples with the same schedule, skip
            raise IndexError(f"No other samples with schedule {schedule} for index {idx}")

        if self.pick_one:
            # Randomly pick one sample with the same schedule
            positive_idx = random.choice(same_schedule_indices)
            positive_item = self.data_items[positive_idx]
            positive_data = positive_item['data']
            positive_vspotid = positive_item['vspotid']
            # Normalize the positive data
            positive_data = (positive_data - self.min_values) / (self.max_values - self.min_values + 1e-8)

            # Get sequence lengths
            target_length = target_data.shape[0]
            positive_length = positive_data.shape[0]

            # Pad sequences
            target_data, positive_data = self._pad_sequences(target_data, positive_data)
            target_data = torch.from_numpy(target_data).float()
            positive_data = torch.from_numpy(positive_data).float()
            target_length = torch.tensor(target_length)
            positive_length = torch.tensor(positive_length)
         
            return target_data, positive_data, target_length, positive_length, vspotid, positive_vspotid
        else:
            # Use all samples with the same schedule
            positive_data_list = []
            for positive_idx in same_schedule_indices:
                positive_item = self.data_items[positive_idx]
                positive_data = positive_item['data']
                positive_vspotid = positive_item['vspotid']
                # Normalize the positive data
                positive_data = (positive_data - self.min_values) / (self.max_values - self.min_values + 1e-8)

                # Get sequence lengths
                target_length = target_data.shape[0]
                positive_length = positive_data.shape[0]

                # Pad sequences
                target_padded, positive_padded = self._pad_sequences(target_data, positive_data)

                target_tensor = torch.from_numpy(target_padded).float()
                positive_tensor = torch.from_numpy(positive_padded).float()
                target_length = torch.tensor(target_length)
                positive_length = torch.tensor(positive_length)
                positive_data_list.append((target_tensor, positive_tensor, target_length, positive_length, vspotid, positive_vspotid))

            # List of all positive pairs for this target
            return positive_data_list

    def _pad_sequences(self, seq1, seq2):
        max_len = max(seq1.shape[0], seq2.shape[0])
        # Pad sequences with zeros
        padded_seq1 = np.pad(seq1, ((0, max_len - seq1.shape[0]), (0, 0)), mode='constant')
        padded_seq2 = np.pad(seq2, ((0, max_len - seq2.shape[0]), (0, 0)), mode='constant')
        return padded_seq1, padded_seq2
        
    def get_vspotid(self, idx):
        """
        Get the vspotid label of the sample at the given index.
        """
        return self.data_items[idx]['vspotid']
    
class ExpulsionDataset(Dataset):
    def __init__(self, hdf5_file_path, partial=False, partial_type='first', N=None):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            partial (bool): If True, select a partial dataset based on partial_type and N.
            partial_type (str): 'first' or 'last', indicating whether to select the first N or last N schedules.
            N (int): Number of schedules to select when partial is True.
        """
        super(ExpulsionDataset, self).__init__()
        self.hdf5_file_path = hdf5_file_path
        self.partial = partial
        self.partial_type = partial_type
        self.N = N
        self.data_items = []
        self.headers = None

        with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
            # Iterate
            for dataset_name in hdf5_file.keys():
                if (int(dataset_name[0]) == 1) and (dataset_name[1] != '4'):
                    dataset = hdf5_file[dataset_name]
                    data = dataset[()]  # Load the data
                    vspotid = dataset.attrs['vspotid']
                    schedule = vspotid[13:] if vspotid != 'c' else 'CCCC'
                    expulsion = dataset.attrs.get('Expulsion', 0)  # Get Expulsion, default to 0
                    if self.headers is None:
                        self.headers = dataset.attrs['headers']
                    attributes = dict(dataset.attrs)
                    self.data_items.append({
                        'data': data,
                        'vspotid': vspotid,
                        'schedule': schedule,
                        'expulsion': expulsion,
                        'attributes': attributes  
                    })
            print('ExpulsionDataset loaded.')

        # Mapping from schedule to indices
        self.schedule_to_indices = {}
        for idx, item in enumerate(self.data_items):
            schedule = item['schedule']
            if schedule not in self.schedule_to_indices:
                self.schedule_to_indices[schedule] = []
            self.schedule_to_indices[schedule].append(idx)

        if self.partial and self.N is not None:
            # A sorted list of unique schedules
            schedules_list = sorted(self.schedule_to_indices.keys())
            if self.partial_type == 'first':
                selected_schedules = schedules_list[:self.N]
            elif self.partial_type == 'last':
                selected_schedules = schedules_list[-self.N:]
            else:
                raise ValueError("partial_type must be 'first' or 'last'")

            # Filter data_items to include only those with selected schedules
            new_data_items = [item for item in self.data_items if item['schedule'] in selected_schedules]
            self.data_items = new_data_items

            # Rebuild schedule_to_indices
            self.schedule_to_indices = {}
            for idx, item in enumerate(self.data_items):
                schedule = item['schedule']
                if schedule not in self.schedule_to_indices:
                    self.schedule_to_indices[schedule] = []
                self.schedule_to_indices[schedule].append(idx)

        self.indices = list(range(len(self.data_items)))# list of indices for __getitem__
        self._calculate_min_max_values()
    
    def _calculate_min_max_values(self):
        # Calculate min and max across all datasets and channels
        all_data = np.concatenate([item['data'] for item in self.data_items], axis=0)
        self.min_values = np.min(all_data, axis=0)
        self.max_values = np.max(all_data, axis=0)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        data = item['data']
        expulsion = item['expulsion']
        vspotid = item['vspotid']

        # Normalize
        data = (data - self.min_values) / (self.max_values - self.min_values + 1e-8)
        seq_length = data.shape[0]
        data = torch.from_numpy(data).float()
        expulsion_label = torch.tensor(expulsion).float()
        seq_length = torch.tensor(seq_length)

        return data, expulsion_label, seq_length, vspotid

# collate function for ExpulsionDataset
def expulsion_collate_fn(batch):
    data_list = []
    expulsion_labels = []
    seq_lengths = []
    vspotid_list = []

    for item in batch:
        data, expulsion_label, seq_length, vspotid = item
        data_list.append(data)
        expulsion_labels.append(expulsion_label)
        seq_lengths.append(seq_length)
        vspotid_list.append(vspotid)

    # Pad
    data_padded = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)
    expulsion_labels = torch.stack(expulsion_labels)
    seq_lengths = torch.stack(seq_lengths)

    return data_padded, expulsion_labels, seq_lengths, vspotid_list
def time_series_collate_fn(batch):
    #time_series_collate_fn code
    target_data_list = []
    positive_data_list = []
    target_lengths = []
    positive_lengths = []
    vspotid_list = []
    positive_vspotid_list = []

    for item in batch:
        target_data, positive_data, target_length, positive_length, vspotid, positive_vspotid = item
        target_data_list.append(target_data)
        positive_data_list.append(positive_data)
        target_lengths.append(target_length)
        positive_lengths.append(positive_length)
        vspotid_list.append(vspotid)
        positive_vspotid_list.append(positive_vspotid)

    # Pad sequences
    target_data_padded = torch.nn.utils.rnn.pad_sequence(target_data_list, batch_first=True)
    positive_data_padded = torch.nn.utils.rnn.pad_sequence(positive_data_list, batch_first=True)

    target_lengths = torch.stack(target_lengths)
    positive_lengths = torch.stack(positive_lengths)

    return target_data_padded, positive_data_padded, target_lengths, positive_lengths, vspotid_list, positive_vspotid_list

def create_expulsion_dataloader(hdf5_file_path, batch_size, partial=False, partial_type='first', N=None, shuffle=True):
    dataset = ExpulsionDataset(hdf5_file_path, partial=partial, partial_type=partial_type, N=N)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=expulsion_collate_fn)
    return dataloader

def create_time_series_dataloader(hdf5_file_path, batch_size, pick_one=True, partial=False, partial_type='first', N=None, shuffle=True):
    dataset = TimeSeriesDataset(hdf5_file_path, pick_one=pick_one, partial=partial, partial_type=partial_type, N=N)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=time_series_collate_fn)
    return dataloader
