import numpy as np
import os
import scipy.sparse as sp
import torch
import pickle
from datetime import datetime, timedelta

# DataLoader class for handling data batches
class DataLoader(object):
    def __init__(self, xs, ys, metadata, batch_size, pad_with_last_sample=True):
        # Batch size for data loading
        self.batch_size = batch_size
        # Current index for iteration
        self.current_ind = 0
        # Metadata associated with the data
        self.metadata = metadata
        if pad_with_last_sample:
            # Calculate the number of padding samples
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            # Create padding for input data
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            # Create padding for target data
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            # Concatenate input data with padding
            xs = np.concatenate([xs, x_padding], axis=0)
            # Concatenate target data with padding
            ys = np.concatenate([ys, y_padding], axis=0)
            if metadata is not None:
                # Create padding for metadata
                m_padding = np.repeat(metadata[-1:], num_padding, axis=0)
                # Concatenate metadata with padding
                metadata = np.concatenate([metadata, m_padding], axis=0)
        # Total number of samples after padding
        self.size = len(xs)
        # Number of batches
        self.num_batch = int(self.size // self.batch_size)
        # Input data
        self.xs = xs
        # Target data
        self.ys = ys

    def shuffle(self):
        # Generate a random permutation of indices
        permutation = np.random.permutation(self.size)
        # Shuffle input data
        xs, ys = self.xs[permutation], self.ys[permutation]
        # Update input data
        self.xs = xs
        # Update target data
        self.ys = ys

    def get_iterator(self):
        # Reset the current index
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                # Calculate the start index of the current batch
                start_ind = self.batch_size * self.current_ind
                # Calculate the end index of the current batch
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                # Get the input data for the current batch
                x_i = self.xs[start_ind:end_ind, ...]
                # Get the target data for the current batch
                y_i = self.ys[start_ind:end_ind, ...]
                # Get the metadata for the current batch if available
                m_i = self.metadata[start_ind:end_ind, ...] if self.metadata is not None else None
                # Yield the batch data
                yield (x_i, y_i, m_i)
                # Increment the current index
                self.current_ind += 1

        return _wrapper()


# StandardScaler class for data normalization
class StandardScaler:
    def __init__(self, mean, std):
        # Mean value for normalization
        self.mean = mean
        # Standard deviation for normalization
        self.std = std

    def transform(self, data):
        # Normalize the data
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # Denormalize the data
        return (data * self.std) + self.mean


# Function to load the dataset
def load_dataset(dataset_dirs, batch_size, valid_batch_size=None, test_batch_size=None):
    # Dictionary to store the loaded data
    data = {"metadata": {}}

    # Load raw data
    for dataset_dir in dataset_dirs:
        for category in ["train", "val", "test"]:
            # Construct the file path
            file_path = os.path.join('data', dataset_dir, category + ".npz")
            if not os.path.exists(file_path):
                # Raise an error if the file is not found
                raise FileNotFoundError(f"File not found: {file_path}")
            # Load the data from the file
            cat_data = np.load(file_path)
            # Store the input data
            data[f"x_{category}"] = cat_data["x"]
            # Store the target data
            data[f"y_{category}"] = cat_data["y"]

    # Data normalization
    scaler = StandardScaler(
        # Calculate the mean of the training input data
        mean=data["x_train"][..., 0].mean(),
        # Calculate the standard deviation of the training input data
        std=data["x_train"][..., 0].std()
    )
    for category in ["train", "val", "test"]:
        # Normalize the input data
        data[f"x_{category}"][..., 0] = scaler.transform(data[f"x_{category}"][..., 0])
        # Normalize the target data
        # data[f"y_{category}"][..., 0] = scaler.transform(data[f"y_{category}"][..., 0])

    # Calculate dataset parameters
    # Number of training samples
    num_train = data["x_train"].shape[0]
    # Number of validation samples
    num_val = data["x_val"].shape[0]
    # Number of test samples
    num_test = data["x_test"].shape[0]
    # Sequence length
    seq_len = data["x_train"].shape[1]
    # Total number of samples
    total_samples = num_train + num_val + num_test
    # Total number of timesteps
    total_timesteps = total_samples * seq_len

    def generate_dates(total_timesteps):
        # Start date
        current = datetime(2016, 4, 1, 0, 0)
        # List to store dates
        dates = []
        for _ in range(total_timesteps):
            # Append date information
            dates.append([
                current.year,
                current.month,
                current.day,
                current.hour,
                current.minute,
                (current.weekday() + 1) % 7  # Adjust to 0=Sunday, 1=Monday...6=Saturday
            ])
            # Increment the time by 30 minutes
            current += timedelta(minutes=30)
        return np.array(dates)

    # Generate all dates
    all_dates = generate_dates(total_timesteps)

    def split_dates(dates, start_step, num_samples):
        """Recalculate the time window based on actual data features"""
        # Calculate the end step
        end_step = start_step + num_samples * seq_len
        # Slice the dates
        sliced = dates[start_step:end_step]

        # Regenerate the time series (fix the step accumulation error)
        base_date = datetime(2016, 4, 1)
        sliced = []
        for i in range(num_samples * seq_len):
            # Calculate the actual time
            actual_time = base_date + timedelta(minutes=30 * (start_step + i))
            sliced.append([
                actual_time.year,
                actual_time.month,
                actual_time.day,
                actual_time.hour,
                actual_time.minute,
                (actual_time.weekday() + 1) % 7
            ])
        # Reshape the sliced dates
        sliced = np.array(sliced).reshape(num_samples, seq_len, 6)
        return sliced

    try:
        # Split the dates for training data
        train_dates = split_dates(all_dates, 0, num_train)
        # Split the dates for validation data
        val_dates = split_dates(all_dates, num_train * seq_len, num_val)
        # Split the dates for test data
        test_dates = split_dates(all_dates, (num_train + num_val) * seq_len, num_test)
    except IndexError as e:
        # Raise an error if data splitting fails
        raise ValueError(f"Data splitting error: {str(e)}")

    # Redesign the metadata generation logic
    for category in ["train", "val", "test"]:
        # Get the input data
        x_data = data[f"x_{category}"]
        # Number of samples
        num_samples = x_data.shape[0]
        # Sequence length
        seq_len = x_data.shape[1]
        # Number of nodes
        num_nodes = x_data.shape[2]

        # Initialize the metadata array
        metadata = np.zeros((num_samples, seq_len, num_nodes, 6), dtype=np.int32)

        # Get the global start time
        global_start = datetime(2016, 4, 1, 0, 0)
        total_steps = 0

        # Modify the logic to get the sample offset in metadata generation
        for sample_idx in range(num_samples):
            # Correctly get the start time offset for each sample
            sample_offset = (sample_idx * seq_len) % (48 * 91)
            for step_idx in range(seq_len):
                # Calculate the current time
                current_time = global_start + timedelta(minutes=30 * (sample_offset + step_idx))

                # Force the correctness of the week information (Python weekday: 0=Monday, April 1st is actually Friday)
                calc_weekday = (current_time.weekday() + 1) % 7  # Convert to 0=Sunday
                if current_time.date() == datetime(2016, 4, 1).date():
                    calc_weekday = 5  # Force April 1st to be Friday

                # Assign the metadata
                metadata[sample_idx, step_idx, :, :] = [
                    current_time.year,
                    current_time.month,
                    current_time.day,
                    current_time.hour,
                    current_time.minute,
                    calc_weekday
                ]

                # Validate the time continuity
                if sample_idx > 0 and step_idx == 0:
                    prev_time = global_start + timedelta(
                        minutes=30 * (sample_offset - seq_len + step_idx))
                    time_diff = current_time - prev_time
                    # if time_diff != timedelta(minutes=30):
                    #     print(f"Time discontinuity: {prev_time} → {current_time}")

        # Store the metadata
        data["metadata"][category] = metadata

    # 对训练集数据进行打乱
    random_train = torch.randperm(num_train)
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]
    data["metadata"]["train"] = data["metadata"]["train"][random_train, ...]

    # 对验证集数据进行打乱
    random_val = torch.randperm(num_val)
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]
    data["metadata"]["val"] = data["metadata"]["val"][random_val, ...]

    # Create the training data loader
    data["train_loader"] = DataLoader(
        data["x_train"], data["y_train"], data["metadata"]["train"], batch_size
    )
    # Create the validation data loader
    data["val_loader"] = DataLoader(
        data["x_val"], data["y_val"], data["metadata"]["val"], valid_batch_size
    )
    # Create the test data loader
    data["test_loader"] = DataLoader(
        data["x_test"], data["y_test"], data["metadata"]["test"], test_batch_size
    )
    # Store the scaler
    data["scaler"] = scaler

    return data

# Function to calculate Mean Absolute Error
def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        # Create a mask based on the mask value
        mask = torch.gt(true, mask_value)
        # Select the predicted values based on the mask
        pred = torch.masked_select(pred, mask)
        # Select the true values based on the mask
        true = torch.masked_select(true, mask)
    # Calculate the mean absolute error
    return torch.mean(torch.abs(true - pred))


# Function to calculate Mean Absolute Percentage Error
def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        # Create a mask based on the mask value
        mask = torch.gt(true, mask_value)
        # Select the predicted values based on the mask
        pred = torch.masked_select(pred, mask)
        # Select the true values based on the mask
        true = torch.masked_select(true, mask)
    # Calculate the mean absolute percentage error
    return torch.mean(torch.abs(torch.div((true - pred), true)))


# Function to calculate Root Mean Squared Error
def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        # Create a mask based on the mask value
        mask = torch.gt(true, mask_value)
        # Select the predicted values based on the mask
        pred = torch.masked_select(pred, mask)
        # Select the true values based on the mask
        true = torch.masked_select(true, mask)
    # Calculate the root mean squared error
    return torch.sqrt(torch.mean((pred - true) ** 2))


# Function to calculate Weighted Mean Absolute Percentage Error
def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        # Create a mask based on the mask value
        mask = torch.gt(true, mask_value)
        # Select the predicted values based on the mask
        pred = torch.masked_select(pred, mask)
        # Select the true values based on the mask
        true = torch.masked_select(true, mask)
    # Calculate the weighted mean absolute percentage error
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss


# Function to calculate multiple evaluation metrics
def metric(pred, real):
    # Calculate Mean Absolute Error
    mae = MAE_torch(pred, real, 0).item()
    # Calculate Mean Absolute Percentage Error
    mape = MAPE_torch(pred, real, 0).item()
    # Calculate Weighted Mean Absolute Percentage Error
    wmape = WMAPE_torch(pred, real, 0).item()
    # Calculate Root Mean Squared Error
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape


# Function to load graph data
def load_graph_data(pkl_filename):
    try:
        # Load the pickle data
        pickle_data = load_pickle(pkl_filename)
        # Handle different data structures
        if isinstance(pickle_data, tuple) and len(pickle_data) == 3:
            return pickle_data  # (sensor_ids, sensor_id_to_ind, adj_mx)
        elif isinstance(pickle_data, dict):
            return (None, None, pickle_data.get('adj_mx', None))
        else:
            # Assume to return the adjacency matrix directly
            return (None, None, pickle_data)
    except Exception as e:
        # Print an error message if loading fails
        print(f"Error loading graph data: {e}")
        return (None, None, None)


# Function to load pickle data
def load_pickle(pickle_file):
    try:
        # Open the pickle file in binary read mode
        with open(pickle_file, 'rb') as f:
            # Load the pickle data
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        # Open the pickle file in binary read mode with encoding
        with open(pickle_file, 'rb') as f:
            # Load the pickle data with encoding
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        # Print an error message if loading fails
        print('Unable to load data ', pickle_file, ':', e)
        # Raise the exception
        raise e
    return pickle_data
