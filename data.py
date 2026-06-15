from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import torch

def get_data():
    data = yf.download('BTC-USD', start='2016-06-01', end='2026-06-01')
    values = data['Close']['BTC-USD'].values
    return values

def divide_into_image_windows(values, window_len=80):
    # sliding window
    windows = []
    images = []
    for i in range(len(values) - window_len + 1):
        orig_window = values[i:i+window_len]
        windows.append(orig_window)

        # slice normalization
        # min-max scaling
        window = np.array(orig_window)
        max_value, min_value = window.max(), window.min()
        if min_value == max_value: # handle no change in values
            window = np.zeros_like(window)
        else:
            window = (window - min_value) / (max_value - min_value)

        matrix = np.zeros((window_len, window_len), dtype=np.float32)
        # where to place our value in a column
        for col_idx in range(window_len):
            # our values are from 0 to 1, so if value is 0.34 then
            # it signalises that it should be 34% from the bottom
            # so then it should be 66% from the top
            value = window[col_idx]
            value_row_idx = round((1 - window[col_idx]) * (window_len - 1))
            matrix[value_row_idx][col_idx] = 1 # drawing line on matrix
        images.append(matrix)

    print(f'{len(windows)} {window_len}-day windows generated, \nso that {len(images)} training data graphs has been created')
    return images, windows

def compare_series_charts(images, windows, steps=5):
    _, ax = plt.subplots(nrows=2,
                         ncols=steps,
                         figsize=(steps * 3, 6),
                         sharey='row',
                         sharex='col')
    for i in range(steps):
        ax[0][i].imshow(images[i], cmap='gray')
        ax[1][i].plot(windows[i], color='black')
    plt.tight_layout()
    plt.savefig('visualizations/charts_comparison.jpg')
    plt.show()

def split_into_x_y(images, window_sight_limit=60, save=True,
                       filename='financial_x_y'):
    X = np.array(images).copy()
    Y = np.array(images).copy()
    X[:, :, window_sight_limit:] = 0
    if save:
        np.savez_compressed(f'data/{filename}.npz', X=X, Y=Y)
    return X, Y

def show_x_y_difference(x, y):
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(8, 6), sharey='row')
    ax[0].imshow(x, cmap='gray')
    ax[1].imshow(y, cmap='gray')
    fig.tight_layout()
    fig.savefig('visualizations/x_y_difference.jpg')
    plt.show()

def load_x_y(filename):
    data = np.load(f'data/{filename}.npz')
    return data['X'], data['Y']

def convert_x_y_to_tensors(X, Y):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_tensor = torch.tensor(X, device=device).unsqueeze(1)
    Y_tensor = torch.tensor(Y, device=device).unsqueeze(1)
    return X_tensor, Y_tensor

def build_dataloaders(X_tensor, Y_tensor, batch_size=128):
    split_val = int(len(X_tensor) * 0.8)
    split_test = int(len(X_tensor) * 0.9)

    train_ds = TensorDataset(X_tensor[:split_val], Y_tensor[:split_val])
    val_ds   = TensorDataset(X_tensor[split_val:split_test], Y_tensor[split_val:split_test])
    test_ds  = TensorDataset(X_tensor[split_test:], Y_tensor[split_test:])

    # Time-series datasets MUST avoid shuffling to keep timeline boundaries clean, avoiding data leakage
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def main():
    values = get_data()
    window_len = 80
    window_sight_limit = 60 # 60/80=0.75 -> train sample see 75% of window 

    # 2D, 1D
    images, windows = divide_into_image_windows(values, window_len)
    X, Y = split_into_x_y(images, window_sight_limit)

    compare_series_charts(images, windows, 5)
    show_x_y_difference(X[0], Y[0])

if __name__ == '__main__':
    main()