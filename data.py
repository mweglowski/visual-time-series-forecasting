import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

def get_data():
    data = yf.download("BTC-USD", start="2016-06-01", end="2026-06-01")
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
    plt.show()