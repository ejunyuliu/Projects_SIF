from PFM import data
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def evenly_spaced_numbers(rs, re, x):
    range_start = rs
    range_end = re
    numbers = []

    # 计算步长
    step = (range_end - range_start) / (x - 1) if x > 1 else 0

    # 生成等间隔的数
    for i in range(x):
        num = round(range_start + i * step)
        if num > range_end:
            break
        numbers.append(num)

    return numbers


if __name__ == '__main__':
    data = tifffile.imread('sample-48-1.tif')
    data = data.transpose((2, 1, 0))

    # img_pool = data[639:776, 1305:1493, :]

    df = pd.read_csv('roi.csv')

    X = df.X.values
    Y = df.Y.values
    Width = df.Width.values
    Height = df.Height.values

    snr = np.ones((data.shape[-1], X.shape[0]))

    # for k in range(X.shape[0]):
    #     img_pool = data[X[k]:X[k] + Width[k], Y[k]:Y[k] + Height[k], :]
    #     img = np.ones_like(img_pool)
    #
    #     for i in range(0, data.shape[-1]):
    #         sel_list = evenly_spaced_numbers(0, data.shape[-1] - 1, i + 1)
    #         img[..., i] = img_pool[:, :, sel_list].mean(axis=-1)
    #
    #     noise_var = np.var(img, axis=(0, 1))
    #     snr[:, k] = 10 *(noise_var[-1] / noise_var)

    for k in range(X.shape[0]):
        img_pool = data[X[k]:X[k] + Width[k], Y[k]:Y[k] + Height[k], :]
        img = np.ones_like(img_pool)

        for i in range(0, data.shape[-1]):
            sel_list = evenly_spaced_numbers(0, data.shape[-1] - 1, i + 1)
            img[..., i] = img_pool[:, :, sel_list].mean(axis=-1)

        signal_var = np.var(img[..., -1], axis=(0, 1))
        noise_var = np.var(np.abs(img - img[..., -1][..., None]), axis=(0, 1))
        snr[:, k] = 10 * (signal_var / noise_var)
