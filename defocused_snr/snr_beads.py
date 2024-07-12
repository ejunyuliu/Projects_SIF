from PFM import data
import tifffile
import numpy as np
import matplotlib.pyplot as plt


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
    data = tifffile.imread('sif-split-1.tif')
    data = data.transpose((2, 1, 0))

    SNR = []

    particle_x = [1243.5, 1202.167, 1213.5, 1234.5, 1220.5, 434.167, 446.5, 455.5, 428.5, 324.833]
    particle_y = [329.5, 485.167, 593.833, 612.5, 698.5, 766.833, 723.5, 701.167, 845.167, 1031.5]

    radius_out = 10
    radius_in = 5

    mask = np.zeros((radius_out * 2, radius_out * 2)).astype(bool)
    mask_in = mask.copy()
    mask_in[(radius_out - radius_in):(radius_out + radius_in), (radius_out - radius_in):(radius_out + radius_in)] = True
    mask_out = ~mask_in

    mask_in, mask_out = mask_in.astype(bool), mask_out.astype(bool)

    for i in range(1, data.shape[-1] + 1):
        sel_list = evenly_spaced_numbers(0, data.shape[-1] - 1, i)
        img = data[:, :, sel_list].mean(axis=-1)

        snr = 0
        for j in range(len(particle_x)):
            x, y = round(particle_x[j]), round(particle_y[j])
            crop = img[x - radius_out:x + radius_out, y - radius_out:y + radius_out]

            snr = snr + np.abs(crop[mask_in].min() - crop[mask_out].mean()) / crop[mask_out].mean()

        SNR.append(snr / len(particle_x))

        # with tifffile.TiffWriter('tas/' + str(i) + '.tif', imagej=True) as tif:
        #     d = np.moveaxis(img[..., None], [2, 1, 0], [0, 1, 2])
        #     tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS

    plt.plot(SNR)
    plt.show()
