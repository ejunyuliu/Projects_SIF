import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.special import j1
from scipy.ndimage import shift
from scipy.signal import convolve2d


def read_tiff(filename):
    with tifffile.TiffFile(filename) as tf:
        tif = np.ascontiguousarray(np.moveaxis(tf.asarray(), [0, 1, 2], [2, 1, 0]))

    return tif


def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


def psf_detection(r, z, wavelength, NA, n, d):
    r[r == 0] = 1e-12
    k = 2 * np.pi / wavelength
    radial_term = (2 * j1(k * NA * r) / (k * NA * r)) ** 2
    axial_term = np.exp(-160 * (k ** 2) * (n ** 2) * (NA ** 2) * (z ** 2) / (d ** 2))
    return radial_term * axial_term


if __name__ == '__main__':
    NA = 1.4
    lam = 488
    n = 1.33

    airydisk = 1.22 * lam / NA
    pinhole_au = 0.3
    pinhole_d = pinhole_au * airydisk

    # generate grid
    pixel_size = 20
    size = 512

    axis_seq = np.arange(-size // 2, size - size // 2) * pixel_size

    X, Y, Z = np.meshgrid(axis_seq, axis_seq, axis_seq, indexing='ij')

    # generate wide-field PSF
    psf_wf = psf_detection(np.sqrt(X ** 2 + Y ** 2), Z, lam, NA, n, lam)
    psf_confocal = psf_detection(np.sqrt(X ** 2 + Y ** 2), Z, lam, NA, n, pinhole_d)

    save_tiff('./PSF/wf.tif', psf_wf)
    save_tiff('./PSF/cf.tif', psf_confocal)

    # generate RO-iSCAT PSF
    # rat = 0.40283650647
    # psf_ro = np.zeros_like(psf_wf)
    # for z in range(0, size):
    #     for phi in range(0, 360, 10):
    #         shift_x = np.abs(z - size // 2) * rat * pixel_size * np.sin(phi / 180 * np.pi)
    #         shift_y = np.abs(z - size // 2) * rat * pixel_size * np.cos(phi / 180 * np.pi)
    #         psf_ro[..., z] = psf_ro[..., z] + shift(psf_wf[..., z], shift=(shift_y, shift_x))

    # generate RO PSF
    rat = 0.40283650647
    mask = np.zeros_like(psf_wf)
    for z in range(0, size):
        R = np.sqrt(X[..., z] ** 2 + Y[..., z] ** 2)
        Z = np.abs(z - size // 2)
        mask[..., z][(R <= rat * (Z + 2) * pixel_size) & (R >= rat * Z * pixel_size)] = 1
        mask[..., z] = mask[..., z] / (mask[..., z].sum() + 1e-7)

    otf_mask = np.fft.fftn(mask, axes=(0, 1))
    otf_wf = np.fft.fftn(psf_wf, axes=(0, 1))
    otf_ro = otf_mask * otf_wf
    psf_ro = np.real(np.fft.fftshift(np.fft.ifftn(otf_ro, axes=(0, 1)), axes=(0, 1)))

    save_tiff('./PSF/ro.tif', psf_ro)


    # check PSF profiles
    plt.figure()
    psf_wf = psf_wf / psf_wf.max()
    psf_confocal = psf_confocal / psf_confocal.max()
    psf_ro = psf_ro / psf_ro.max()

    plt.plot(axis_seq, psf_wf[size // 2, size // 2, :], 'g')
    plt.plot(axis_seq, psf_confocal[size // 2, size // 2, :], 'k')
    plt.plot(axis_seq, psf_ro[size // 2, size // 2, :], 'r')
    plt.show()

    # plt.plot(axis_seq, psf_wf.sum(axis=(0, 1)), 'g')
    # plt.plot(axis_seq, psf_confocal.sum(axis=(0, 1)), 'k')
    # plt.plot(axis_seq, psf_ro.sum(axis=(0, 1)), 'r')
    # plt.show()
