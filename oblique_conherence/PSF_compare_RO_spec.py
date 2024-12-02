# DEFINE iPSF

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import h5py
from pathlib import Path
from tqdm import tqdm

def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.write(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.write(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


# DEFINE MESH
def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma = k * ((xv - x0) * np.cos(ma_phi * np.pi / 180) + (yv - y0) * np.sin(ma_phi * np.pi / 180)) * np.sin(
        ma_theta * np.pi / 180)  # misalignment

    phi_ma = ma
    phi_de = - (phi_inc + phi_sca)
    phi_diff = ma - (phi_inc + phi_sca)  # phase difference

    iPSF_cos = np.cos(phi_diff)
    iPSF = 2 * Escat * np.cos(phi_diff)

    return phi_ma, phi_de, phi_diff, iPSF_cos, iPSF


if __name__ == '__main__':
    # Set to number of cores on your machine
    CORES = 4
    plt.rcParams['figure.figsize'] = [9.5, 6]

    # DEFINE CONSTANTS

    fov = 14600  # size of field of view

    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
    xv = xv * fov
    yv = yv * fov

    oblique = 22

    E0 = 0.3

    lam=488
    lam_list = range(400, 600, 1)

    zpp = 10000
    zpp_list = range(0, 10000, 200)

    # ------------------------------------------- Single depth ------------------------------------------- #
    print('----- Single depth -----')

    # non-RO
    print('non-RO')
    k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium
    E = E0
    phi = 0
    phi_ma, phi_de, phi_diff, iPSF_cos, iPSF_nonRO = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E, np.pi / 2,
                                                                       oblique, phi)
    # non-RO, spectrum
    print('non-RO, spectrum')
    iPSF_nonRO_spec = 0
    phi = 0
    for lam in tqdm(lam_list):
        E = E0 / (lam ** 3)
        k = 2 * np.pi / (lam / 1.33)
        phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E,
                                                                     np.pi / 2,
                                                                     oblique, phi)
        iPSF_nonRO_spec = iPSF_nonRO_spec + iPSF

    # RO
    print('RO')
    iPSF_RO = 0
    k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium
    E = E0
    for phi in range(0, 360, 10):
        phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E, np.pi / 2,
                                                                     oblique, phi)
        iPSF_RO = iPSF_RO + iPSF

    # RO,spectrum
    print('RO, spectrum')
    iPSF_RO_spec = 0
    for lam in tqdm(lam_list):
        E = E0 / (lam ** 3)
        k = 2 * np.pi / (lam / 1.33)
        for phi in range(0, 360, 10):
            phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E,
                                                                         np.pi / 2,
                                                                         oblique, phi)
            iPSF_RO_spec = iPSF_RO_spec + iPSF

    save_tiff('iPSF_nonRO.tif', iPSF_nonRO[..., None])
    save_tiff('iPSF_nonRO_spec.tif', iPSF_nonRO_spec[..., None])
    save_tiff('iPSF_RO.tif', iPSF_RO[..., None])
    save_tiff('iPSF_RO_spec.tif', iPSF_RO_spec[..., None])

    # ------------------------------------------- Multi depth ------------------------------------------- #
    print('----- Multi depth -----')

    center_nonRO = []
    center_nonRO_spec=[]
    center_RO = []
    center_RO_spec = []

    for zpp in tqdm(zpp_list):
        # nonRO
        k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium
        E = E0
        phi = 0
        phi_ma, phi_de, phi_diff, iPSF_cos, iPSF_nonRO = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E, np.pi / 2,
                                                                           oblique, phi)

        center_nonRO.append(iPSF_nonRO[100, 100])

        # non-RO, spectrum
        iPSF_nonRO_spec = 0
        phi=0
        for lam in lam_list:
            E = E0 / (lam ** 3)
            k = 2 * np.pi / (lam / 1.33)
            phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E,
                                                                         np.pi / 2,
                                                                         oblique, phi)
            iPSF_nonRO_spec = iPSF_nonRO_spec + iPSF

        center_nonRO_spec.append(iPSF_nonRO_spec[100, 100])

        # RO
        iPSF_RO = 0

        k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium
        E = E0
        for phi in range(0, 360, 10):
            phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E, np.pi / 2,
                                                                         oblique, phi)
            iPSF_RO = iPSF_RO + iPSF

        center_RO.append(iPSF_RO[100, 100])

        # RO, spectrum
        iPSF_RO_spec = 0

        for lam in lam_list:
            E = E0 / (lam ** 3)
            k = 2 * np.pi / (lam / 1.33)
            for phi in range(0, 360, 10):
                phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, E,
                                                                             np.pi / 2,
                                                                             oblique, phi)
                iPSF_RO_spec = iPSF_RO_spec + iPSF

        center_RO_spec .append(iPSF_RO_spec[100, 100])

    plt.figure()
    plt.plot(zpp_list, center_nonRO, 'b')
    plt.show()

    plt.figure()
    plt.plot(zpp_list, center_nonRO_spec, 'r')
    plt.show()

    plt.figure()
    plt.plot(zpp_list, center_RO, 'k')
    plt.show()

    plt.figure()
    plt.plot(zpp_list, center_RO_spec, 'g')
    plt.show()
