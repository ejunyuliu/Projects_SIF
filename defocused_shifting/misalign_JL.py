# DEFINE iPSF

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import h5py
from pathlib import Path

# Set to number of cores on your machine
CORES = 4
plt.rcParams['figure.figsize'] = [9.5, 6]

# DEFINE CONSTANTS
k = 2 * np.pi / (635 / 1.33)  # wavenumber in medium
fov = 14600  # size of field of view


# DEFINE MESH
def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
xv = xv * fov;
yv = yv * fov


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma = k * ((xv - x0) * np.cos(ma_phi * np.pi / 180) + (yv - y0) * np.sin(ma_phi * np.pi / 180)) * np.sin(
        ma_theta * np.pi / 180)  # misalignment

    phi_diff = ma - (phi0 + phi_inc + phi_sca)  # phase difference
    # phi_diff = ma
    # phi_diff = - (phi0 + phi_inc)
    # phi_diff = - phi_sca
    # phi_diff = ma - phi_sca

    # iPSF = phi_diff
    # iPSF = Escat
    iPSF = 2 * Escat * np.cos(phi_diff)
    # iPSF = 2 * np.cos(phi_diff)

    return iPSF


# AN EXAMPLE, theta=15 deg and phi=45 deg.
# iPSF = iPSF_misalignment(xv, yv, k, 0, 0, 300, 0.3, np.pi / 2, 15, 45)
#
# plt.figure()
# plt.imshow(iPSF, cmap='gray', extent=[np.amin(xv), np.amax(xv), np.amin(yv), np.amax(yv)], origin='lower')
# plt.xlabel('$x$ [nm]')
# plt.ylabel('$y$ [nm]')
# plt.show()

# iPSF = iPSF_misalignment(xv, yv, k, 0, 0, 1000, 0.3, np.pi / 2, 22, 0)
# plt.figure()
# plt.imshow(iPSF, cmap='gray', extent=[np.amin(xv), np.amax(xv), np.amin(yv), np.amax(yv)], origin='lower')
# plt.xlabel('$x$ [nm]')
# plt.ylabel('$y$ [nm]')
# plt.show()
#
# iPSF = np.zeros_like(xv)
# for phi in np.arange(0, 360, 10):
#     iPSF = iPSF + iPSF_misalignment(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, 22, phi)
#
# plt.figure()
# plt.imshow(iPSF, cmap='gray', extent=[np.amin(xv), np.amax(xv), np.amin(yv), np.amax(yv)], origin='lower')
# plt.xlabel('$x$ [nm]')
# plt.ylabel('$y$ [nm]')
# # plt.savefig('./output/2p5/' + str(phi) + '.png')
# plt.show()
# # plt.clf()

# profile=[]
# for theta in range(0,3600):
#     iPSF = iPSF_misalignment(xv, yv, k, 0, 0, 3000, 0.3, np.pi / 2, theta, 45)
#     profile.append(iPSF[100,100])
#
# plt.figure()
# plt.plot(profile)
# plt.show()

a = []
for zpp in np.arange(0, 1000, 1):
    iPSF = np.zeros_like(xv)
    for phi in np.arange(0, 360, 10):
        iPSF = iPSF + iPSF_misalignment(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, 22, phi)

    a.append(iPSF[100, 100])

plt.plot(a)
plt.show()

# plt.figure()
# plt.imshow(iPSF, cmap='gray', extent=[np.amin(xv), np.amax(xv), np.amin(yv), np.amax(yv)], origin='lower')
# plt.xlabel('$x$ [nm]')
# plt.ylabel('$y$ [nm]')
# # plt.savefig('./output/2p5/' + str(phi) + '.png')
# plt.show()
# # plt.clf()
