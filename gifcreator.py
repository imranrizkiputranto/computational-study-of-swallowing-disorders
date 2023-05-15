import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.cm import ScalarMappable

# Making GIF file
# length = 1536
length = np.linspace(0, 29920, 188)
length = length[:-1]

dt = 0.0001572189609580173
# dt = 0.00017857

n_particles = 3914
n_fluid = 2278

# n_fluid = 3381
# n_particles = 5381

# n_fluid = 8098

a = 0

images = []
directory1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/posfiles/pos'
directory2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/velocity/vel'
directory3 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/acceleration/accel'
directory4 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/pressure/pressure'
directory5 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/density/density'
directory6 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/surgefront/density'
ScatterDotSize = 6

for iter in tqdm(length):
    # plt.style.use("dark_background")
    fig, ax = plt.subplots()

    '''
    file = np.load(directoryboundary + str(int(iter)) + '.npy')
    ax.scatter(file[:, 0], file[:, 1], s=ScatterDotSize)
    ax.set_title(f'time step = {a}')
    ax.set_xlim(200, 390)
    ax.set_ylim(260, 500)
    '''
    a += 1

    file1 = np.load(directory1 + str(int(iter)) + '.npy')
    file2 = np.load(directory4 + str(int(iter)) + '.npy')
    newfile2 = np.zeros((len(file2), 1))

    for i in range(len(file2)):
        newfile2[i] = np.linalg.norm(file2[i])
    # Set your desired minimum and maximum values for the colorbar
    min_value = 0
    max_value = 12000000000
    # min_value = 0  # -2000000000
    # max_value = 12000000000

    # Clip the values in file2 to the range between min_value and max_value
    # clipped_file2 = np.clip(newfile2[0:n_fluid], min_value, max_value)
    clipped_file2 = np.copy(newfile2[0:n_fluid])
    n_particles = len(file1)

    ax.scatter(file1[0:n_fluid, 0], file1[0:n_fluid, 1], s=ScatterDotSize)
    # scatter = ax.scatter(file1[0:n_fluid, 0], file1[0:n_fluid, 1], s=ScatterDotSize, c=clipped_file2[:], cmap='coolwarm', vmin=0, vmax=12000000000)
    ax.scatter(file1[n_fluid:n_particles, 0], file1[n_fluid:n_particles, 1], s=ScatterDotSize, c='lightcoral')
    # plt.xlim(200, 500)
    # plt.ylim(-250, -700)
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    # plt.gca().invert_yaxis()
    # plt.xticks([], [])
    # plt.yticks([], [])
    x = iter * dt
    ax.set_title(f'Dam Break, t = {round(x, 3)}s')
    # sm = ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=min(clipped_file2[:]), vmax=max(clipped_file2[:])))
    # sm = ScalarMappable(cmap='coolwarm', norm=plt.Normalize(min_value, max_value))
    # sm.set_array(None)
    # plt.colorbar(sm)
    plt.tight_layout()
    # plt.colorbar(scatter, ticks=np.linspace(0, 12000000000, 7), label="Pressure $[kg / s^2 px]$")

    # Save the plot as an image
    filename = f'frame_{iter}.png'
    plt.savefig(filename)

    # Append the image to the list of images
    images.append(imageio.imread(filename))

    # Clear the plot
    plt.clf()

imageio.mimsave('DamBreak.gif', images, duration = 0.0001)
