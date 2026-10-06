# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc_opt
import itertools as it
from time import gmtime, strftime, perf_counter
from numba import jit, cuda
import rocket_fft

# %% [markdown]
# Numba:

# %%
"""
This implementation of Lenia is written with the Numba library, meaning that some functions 
previously handled by NumPy have had to be written by hand. Most obviously in this case are the 
np.roll and np.clip functions, not supported by Numba and so written by hand as for loops.

The underlying algorithm is the same as the NumPy implementation.

The @jit(nopython=True) decorator is used to call Numba to compile the function to machine code.

As Numba doesn't fit into the normal python function notation, the local rule is written for each task,
so this cell is not called on by future cells.
"""
R = 13
world_size = 100
T = 10
#pre-computing kernel and its fft
lenia_p_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
lenia_p_kernel = (lenia_p_kernel<1) * np.exp(4 - 4/(4*lenia_p_kernel*(1-lenia_p_kernel)))
##lenia_p_kernel = (lenia_p_kernel<1) * bert_exp(x=lenia_p_kernel, a=4)
lenia_p_kernel = lenia_p_kernel / np.sum(lenia_p_kernel)
#its fft:
pad_shape = world_size - np.array(lenia_p_kernel.shape)
lenia_p_kernel_padded = np.pad(lenia_p_kernel, [(0, i) for i in pad_shape], mode='constant')

lenia_p_kernel_fft = np.fft.fft2(lenia_p_kernel_padded)

pot_roll_iteratable = np.array(list(it.product(range(world_size), range(world_size))))

@jit(nopython=True)
def numba_lenia(world_size, R, T, frames, m, s):
    dt = 1/T

    world = np.random.rand(world_size, world_size)

    for i in range(frames):
        pot = np.fft.ifft2(lenia_p_kernel_fft * np.fft.fft2(world))
        for j, k in pot_roll_iteratable:
            pot[j,k] = pot[((j+12)%world_size),((k+12)%world_size)]

        updater = dt*(2*np.exp((-((pot-m)/s)**2)/2) - 1)
        
        world = np.minimum(1, np.maximum(world.real + updater.real, 0))
        

    return world.real


# %% [markdown]
# Timing vs N:

# %%
#setting up the N values to test:
N_set = np.arange(30,1000,2)
time_set = []

for N in N_set:
    R = 13
    world_size = N
    T = 10
    #pre-computing kernel and its fft
    lenia_p_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
    lenia_p_kernel = (lenia_p_kernel<1) * np.exp(4 - 4/(4*lenia_p_kernel*(1-lenia_p_kernel)))
    ##lenia_p_kernel = (lenia_p_kernel<1) * bert_exp(x=lenia_p_kernel, a=4)
    lenia_p_kernel = lenia_p_kernel / np.sum(lenia_p_kernel)
    #its fft:
    pad_shape = world_size - np.array(lenia_p_kernel.shape)
    lenia_p_kernel_padded = np.pad(lenia_p_kernel, [(0, i) for i in pad_shape], mode='constant')

    lenia_p_kernel_fft = np.fft.fft2(lenia_p_kernel_padded)

    """
    Re-write Numba Lenia local rule:
    """
    pot_roll_iteratable = np.array(list(it.product(range(world_size), range(world_size))))

    @jit(nopython=True)
    def numba_lenia(world_size, R, T, frames, m, s):
        dt = 1/T

        world = np.random.rand(world_size, world_size)

        for i in range(frames):
            pot = np.fft.ifft2(lenia_p_kernel_fft * np.fft.fft2(world))
            for j, k in pot_roll_iteratable:
                pot[j,k] = pot[((j+12)%world_size),((k+12)%world_size)]

            updater = dt*(2*np.exp((-((pot-m)/s)**2)/2) - 1)
        
            world = np.minimum(1, np.maximum(world.real + updater.real, 0))
        

        return world.real
    
    #time function and call output
    pre = perf_counter()
    output = numba_lenia(world_size=world_size, R=R, T=T, frames=100, m=0.15, s=0.015)
    post = perf_counter()
    print(post-pre)
    time_set.append(post-pre)


# %% [markdown]
# Load arrays from csv file to avoid having to recalculate (which would take 50 minutes):

# %%
time_set = np.loadtxt("time_set_numba.csv", delimiter=",", dtype=float)
N_set = np.loadtxt("N_set_numba.csv", delimiter=",", dtype=float)

# %%
#fitting the predicted complexity relation to the data using scipy's curve_fit
def log_for_fit(x, a, b, c):
    return a*x**2*np.log(b*x) + c

popt = sc_opt.curve_fit(log_for_fit, N_set, time_set)[0]
a_opt, b_opt, c_opt = popt[0], popt[1], popt[2]

# %%
#arrays for plotting fitted prediction
x = np.linspace(0,1000,1000)
y = a_opt*x**2*np.log(b_opt*x) + c_opt

#calculating the moving average
mask=np.ones((1,20))/20
mask=mask[0,:]
move_average = np.convolve(time_set, mask, "same")


# %%
#calculating the coefficients of determination for the raw data and the moving average
residuals_raw = time_set - log_for_fit(N_set, a_opt, b_opt, c_opt)
res_ss_raw = np.sum(residuals_raw**2)
ss_tot_raw = np.sum((time_set - np.mean(time_set))**2)
r_squared_raw = 1 - (res_ss_raw / ss_tot_raw)

print("R^2 of fit on raw data = ", r_squared_raw)

residuals_move_avg = move_average - log_for_fit(N_set, a_opt, b_opt, c_opt)
res_ss_move_avg = np.sum(residuals_move_avg**2)
ss_tot_move_avg = np.sum((move_average - np.mean(move_average))**2)
r_squared_move_avg = 1 - (res_ss_move_avg / ss_tot_move_avg)

print("R^2 of fit on moving average data = ", r_squared_move_avg)

# %%
#plotting the raw data and moving average, and the fitted prediction
print(len(time_set))
print(len(N_set))
plt.scatter(N_set, time_set, label="Data")
plt.plot(N_set, move_average, label="Moving Average", c="tab:red")
plt.plot(x,y, label="Fitted N^2logN", c="tab:orange")
plt.legend()
plt.xlabel("World Size")
plt.ylabel("Processing Time/s")
plt.title("Numba")
#plt.savefig("numba N plot_100.png", dpi=1000)

# %% [markdown]
# Parameter Map Numba:

# %%
"""
Again, because of Numba's differences from the base python environment, this process has to be 
written less elegantly than the NumPy equivalent.
"""

#setting up the parameter space to test
total_loci = 100
map_size = int(np.sqrt(total_loci))
map_array = np.zeros((map_size, map_size))
mu_set = np.linspace(0.1, 0.5, map_size)
std_set = np.linspace(0.001, 0.12, map_size)

#unchanged parameters
R = 13
world_size = 100
T = 10

#pre-computing kernel and its fft
lenia_p_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
lenia_p_kernel = (lenia_p_kernel<1) * np.exp(4 - 4/(4*lenia_p_kernel*(1-lenia_p_kernel)))
##lenia_p_kernel = (lenia_p_kernel<1) * bert_exp(x=lenia_p_kernel, a=4)
lenia_p_kernel = lenia_p_kernel / np.sum(lenia_p_kernel)
#its fft:
pad_shape = world_size - np.array(lenia_p_kernel.shape)
lenia_p_kernel_padded = np.pad(lenia_p_kernel, [(0, i) for i in pad_shape], mode='constant')

lenia_p_kernel_fft = np.fft.fft2(lenia_p_kernel_padded)

pot_roll_iteratable = np.array(list(it.product(range(world_size), range(world_size))))

#initialising the map array
map_array = np.zeros((map_size, map_size))

for mu, std in it.product(mu_set, std_set):
    
    @jit(nopython=True)
    def numba_lenia(world_size, R, T, frames, m, s):
        dt = 1/T

        world = np.random.rand(world_size, world_size)

        for i in range(frames):
            pot = np.fft.ifft2(lenia_p_kernel_fft * np.fft.fft2(world))
            for j, k in pot_roll_iteratable:
                pot[j,k] = pot[((j+12)%world_size),((k+12)%world_size)]

            updater = dt*(2*np.exp((-((pot-m)/s)**2)/2) - 1)
        
            world = np.minimum(1, np.maximum(world.real + updater.real, 0))
        

        return world.real
    
    output = numba_lenia(world_size=world_size, R=R, T=T, frames=150, m=mu, s=std)

    if np.all(output == 0):
        map_array[np.where(mu_set == mu), np.where(std_set == std)] = 0
    else:
        map_array[np.where(mu_set == mu), np.where(std_set == std)] = 1

    


# %% [markdown]
# Load map array from NumPy data file:

# %%
map_array = np.load("map_array.npy")

# %%
"""
Plotting the map_array with modified axis labels to correctly represent the mean and standard deviation.

The outcome boundary is also plotted.
"""


fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.imshow(map_array)
ax.invert_yaxis()
ax.text(10,80, "Zero")
ax.text(60,50, "Non-Zero", c="white")
ax.set_xlabel("Standard Deviation")
ax.set_ylabel("Mean")

std_tick_locs = np.arange(0,110,10)
std_tick_lbls = np.round(np.linspace(0, 0.12, 11), 2)
plt.xticks(std_tick_locs, std_tick_lbls)

mu_tick_locs = np.arange(0,110,10)
mu_tick_lbls = np.round(np.linspace(0.1, 0.5, 11), 2)
plt.yticks(mu_tick_locs, mu_tick_lbls)


x=np.linspace(7, 31, 100)
y = 4.1*(x-4)-12
ax.plot(x,y, "tab:red", label = "Outcome Boundary")
ax.legend()
#plt.savefig("map_array.png", dpi=1000)


