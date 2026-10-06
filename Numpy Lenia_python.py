# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import scipy.signal as signal
import itertools as it
import warnings
from time import gmtime, strftime, perf_counter
import scipy.optimize as sc_opt

# %%
def display(initial_state, update_fn,save=False):
    """
    Displays an animation of the lattice evolution given an initial state and an update function (in our case the local rule).
    The function can save the animation as an mp4 file.
    Uses matplotlib's FuncAnimation and IPython's HTML display.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(initial_state, animated=True)
    global lattice
    lattice = initial_state
    def update(frame):
        global lattice
        lattice = update_fn(lattice)
        im.set_array(np.real(lattice))
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=150, interval = 50, blit=True)
    FFwriter = animation.FFMpegWriter(fps=30)
    if save:
        time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        
        ani.save('lenia_'+time_str+'.mp4', writer = FFwriter)

    jshtml = ani.to_jshtml()
    return HTML(jshtml)

def K_G_display(kernel, growth, x, kernel_type="continuous", save=False):
    """
    Displays the kernel and growth function, just for visualization.

    The kernal is displayed via a colour map and a cross section.
    Can also save the plot as a png file.
    """
    
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    ax[0].imshow(kernel)
    ax[0].set_title("Kernel")
    ax[1].step(x, growth(x))
    ax[1].set_title("Growth function")
    ax[1].set_xlabel("Potential")
    ax[1].set_ylabel("Growth function(Potential)")

    k_size = kernel.shape[0]
    if kernel_type == "continuous":
        ax[2].plot(range(k_size), kernel[k_size//2, :])
    elif kernel_type == "discrete":
        ax[2].bar(range(k_size), kernel[k_size//2, :], width=1)
    else:
        raise ValueError("Invalid kernel type")
    ax[2].set_title("Kernel slice")
    
    if save:
        time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        plt.savefig("K_G_display_" + time_str + ".png")
    plt.show()

def kernel_fft(kernel, world_size):
    """
    Returns the FFT of the kernel, padded to the size of the world.
    Returns a dictionary containing the unshifted and shifted FFTs.
    Has some error checking.
    """

    if kernel.shape[0] > world_size or kernel.shape[1] > world_size:
        raise ValueError("Kernel is larger than the world size")
    elif (world_size-kernel.shape[0]) % 2 != 0 or (world_size-kernel.shape[1]) % 2 != 0:
        raise ValueError("Differences in sizes is not even)")
    elif kernel.shape == (world_size, world_size):
        warnings.warn("Kernel is the same size as the world, no padding needed")
        return np.real(np.fft.fft2(kernel))
    else:
        #padded_kernel = np.pad(kernel, (((world_size-kernel.shape[0])//2, (world_size-kernel.shape[0])//2), ((world_size-kernel.shape[1])//2, (world_size-kernel.shape[1])//2)), 'constant')
        pad_shape = world_size - np.array(kernel.shape)
        kernel_padded = np.pad(kernel, [(0, i) for i in pad_shape], mode='constant')
        
        fft_kernel = np.fft.fft2(kernel_padded)
        fft_kernel_shift = np.fft.fftshift(fft_kernel)
        
        return {
            "unshifted": fft_kernel,
            "shifted": fft_kernel_shift
        }




# %% [markdown]
# Lenia Kernel and Growth Function:

# %%
#temporal (T) and spatial (R) resolution parameters:
T = 10
dt = 1/T

size = 100
if size % 2 != 0:
    raise ValueError("Size must be even")
R = 13


#growth function parameters:
mu = 0.15
std = 0.015


#setting up the kernal, a gaussian bump
bert_exp = lambda x, a: np.exp(a - a/(4*x*(1-x)))
lenia2_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
lenia2_kernel = (lenia2_kernel<1) * bert_exp(x=lenia2_kernel, a=4)
lenia2_kernel = lenia2_kernel / np.sum(lenia2_kernel)

#call FFT function
lenia2_kernel_fft = kernel_fft(lenia2_kernel, size)["unshifted"]

#setting up the growth function
guassian = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)
def lenia2_growth(pot):
    return guassian(x=pot, m=mu, s=std)*2 - 1

#method to use for the convolution
method = "fft"

def lenia2_local_rule(i):
    """
    The local rule defining Lenia's time step evolution.
    Uses one of three methods to convolve the kernel with the world.

    This version of the local rule is specifically for use with the display function.
    """
    global world
    if method == "fft":
        #fastest method (at large N) using FFT convolution theorem
        pot = np.fft.ifft2(lenia2_kernel_fft * np.fft.fft2(world))
        pot = np.roll(np.roll(pot, -12, axis=0),-12, axis=1)
       

    elif method == "direct":
        #uses direct convolution
        pot = signal.convolve2d(world, lenia2_kernel, mode='same', boundary='wrap')

    elif method == "scihub fft":
        #uses scipy.signal.fftconvolve, which is worse because it computes the kernel FFT every time
        pot = signal.fftconvolve(world, lenia2_kernel, mode='same')
    else:
        raise ValueError("Invalid method")
    
    world = np.clip(world + dt*lenia2_growth(pot), 0, 1)
    return world


# %%
#display kernel and growth function of Lenia
K_G_display(kernel=lenia2_kernel, growth=lenia2_growth, x=np.linspace(0, 1, 1000), save=False)

# %%
#initialise random initial world and call display function
world = np.random.rand(size, size)
display(world, lenia2_local_rule, save=False)

# %% [markdown]
# Orbium:

# %%
#initiallising world with pre-defined Orbium pattern, and display
orbium_pattern = np.asarray([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
initial_position = (50, 50)

world = np.zeros((size, size))
world[initial_position[0]:initial_position[0]+orbium_pattern.shape[0], initial_position[1]:initial_position[1]+orbium_pattern.shape[1]] = orbium_pattern


display(world, lenia2_local_rule, save=False)


# %% [markdown]
# Parameter Map and Scaling Relation:

# %%
def lenia_local_rule_for_outcome(world_in, growth_fn, mu, std):
    """
    Lenia's local rule, via FFT, specifically for use with the find_outcome function.
    Takes growth function as input allowing automatic variation of mu and std.
    """
    pot = np.fft.ifft2(lenia2_kernel_fft * np.fft.fft2(world_in))
    pot = np.roll(np.roll(pot, -12, axis=0),-12, axis=1)
    world_out = np.clip(world_in + dt*growth_fn(pot, mu, std), 0, 1)
    return world_out

def find_outcome(mu, std, local_rule, initial_state, find_length):
    """
    Finds if the world reaches the uniform desert (all cells 0) within find_length iterations.
    Calls on the lenia_local_rule_for_outcome function.
    Returns:
    0 if uniform desert is reached
    1 if uniform ones is reached
    -1 if the world is not uniform after find_length iterations
    """
    world = initial_state

    
    growth_function = lambda x,mu, std: guassian(x, mu, std)*2 - 1
    
    for i in range(find_length):
        world = local_rule(world, growth_function, mu, std)
        if np.all(world == 0):
            return 0
        elif np.all(world == 1):
            return 1
        else:
            continue

    return -1

def lenia_local_rule_for_iterator(world_in, growth_fn, mu, std, kernel_fft):
    """
    Local rule for use with the iterator function, via FFT method.
    Takes the kernel FFT as inpput to allow efficient varying of world size.
    """
    pot = np.fft.ifft2(kernel_fft * np.fft.fft2(world_in))
    pot = np.roll(np.roll(pot, -12, axis=0),-12, axis=1)
    world_out = np.clip(world_in + dt*growth_fn(pot, mu, std), 0, 1)
    return world_out

def lenia_local_rule_for_iterator_direct_conv(world_in, growth_fn, mu, std, kernel):
    """
    Local rule for use with the iterator function, using scipy's direct convolution function.
    Takes the kernel FFT as inpput to allow efficient varying of world size.
    """
    pot = signal.convolve2d(world_in, kernel, mode='same', boundary='wrap')
    world_out = np.clip(world_in + dt*growth_fn(pot, mu, std), 0, 1)
    return world_out
        
def iterator(initial_state, local_rule, mu, std, iterations, kernel_fft):
    """
    Simple auxiliary function to iterate the local rule over the world.
    """
    world = initial_state
    growth_function = lambda x,mu, std: guassian(x, mu, std)*2 - 1
    for i in range(iterations):
        world = local_rule(world, growth_function, mu, std, kernel_fft)
    return world.real


# %% [markdown]
# Scaling relation with N:

# %%
#set unchanged parameters
R = 13
T = 10 
dt = 1/T

#set of world sizes to test
N_set = np.arange(30,1000,2)
time_set = []

for N in N_set:

    #use time package to time each world
    pre = perf_counter()
    
    #pre-computing kernel and its fft
    lenia_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
    lenia_kernel = (lenia_kernel<1) * np.exp(4 - 4/(4*lenia_kernel*(1-lenia_kernel)))
    lenia_kernel = lenia_kernel / np.sum(lenia_kernel)
    #its fft:
    pad_shape = N - np.array(lenia_kernel.shape)
    lenia_kernel_padded = np.pad(lenia_kernel, [(0, i) for i in pad_shape], mode='constant')
    lenia_kernel_fft = np.fft.fft2(lenia_kernel_padded)

    initial = np.random.rand(N, N)
    
    #call iterator and time
    out = iterator(initial, lenia_local_rule_for_iterator, mu, std, 100, lenia_kernel_fft)
    post = perf_counter()
    time_set.append(post-pre)
    

# %% [markdown]
# Load from file to avoid recalculating:

# %%
time_set = np.loadtxt("time_set_numpy.csv", delimiter=",", dtype=float)
N_set = np.loadtxt("N_set_numpy.csv", delimiter=",", dtype=float)

# %%
#compute moving average via convolution
mask=np.ones((1,20))/20
mask=mask[0,:]
move_average = np.convolve(time_set, mask, "same")

# %%
#fitting predicted complexity relationship with scipy's curve_fit
def log_for_fit(x, a, b, c):
    return a*x**2*np.log(b*x) + c

popt = sc_opt.curve_fit(log_for_fit, N_set, time_set)[0]
a_opt, b_opt, c_opt = popt[0], popt[1], popt[2]

# %%
#computing coefficient of determination for both raw and moving average data
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
#arrays for plotting fitted curve
x = np.linspace(0,1000,1000)
y = a_opt*x**2*np.log(b_opt*x) + c_opt

# %%
#plot data, moving average, and fitted curve
plt.scatter(N_set, time_set, label="Data")
plt.plot(N_set, move_average, label="Moving Average", c="tab:red")
plt.plot(x,y, label="Fitted N^2logN", c="tab:orange")
plt.legend()
plt.xlabel("World Size")
plt.ylabel("Processing Time/s")
plt.title("Numpy")
#plt.savefig("numpy N plot.png", dpi=1000)

# %%
#pulling in data from Numba algirthm (saved to file) to compare NumPy and Numba
time_set_numba = np.loadtxt("time_set_numba.csv", delimiter=",", dtype=float)
N_set_numba = np.loadtxt("N_set_numba.csv", delimiter=",", dtype=float)

# %%
#moving average (reusing mask)
move_average_numba = np.convolve(time_set_numba, mask, "same")

# %%
#Plotting NumPy and Numba data to compare
plt.plot(N_set, move_average, label="Numpy")
plt.plot(N_set_numba, move_average_numba, label="Numba")
plt.legend()
plt.xlabel("World Size")
plt.ylabel("Processing Time/s")
#plt.savefig("numpy vs numba_2.png", dpi=1000)

# %% [markdown]
# Time vs N: Direct Convolution
# 

# %%
#setting unchanged variables
R = 13
T = 10 
dt = 1/T

mu = 0.15
std = 0.015

#pre-computing kernel
lenia_kernel = np.linalg.norm((np.mgrid[-R:R, -R:R]+1) / R, axis=0)
lenia_kernel = (lenia_kernel<1) * np.exp(4 - 4/(4*lenia_kernel*(1-lenia_kernel)))
lenia_kernel = lenia_kernel / np.sum(lenia_kernel)

#setting up the set of world sizes to test
N_set_direct = np.arange(30,1000,2)
time_set_direct = []

#same process as for NumPy and Numba, but using direct convolution
for N in N_set_direct:


    pre = perf_counter()
    
    initial = np.random.rand(N, N)
    
    out = iterator(initial, lenia_local_rule_for_iterator_direct_conv, mu, std, 100, lenia_kernel)
    post = perf_counter()
    time_set_direct.append(post-pre)

# %% [markdown]
# Load data from file to avoid recalculating

# %%
N_set_direct = np.loadtxt("N_set_direct.csv", delimiter=",", dtype=float)
time_set_direct = np.loadtxt("time_set_direct.csv", delimiter=",", dtype=float)

# %%
#fitting predicted complexity relationship with scipy's curve_fit
def quad_for_fit(x, a, b):
    return a*x**2 + b

popt_direct = sc_opt.curve_fit(quad_for_fit, N_set_direct, time_set_direct)[0]
a_opt_direct, b_opt_direct = popt_direct[0], popt_direct[1]

# %%
#computing coefficient of determination for the direct data, no moving average is computed as the data is less noisy than for NumPy or Numba
residuals_direct = time_set_direct - quad_for_fit(N_set_direct, a_opt_direct, b_opt_direct)
res_ss_direct = np.sum(residuals_direct**2)
ss_tot_direct = np.sum((time_set_direct - np.mean(time_set_direct))**2)
r_squared_direct = 1 - (res_ss_direct / ss_tot_direct)

print("R^2 of fit on direct data = ", r_squared_direct)

# %%
#arrays to be plotted of fitted curve
x = np.linspace(0,1000,1000)
y_direct = a_opt_direct*x**2 + b_opt_direct

# %%
#plotting data and fitted curve
plt.scatter(N_set_direct, time_set_direct, label="Data")
plt.plot(x,y_direct, label="Fitted N^2", c="tab:orange")
plt.legend()
plt.xlabel("World Size")
plt.ylabel("Processing Time/s")
plt.title("Direct Convolution")
#plt.savefig("direct N plot.png", dpi=1000)

# %% [markdown]
# Comparing NumPy FFT method and direct convolution:

# %%
#plot NumPy FFT moving average data and direct convolution data to compare
plt.plot(N_set, move_average, label="Numpy FFT")
plt.plot(N_set_direct, time_set_direct, label="Direct")
plt.legend()
plt.xlabel("World Size")
plt.ylabel("Processing Time/s")
plt.title("Numpy FFT vs Direct")
#plt.savefig("numpy fft vs direct.png", dpi=1000)


# %% [markdown]
# Parameter Map:

# %%
#setting up the set of worlds to test, with varied mu and std, and choosing the number of data points
total_loci = 100
map_size = int(np.sqrt(total_loci))
map_array = np.zeros((map_size, map_size))
mu_set = np.linspace(0.1, 0.5, map_size)
std_set = np.linspace(0, 0.12, map_size)


#iterate over parameter space calling find_outcome function
for i, j in it.product(range(map_size), range(map_size)):
    mu = mu_set[i]
    std = std_set[j]
    map_array[i, j] = find_outcome(mu, std, lenia_local_rule_for_outcome, np.random.rand(100, 100), 150)

# %%
plt.matshow(map_array)
plt.colorbar()


