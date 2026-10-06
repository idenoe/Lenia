## Abstract

The cellular automaton Lenia is implemented in Python, making use of NumPy and Numba to compile code into lower-level languages to improve efficiency. The computational complexities of multiple Lenia algorithms are predicted and verified experimentally. The relationship between growth function parameters and world outcome is investigated by mapping the outcome of worlds at many parameter choices. Based on this investigation, a rule for determining the outcome of a world from the parameter choices is proposed. The implementation is verified by successfully recreating results from previous researchers. The methods of: FFT convolution, NumPy vectorisation, and Numba compiling are used and each one is found to significantly improve efficiency.

## 1.1 Description of code structure

The files attached are:

| File Name | Description |
| --- | --- |
| `lenia_report.pdf` | Full project report |
| `environment.yml` | YAML Conda environment file for both implementations |
| `Numpy Lenia.ipynb` | Jupyter Notebook containing NumPy Lenia |
| `Numba Lenia.ipynb` | Jupyter Notebook containing Numba Lenia |
| `Numpy Lenia_python.py` | `Numpy Lenia.ipynb` converted to .py file |
| `Numba Lenia_python.py` | `Numba Lenia.ipynb` converted to .py file |
| `orbium.mp4` | Animation of the species *Orbium*, referenced in Figure 3 |
| `lenia_demo.mp4` | Animation of Lenia, referenced in Section 4.1 |
| `gol_demo.mp4` | Animation of Conway's Game of Life, referenced in section 4.1 |
| `N_set_XXXXX.csv` (5 files) | Comma delimited text files containing the set of N values tested where `XXXXX` is the method name |
| `time_set_XXXXX.csv` (5 files) | Comma delimited text files containing the set of time values tested where `XXXXX` is the method name |
| `map_array.npy` | NumPy file containing the array of outcome codes for the parameter map |

This project was completed exclusively in Jupyter Notebooks and is intended to be used as such, a regular python file is included.

## 1.2 NumPy Lenia

The code for Lenia using NumPy is in the notebook "`Numpy Lenia.ipynb`". It has the required packages:

| Package | Nickname in project | Description |
| --- | --- | --- |
| `NumPy` | `np` | Workhorse of this implementation, runs key functions in faster C environment. |
| `matplotlib.pyplot` | `plt` | Used for plotting of timing data and parameter map. |
| `matplotlib.animation` | `animation` | Used for animation of Lenia. |
| `IPython.display.HTML` | `HTML` | Used to display Lenia animations as interactive HTML objects. |
| `scipy.signal` | `signal` | Used to compute 2D FFT Convolution, for testing only. |
| `itertools` | `it` | Used for iterating over multiple iterables. |
| `warnings` | `warnings` | Used to raise warnings to the user. |
| `time.gmtime`, `strftime` | `gmtime`, `strftime` | Used for naming saved files. |
| `time.perf_counter` | `perf_counter` | Used for timing functions. |
| `scipy.optimize` | `sc_opt` | Used to fit the predicted complexity curve to experimental data. |

The first functions defined in the Numpy Lenia project are utility functions for displaying animations, displaying the kernel and growth function, and computing the kernel's FFT.

Next, Lenia's kernel, growth function, and local rule are defined using NumPy's FFT function. Lenia is then animated.

Various functions for use in computing the parameter map are defined, most importantly `find_outcome` that returns the outcome of a world within a specified number of frames.

The scaling relation with N is investigated using the iterator function.

The parameter map is computed using the previously mentioned functions.

## 1.3 Numba Lenia

The code for Lenia using Numba compiling is in the notebook "`Numba Lenia.ipynb`". It has the required packages:

| Package | Nickname in project | Description |
| --- | --- | --- |
| `NumPy` | `np` | Workhorse of this implementation, runs key functions in faster C environment. |
| `matplotlib.pyplot` | `plt` | Used for plotting of timing data and parameter map. |
| `time.perf_counter` | `perf_counter` | Used for timing functions. |
| `scipy.optimize` | `sc_opt` | Used to fit the predicted complexity curve to experimental data. |
| `numba.jit` | `jit` | Decorator used to call Numba to compile Python code into machine language. |
| `rocket_fft` | N/A | An auxiliary package that makes Numba aware of NumPy's 2D FFT function and allows it to compile it. |

Each task requires the Lenia local rule to be re-defined to allow Numba to compile it in each case. The Numba code is therefore less readable that the NumPy code, but follows a very similar structure.
