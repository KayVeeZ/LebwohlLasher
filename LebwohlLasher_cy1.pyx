import numpy as np
import time
import sys
from cython.parallel import prange

cdef double[:, ::1] energy_cy(double[:, ::1] spins, int nmax):
    cdef int i, j
    cdef double[:, ::1] energy = np.zeros((nmax, nmax), dtype=np.float64)

    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            energy[i, j] = -1.0 * (spins[i, j] * (spins[(i + 1) % nmax, j] + spins[i, (j + 1) % nmax] + spins[(i - 1) % nmax, j] + spins[i, (j - 1) % nmax]))

    return energy

cdef double[:, ::1] mc_step(double[:, ::1] spins, double Ts, int nmax):
    cdef int i, j
    cdef double[:, ::1] energy = energy_cy(spins, nmax)
    cdef double[:, ::1] new_spins = np.copy(spins)
    cdef double[:, ::1] order = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] ran = np.random.rand(nmax, nmax)

    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            if ran[i, j] < np.exp(-2.0 * Ts * energy[i, j]):
                new_spins[i, j] *= -1.0
                order[i, j] = 1.0
            else:
                order[i, j] = 0.0

    return new_spins, order

def run_simulation(int nmax, int nsteps, double Ts):
    cdef int i, j, step
    cdef double[:, ::1] spins = np.random.choice([-1.0, 1.0], size=(nmax, nmax))
    cdef double[:, ::1] order
    cdef double[:, ::1] new_spins
    cdef double[:, ::1] energy
    cdef double[:, ::1] order_sum = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] energy_sum = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] order_ratio = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] energy_ratio = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] order_avg = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] energy_avg = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] order_avg_sum = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] energy_avg_sum = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] order_avg_ratio = np.zeros((nmax, nmax), dtype=np.float64)
    cdef double[:, ::1] energy_avg_ratio = np.zeros((nmax, nmax), dtype=np.float64)

    for step in range(nsteps):
        new_spins, order = mc_step(spins, Ts, nmax)
        energy = energy_cy(new_spins, nmax)
        order_sum += order
        energy_sum += energy

        order_ratio = order_sum / (step + 1)
        energy_ratio = energy_sum / (step + 1)

        order_avg = (order_sum / (step + 1))[::2, ::2]
        energy_avg = (energy_sum / (step + 1))[::2, ::2]

        order_avg_sum += order_avg
        energy_avg_sum += energy_avg

        order_avg_ratio = order_avg_sum / (step + 1)
        energy_avg_ratio = energy_avg_sum / (step + 1)

    return spins, order_avg_ratio, energy_avg_ratio

def main(int nmax, int nsteps, double Ts):
    cdef double[:, ::1] spins
    cdef double[:, ::1] order_avg_ratio
    cdef double[:, ::1] energy_avg_ratio

    spins, order_avg_ratio, energy_avg_ratio = run_simulation(nmax, nsteps, Ts)

    print("Spins:")
    print(spins)

    print("Order Ratio:")
    print(order_avg_ratio)

    print("Energy Ratio:")
    print(energy_avg_ratio)

if __name__ == '__main__':
    nmax = int(sys.argv[1])
    nsteps = int(sys.argv[2])
    Ts = float(sys.argv[3])

    main(nmax, nsteps, Ts)