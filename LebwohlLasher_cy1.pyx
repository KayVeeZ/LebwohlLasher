
import numpy as np

cimport numpy as np



def one_energy(np.ndarray[np.float64_t, ndim=2] arr, int ix, int iy, int nmax):
    cdef double en = 0.0
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1) % nmax
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1) % nmax
    cdef double ang

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    return en

cpdef double all_energy(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    cdef double enall = 0.0
    cdef int i, j

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall


cpdef double get_order(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    cdef double[:, :] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:, :] delta = np.eye(3, dtype=np.float64)
    cdef double[:, :, :] lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    cdef int a, b, i, j

    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]

    # Convert Qab to a NumPy array before performing the division
    cdef np.ndarray[np.float64_t, ndim=2] Qab_np = np.asarray(Qab)
    Qab_np /= (2 * nmax * nmax)

    # Return the maximum eigenvalue of Qab
    return np.linalg.eigvals(Qab_np).max()


cpdef MC_step(np.ndarray[np.float64_t, ndim=2] arr, double Ts, int nmax):
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    cdef np.ndarray[int, ndim=2] xran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] yran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    return accept / (nmax * nmax)


