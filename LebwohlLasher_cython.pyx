import numpy as np
cimport numpy as cnp
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
from cython cimport wraparound, boundscheck

@boundscheck(False)
@wraparound(False)
cdef float cy_one_energy(cnp.ndarray[cnp.float64_t, ndim=2] arr, int ix, int iy, int nmax):
    cdef float en = 0.0
    cdef int ixp, ixm, iyp, iym
    cdef float ang
    ixp = (ix+1)%nmax
    ixm = (ix-1)%nmax
    iyp = (iy+1)%nmax
    iym = (iy-1)%nmax
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en

@boundscheck(False)
@wraparound(False)
cdef float cy_all_energy(cnp.ndarray[cnp.float64_t, ndim=2] arr, int nmax):
    cdef float enall = 0.0
    cdef int i, j
    for i in range(nmax):
        for j in range(nmax):
            enall += cy_one_energy(arr, i, j, nmax)
    return enall

@boundscheck(False)
@wraparound(False)
cdef float cy_get_order(cnp.ndarray[cnp.float64_t, ndim=2] arr, int nmax):
    cdef cnp.ndarray[double, ndim=2] Qab = np.zeros((3,3))
    cdef cnp.ndarray[double, ndim=2] delta = np.eye(3,3)
    cdef cnp.ndarray[double, ndim=3] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    cdef int a, b, i, j
    cdef double val
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab /= 2*nmax*nmax
    eigenvalues = np.linalg.eigvals(Qab)
    val = eigenvalues.max()
    return val

@boundscheck(False)
@wraparound(False)
def cy_MC_step(cnp.ndarray[cnp.float64_t, ndim=2] arr, float Ts, int nmax):
    cdef float accept = 0
    cdef float scale = 0.1 + Ts
    cdef cnp.ndarray[int, ndim=2] xran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=int)
    cdef cnp.ndarray[int, ndim=2] yran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=int)
    cdef cnp.ndarray[float, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    cdef int ix, iy
    cdef float ang, en0, en1, boltz
    for ix in range(nmax):
        for iy in range(nmax):
            ang = aran[ix,iy]
            en0 = cy_one_energy(arr, ix, iy, nmax)
            arr[ix,iy] += ang
            en1 = cy_one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= rand() / RAND_MAX:
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept / (nmax * nmax)
