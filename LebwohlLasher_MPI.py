from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import sys
import os
import shutil
#=======================================================================
def initdat(nmax, comm):
    """
    Initialize the lattice data array.

    Args:
    - nmax: int, size of lattice to create (nmax x nmax).
    - comm: MPI communicator.

    Returns:
    - arr: numpy array, lattice data array.
    """
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================

def all_energy(arr, nmax):
    """
    Compute the energy of the entire lattice.

    Args:
    - arr: numpy array, lattice data array.
    - nmax: int, side length of square lattice.

    Returns:
    - enall: float, reduced energy of lattice.
    """
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall
#=======================================================================
def get_order(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
def MC_step(arr, Ts, nmax):
    """
    Perform one Monte Carlo step.

    Args:
    - arr: numpy array, lattice data array.
    - Ts: float, reduced temperature (range 0 to 2).
    - nmax: int, side length of square lattice.

    Returns:
    - accept_ratio: float, acceptance ratio for current MCS.
    """
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))
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
#=======================================================================
def main(comm, program, nsteps, nmax, temp, pflag):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute workload among processes
    local_nmax = nmax // size

    # Initialize lattice data for this process
    local_lattice = initdat(local_nmax, comm)

    # Perform Monte Carlo steps
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy(local_lattice, local_nmax)
    ratio[0] = 0.5
    order[0] = get_order(local_lattice, local_nmax)

    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(local_lattice, temp, local_nmax)
        energy[it] = all_energy(local_lattice, local_nmax)
        order[it] = get_order(local_lattice, local_nmax)

    # Gather results from all processes
    global_energy = None
    global_ratio = None
    global_order = None

    if rank == 0:
        global_energy = np.zeros((size, nsteps + 1), dtype=np.float64)
        global_ratio = np.zeros((size, nsteps + 1), dtype=np.float64)
        global_order = np.zeros((size, nsteps + 1), dtype=np.float64)

    comm.Gather(energy, global_energy, root=0)
    comm.Gather(ratio, global_ratio, root=0)
    comm.Gather(order, global_order, root=0)

    final = time.time()
    runtime = final - initial

    if rank == 0:
        # Final outputs
        fin1 = "{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax, nsteps, temp, global_order[-1][-1], runtime)
        print(fin1)

        # Save results to file
        save_results(global_energy, global_ratio, global_order, nsteps, temp, runtime, nmax)

        # Plot final frame of lattice
        plotdat(local_lattice, pflag, nmax)
#=======================================================================
def save_results(energy, ratio, order, nsteps, temp, runtime, nmax):
    """
    Save the energy, order, and acceptance ratio per Monte Carlo step to a text file.

    Args:
    - energy: numpy array, array of energies per MCS.
    - ratio: numpy array, array of acceptance ratios per MCS.
    - order: numpy array, array of order parameters per MCS.
    - nsteps: int, number of Monte Carlo steps.
    - temp: float, reduced temperature.
    - runtime: float, runtime of simulation.
    - nmax: int, side length of square lattice.
    """
    # Create subdirectory based on parameters
    path1 = f"iter_{nsteps}_temp_{temp}_side_{nmax}"
    if not os.path.exists(path1):
        os.makedirs(path1)

    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"{path1}/LL-Output-{current_datetime}.txt"

    with open(filename, "w") as f:
        print("#=====================================================", file=f)
        print("# File created:        {:s}".format(current_datetime), file=f)
        print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=f)
        print("# Number of MC steps:  {:d}".format(nsteps), file=f)
        print("# Reduced temperature: {:5.3f}".format(temp), file=f)
        print("# Run time (s):        {:8.6f}".format(runtime), file=f)
        print("#=====================================================", file=f)
        print("# MC step:  Ratio:     Energy:   Order:", file=f)
        print("#=====================================================", file=f)
        for i in range(nsteps + 1):
            print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, ratio[0, i], energy[0, i], order[0, i]), file=f)
#=======================================================================
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(comm, PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        if rank == 0:
            print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
