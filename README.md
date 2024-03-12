# Lebwohl Lasher Simulation model
This repo covers acceleration of this titled model using various approaches and covers using version control to basically cover problem solving, i.e. the need to backtrack, etc.

## I used setup.py to compile the cython file.
#### And used this command to build: python setup.py build_ext --inplace

## I used this command to run the mpi4py code:
#### mpiexec -n 4 python LebwohlLasher_MPI.py 1000 20 0.65 0
