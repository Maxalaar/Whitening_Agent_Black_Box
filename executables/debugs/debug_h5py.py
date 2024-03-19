from mpi4py import MPI

if __name__ == "__main__":
    print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())