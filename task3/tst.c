#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
 
 int rank, proc_num;
  
  int main(int argc, char **argv)
  {
	  int error = MPI_Init(&argc, &argv);
	  if (error != MPI_SUCCESS) {
	  	fprintf (stderr, "MPI_Init error \n");
								                  return 1;
												          }
														          MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
																		          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
																				          printf("%d / %d\n", rank, proc_num);
#pragma omp parallel
																						  {
																							          printf("%d / %d: %d\n", rank, proc_num, omp_get_num_threads());
																						  }
																						          MPI_Finalize();
																								          return 0;
  }
