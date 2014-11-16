#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define PI 3.14159265

double maxeps = 0.01;
int N = 512;
// spectral radius
int itmax = 100000;
int proc_size;
int proc_rank;
int box_size_i;
int box_size_j;
MPI_Comm topology;
double eps;
double** U = NULL;
void relax();
void init();
void verify();
void wtime();

void verify()
{ 
	double s;
	s = 0.;
	int i, j;
	for(i = 0; i < N+2; i++)
		for(j = 0; j < N+2; j++)
			{
				s = s + U[i][j] * (i + 1) * (j + 1) / (N * N);
			}
	printf(" S = %f\n",s);
}
void wtime(double *t)
{
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, (void *)0);
	if (sec < 0) 
		sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

double omega(int n)
{
    double p = 1. - (PI / ((N - 1)))*(PI / ((N - 1))); 
	if (n == 0)
		return 0;
	else if (n == 1)
	{
		return 1./(1. - p*p/2); 
	}
	else
	{
		return 1./(1. - omega(n-1)*p*p/4); 
	}
}
void output(const char* name)
{
	// write matrix to file
	FILE* grid = fopen(name, "w");
	int i, j;
	for (i = 0; i < N+2; i++)
	{
		for (j = 0; j < N+2; j++)
		{
			fprintf(grid, "%lf ", U[i][j]);
		}
		fprintf(grid, "\n");
	}
	fclose(grid);
}
void init()
{
	int coords[2], r;
    MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 2, coords);
	//printf( "Hello world from process %d of %d\n", proc_rank, proc_size );
	//printf( "Hello world from coords %d x %d\n", coords[0], coords[1]);
	int i, j;
	U = calloc(box_size_i + 2, sizeof(U[0]));
	for (i = 0; i < box_size_i + 2; i++) 
	{
		U[i] = calloc(box_size_j + 2, sizeof(U[i][0]));
	}
    for(i=0; i< box_size_i+2; i++)
        for(j=0; j< box_size_j+2; j++)
        { 
			int ti = box_size_i*coords[0] + i;
			int tj = box_size_j*coords[1] + j;

			double x = ti * (1.0 / (N + 1));
			double y = tj * (1.0 / (N + 1));
			
			if(tj == 0) 
            {    
				U[i][j] = sin(PI * x / 2);
            }
			else if (tj == N + 1)
			{
				U[i][j] = sin(PI * x / 2) * exp(-PI/2);
			}
			else if (ti == N + 1)
			{
				U[i][j] = exp(- PI * y / 2);
			}
			else 
				U[i][j] = 0;
				//U[i][j] = sin(PI * x / 2) * exp(-PI * y/2);
        }
}

// parameter: iteration number
void relax(int iter)
{
	int coords[2], r;
    MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 2, coords);
	MPI_Status status;
	MPI_Request request;
	
	double loc_eps = 0.;
    int i, j;
	double om = omega(iter);
	// recieve
	// Isend of the current iteration to the end of the cycle
	if (coords[0] - 1 >= 0) // top
	{
		int r_top, coords_top[2];
		coords_top[0] = coords[0] - 1;
		coords_top[1] = coords[1];
		MPI_Cart_rank(topology, coords_top, &r_top);
		//MPI_Isend(U[1] + 1, box_size_j, MPI_DOUBLE, r_top, 0,
		  //           topology, &request);
		MPI_Recv(U[0] + 1, box_size_j, MPI_DOUBLE, r_top, 0,
		             topology, &status);
		int count;
		MPI_Get_count (&status, MPI_DOUBLE, &count);
		if (count != box_size_j)
			printf("Error while recieving from top\n");
	}
	
	if (coords[1] - 1 >= 0) // left
	{
		int r_left, coords_left[2];
		coords_left[0] = coords[0];
		coords_left[1] = coords[1] - 1;
		MPI_Cart_rank(topology, coords_left, &r_left);
		double* buf = (double*) malloc (box_size_i * sizeof(double));
		for (i = 1 ; i < box_size_i + 1; i++)
			buf[i] = U[i][1];
		//MPI_Isend(buf, box_size_i, MPI_DOUBLE, r_left, 0,
		 //            topology, &request);
		MPI_Recv(buf, box_size_i, MPI_DOUBLE, r_left, 0,
		             topology, &status);
		for (i = 1 ; i < box_size_i + 1; i++)
			U[i][0] = buf[i];
		free(buf);
		int count;
		MPI_Get_count (&status, MPI_DOUBLE, &count);
		if (count != box_size_i)
			printf("Error while recieving from left\n");
	}
	
	// Me
	for(i=1; i < box_size_i+1; i++)
        for(j=1; j< box_size_j+1; j++)
        {
            double U_dop = (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1])/4.;
			double e;
			e = U[i][j];
            U[i][j] = om * U_dop + (1. - om) * U[i][j];
			loc_eps = loc_eps > fabs(e - U[i][j]) ? loc_eps : fabs(e - U[i][i]);
        }
	//if (proc_rank == 6)
    //   	printf( "Moment: it=%4i eps=%f\n", iter, loc_eps);
	// send
	
	// Isend of the current iteration to the end of the cycle
	if (coords[0] - 1 >= 0) // top
	{
		int r_top, coords_top[2];
		coords_top[0] = coords[0] - 1;
		coords_top[1] = coords[1];
		MPI_Cart_rank(topology, coords_top, &r_top);
		MPI_Isend(U[1] + 1, box_size_j, MPI_DOUBLE, r_top, 0,
		             topology, &request);
	}
	
	if (coords[1] - 1 >= 0) // left
	{
		int r_left, coords_left[2];
		coords_left[0] = coords[0];
		coords_left[1] = coords[1] - 1;
		MPI_Cart_rank(topology, coords_left, &r_left);
		double* buf = (double*) malloc (box_size_i * sizeof(double));
		for (i = 1 ; i < box_size_i + 1; i++)
			buf[i] = U[i][1];
		MPI_Isend(buf, box_size_i, MPI_DOUBLE, r_left, 0,
		             topology, &request);
	}
	
	if (coords[0] + 1 <= N / box_size_i - 1) // bottom
	{
		int r_b, coords_b[2];
		coords_b[0] = coords[0] + 1;
		coords_b[1] = coords[1];
		MPI_Cart_rank(topology, coords_b, &r_b);
		MPI_Isend(U[box_size_i] + 1, box_size_j, MPI_DOUBLE, r_b, 0,
		             topology, &request);
		// last iterations data
		MPI_Recv(U[box_size_i+1] + 1, box_size_j, MPI_DOUBLE, r_b, 0,
		             topology, &status);
		int count;
		MPI_Get_count (&status, MPI_DOUBLE, &count);
		if (count != box_size_j)
			printf("Error while recieving from bottom\n");
	}
	
	if (coords[1] + 1 <= N / box_size_j - 1) // right
	{
		int r_right, coords_r[2];
		coords_r[0] = coords[0];
		coords_r[1] = coords[1] + 1;
		MPI_Cart_rank(topology, coords_r, &r_right);
		double* buf = (double*) malloc (box_size_i * sizeof(double));
		for (i = 1 ; i < box_size_i + 1; i++)
			buf[i] = U[i][box_size_j];
		MPI_Isend(buf, box_size_i, MPI_DOUBLE, r_right, 0,
		             topology, &request);
		// previous iteration data
		MPI_Recv(buf, box_size_i, MPI_DOUBLE, r_right, 0,
		             topology, &status);
		for (i = 1 ; i < box_size_i + 1; i++)
			U[i][box_size_j+1] = buf[i];
		int count;
		MPI_Get_count (&status, MPI_DOUBLE, &count);
		if (count != box_size_i)
			printf("Error while recieving from right\n");
		free(buf);
	}
	//eps = loc_eps;
	//MPI_Barrier(topology);
	MPI_Allreduce(&loc_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, topology);
}

void work()
{
	if ((int)sqrt(proc_size) == sqrt(proc_size))
	{
		box_size_j = sqrt(N*N / proc_size);
		box_size_i = box_size_j;
	}
	else
	{
		box_size_j = sqrt(N*N / (proc_size / 2));
		box_size_i = box_size_j / 2;
		
	}
	if (proc_rank == 0)
	{
		//printf("Box size is %d x %d\n", box_size_j, box_size_i);
		printf("N is %d\n", N);
		printf("Eps is %lf\n", maxeps);
		printf("Proc is %d\n", proc_size);
		//printf( "Dimentions %d x %d\n", N/box_size_i, N/box_size_j );
	}

	int dims[2], periods[2], reorder;
	dims[0] = N / box_size_i;
	dims[1] = N / box_size_j;
	periods[0] = 0;
	periods[1] = 0;
	reorder = 1;

	int err = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &topology);
    if (topology == MPI_COMM_NULL)
	{
		printf("Error while creating Cart\n");
		exit(1);
	}
	init();

    
	int it = 0;
	int final_it = 0;
    while(1)
	{
        it++;
		eps = 0.;
        relax(it);
//        	printf( "ALL: it=%4i eps=%f\n", it, eps);
        if (eps < maxeps || it > itmax) 
		{
			final_it = it;
			break;
		}
    }
	if (proc_rank == 0)
		printf("Iterations: %d\n", final_it);
		printf("Resulting eps: %lf\n", eps);

	//MPI_Gather(U, int sendcount, MPI_DOUBLE,
	 //              void *recvbuf, (N+2)*(N+2), MPI_DOUBLE, 0, topology);
    //verify();
}


int main(int argc, char ** argv)
{ 
	if (argc > 1)
	{
		sscanf(argv[1], "%d", &N);
	}
	if (N <= 0)
	{
		printf("Error: grid size is too small\n");
		return 3;
	}
	if (argc > 2)
	{
		sscanf(argv[2], "%lf", &maxeps);
	}
	if (maxeps <= 0)
	{
		printf("Error: epsilon is too small\n");
		return 4;
	}
	
	int err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS)
	{
		fprintf(stdout, "Error: MPI_Init. Aborting...\n");
		return 1;
	}
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	
    double start, fin;
	if (proc_rank == 0)
		wtime(&start);

	work();

	if (proc_rank == 0)
    {
		wtime(&fin);
		printf("Time in seconds=%gs\n", fin - start);
		if (argc > 3)
		{
			//output(argv[3]);
		}
		else 
		{
			//output("grid.txt");
		}
	}

	err = MPI_Finalize();
	if (err)
	if (err != MPI_SUCCESS)
	{
		fprintf(stdout, "Error: MPI_Finalize. Aborting...\n");
		return 2;
	}
	int i;
	for (i = 0; i < box_size_i + 2; ++i) 
	{
		free(U[i]);
	}
	free(U);
    return 0;
}
