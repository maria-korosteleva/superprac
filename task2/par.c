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
int itmax = 10000;
int proc_size;
int proc_rank;
double eps;
double** U = NULL;
void relax();
void init();
void verify();
void wtime();

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

// parameter: iteration number
void relax(int iter)
{
    int i, j;
	double om = omega(iter);
	for(i=1; i < N+1; i++)
        for(j=1; j< N+1; j++)
        {
            double U_dop = (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1])/4.;
			double e;
			e = U[i][j];
            U[i][j] = om * U_dop + (1. - om) * U[i][j];
			eps = eps > fabs(e - U[i][j]) ? eps : fabs(e - U[i][i]);
        }
}

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


void init()
{
	int i, j;
	U = calloc(N + 2, sizeof(U[0]));
	for (i = 0; i < N + 2; i++) 
	{
		U[i] = calloc(N + 2, sizeof(U[i][0]));
	}
    for(i=0; i< N+2; i++)
        for(j=0; j< N+2; j++)
        { 
			double x = i * (1.0 / (N + 1));
			double y = j * (1.0 / (N + 1));
			
			if(j == 0) 
            {    
				U[i][j] = sin(PI * x / 2);
            }
			else if (j == N + 1)
			{
				U[i][j] = sin(PI * x / 2) * exp(-PI/2);
			}
			else if (i == N + 1)
			{
				U[i][j] = exp(- PI * y / 2);
			}
			else 
				U[i][j] = 0 ;
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
	flose(grid);
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
		fprintf(stderr, "Error: MPI_Init. Aborting...\n");
		return 1;
	}
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	
// we define process with rank 0 as the main one
    double start, fin;
	if (proc_rank == 0)
		wtime(&start);
    init();

	printf( "Hello world from process %d of %d\n", proc_rank, proc_size );
    
	int it;
	for(it = 1; it < itmax; it++)
    {
        eps = 0.;
        relax(it);
		
        printf( "it=%4i eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }
	if (proc_rank == 0)
    {
		wtime(&fin);
		printf("Time in seconds=%gs\t", fin - start);
		if (argc > 3)
		{
			output(argv[2]);
		}
		else 
		{
			output("grid.txt");
		}
	}
    verify();

	
	err = MPI_Finalize();
	if (err)
	if (err != MPI_SUCCESS)
	{
		fprintf(stderr, "Error: MPI_Finalize. Aborting...\n");
		return 2;
	}
	int i;
	for (i = 0; i < N + 2; ++i) 
	{
		free(U[i]);
	}
	free(U);
    return 0;
}
