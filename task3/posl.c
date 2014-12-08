#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
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

double loc_prev_om = 0;
double omega(int n)
{
    double p = 1. - (PI / ((N - 1)))*(PI / ((N - 1))); 
	double om;
	if (n == 0)
		loc_prev_om = 0;
	else if (n == 1)
	{
		loc_prev_om = 1./(1. - p*p/2); 
	}
	else
	{
		loc_prev_om = 1./(1. - loc_prev_om*p*p/4); 
	}
	//printf("Omega is %lf with n == %d, p is %lf\n", loc_prev_om, n, p);
	
	return loc_prev_om;
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
	int i, j;
	U = calloc(N + 2, sizeof(U[0]));
	for (i = 0; i < N + 2; i++) 
	{
		U[i] = calloc(N + 2, sizeof(U[i][0]));
	}
	for (i = 0; i < N + 2; ++i) 
	{
		double x = 1.0 / (N + 1) * i;
		double y = x;
		U[i][0] = sin((PI * x) / 2);
		U[i][N + 1] = sin((PI * x) / 2) * exp(-PI / 2);
		U[N + 1][i] = exp(-(PI * y) / 2);
	}
	/*for(i=0; i< N+2; i++)
        for(j=0; j< N+2; j++)
        { 
			int ti = i;
			int tj = j;

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
		*/
}

// parameter: iteration number
void relax(int iter)
{
	double loc_eps = 0.;
    int i, j;
	double om = omega(iter);
	
	// Me
	for(i=1; i < N+1; i++)
        for(j=1; j< N+1; j++)
        {
            double U_dop = (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1])/4.;
			double e;
			e = U[i][j];
            U[i][j] = om * U_dop + (1. - om) * U[i][j];
			loc_eps = loc_eps > fabs(e - U[i][j]) ? loc_eps : fabs(e - U[i][i]);
        }
	eps = loc_eps;
	//MPI_Barrier(topology);
}

void work()
{
		printf("Box size is %d x %d\n", N, N);
		printf( "Dimentions %d x %d\n", N/N, N/N );
		printf("N is %d\n", N);
		printf("Eps is %lf\n", maxeps);
		printf("Proc is %d\n", proc_size);
	init();
    
	int it = 0;
	int final_it = 0;
    while(1)
	{
        it++;
		eps = 0.;
        relax(it);
		printf( "ALL: it=%4i eps=%f\n", it, eps);
        if (eps < maxeps || it > itmax) 
		{
			final_it = it;
			break;
		}
    }
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

	int i;
	for (i = 0; i < N + 2; ++i) 
	{
		free(U[i]);
	}
	free(U);
    return 0;
}
