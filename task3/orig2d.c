#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N 256

double maxeps = 0.1e-7;
int itmax = 100;
int proc_size;
int proc_rank;
double eps;
double U[N][N], U_dop[N][N];
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


void relax()
{
    for(i=1; i<=N-2; i++)
        for(j=1; j<=N-2; j++)
        {
            U[i][j] = (U[i-1][j] + U[i+1][j])/2.;
        }
    for(i=1; i<=N-2; i++)
        for(j=1; j<=N-2; j++)
        {
            U[i][j] = (U[i][j-1] + U[i][j+1])/2.;
			double e;
			e = A[i][j];
			eps = Max(eps, fabs(e - U[i][j]);
        }
}

void verify()
{ 
	double s;
	s = 0.;
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			{
				s = s + U[i][j] * (i + 1) * (j + 1) / (N * N);
			}
	printf(" S = %f\n",s);
}


void init()
{
    proc_size = MPI_Group_size(MPI_COMM_WORLD, &proc_size);
    proc_rank = MPI_Group_rank(MPI_COMM_WORLD, &proc_rank);
	
	int i, j;
    for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
        { 
            if(i == 0 || i == N - 1 || j == 0 || j == N - 1)
                U[i][j] = 0.;
            else 
				U[i][j] = ( 4. + i + j) ;
        }
}


int main(int an, char **as)
{ 
	int err = MPI_Init(&an, &as);
	if (err)
	{
		printf("Error: MPI_Init. Aborting...");
		return 1;
	}

	wtime(&start);
	int it;
    double start, fin;
    init();
    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax();
		
        printf( "it=%4i eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }
    wtime(&fin);
    printf("Time in seconds=%gs\t", fin - start);
    verify();
	
	err = MPI_Finalize();
	if (err)
	{
		printf("Error: MPI_Finalize. Aborting...");
		return 2;
	}
    return 0;
}
