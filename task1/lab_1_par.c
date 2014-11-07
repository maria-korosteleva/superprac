#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N 256

double maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double A [N][N][N];
void relax();
void init();
void verify();
void wtime();

int main(int an, char **as)
{ 
    int it;
    double time0, time1;
    init();
    /* time0=omp_get_wtime (); */
    wtime(&time0);
    
	for(it=1; it<=itmax; it++)
    {
        eps = 0.;
		relax();
        printf( "it=%4i eps=%f\n", it,eps);
		if (eps < maxeps) break; 
    }
    
	wtime(&time1);
    /* time1=omp_get_wtime (); */
    printf("Time in seconds=%gs\t",time1-time0);
    verify();
    return 0;
}

void init()
{
	for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
            for(k=0; k<=N-1; k++)
            { 
                if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                    A[i][j][k]= 0.;
                    else A[i][j][k]= ( 4. + i + j + k) ;
            }
}

void relax()
{
#pragma omp parallel for ordered schedule(static, 1) private(i, j, k) shared(A)
    for(i=1; i<=N-2; i++)
	{
#pragma omp ordered
		{
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
			}
		}
	}
#pragma omp parallel for schedule(guided) private(i, j, k) shared(A)
	for(i=1; i<=N-2; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
				A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
			}
#pragma omp parallel for schedule(guided) private(i,j,k) shared(A, eps)
	for(i=1; i<=N-2; i++)
	{
		double eps_loc = .0;
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
				double e;
				e=A[i][j][k];
				A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
				eps_loc=Max(eps_loc, fabs(e-A[i][j][k]));
			}
#pragma omp critical
		{
			eps = Max(eps, eps_loc);
		}
	
	}
}

void verify()
{ 
	double s;
	s=0.;
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			for(k=0; k<=N-1; k++)
			{
				s+=A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
			}

	printf(" S = %f\n",s);
}

void wtime(double *t)
{
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, (void *)0);
	if (sec < 0) sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

