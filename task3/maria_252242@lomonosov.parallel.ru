#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define Min(a,b) ((a)<(b)?(a):(b))

double maxeps = 0.01;
int N = 512;
int threads_num = 4;
// spectral radius
int itmax = 10000;
int proc_size;
int proc_rank;
int box_size_i;
int box_size_j;
int box_size_k;
MPI_Comm topology;
double eps;
double*** U = NULL;
void relax();
void init();
void verify();
void wtime();

void verify()
{ 
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	
	double s, allsum = 0.;
	s = 0.;
	int i, j, k;
	for(i = 1; i < box_size_i+1; i++)
		for(j = 1; j < box_size_j+1; j++)
			for(k = 1; k < box_size_k+1; k++)
			{
				int ti = box_size_i*coords[0] + i;
				int tj = box_size_j*coords[1] + j;
				int tk = box_size_k*coords[2] + k;
				s = s + U[i][j][k] * (ti+1) * (tj+1)*(tk+1) / (N * N * N);
			}
	//printf("Proc %d,  Sum = %f\n", proc_rank, s);
	MPI_Allreduce(&s, &allsum, 1, MPI_DOUBLE, MPI_SUM, topology);
	if (proc_rank == 0)
	{
		printf(" S = %f\n", allsum);
	}
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
	double p = 1. - (M_PI / ((N - 1)))*(M_PI / ((N - 1))); 
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
	if (proc_rank == 0)
	{
		//printf("Omega is %lf with n == %d, p is %lf\n", loc_prev_om, n, p);
	}
	return loc_prev_om;
}

void init()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);

	//printf( "Hello world from process %d of %d\n", proc_rank, proc_size );
	//printf( "Hello world from coords %d x %d x %d\n", coords[0], coords[1], coords[2]);

	int i, j, k;
	U = calloc(box_size_i + 2, sizeof(U[0]));

	#pragma omp parallel for schedule(guided) private(i,j) shared(U)
	for (i = 0; i < box_size_i + 2; i++) 
	{
		U[i] = calloc(box_size_j + 2, sizeof(U[i][0]));
		for (j = 0; j < box_size_j + 2; j++)
		{
			U[i][j] = calloc(box_size_k + 2, sizeof(U[i][j][0]));
		}
	}
	k = box_size_k+1;
	int tk = box_size_k*coords[2] + k;
	if (tk == N + 1)
	{
		for(i=0; i< box_size_i+2; i++)
			for(j=0; j< box_size_j+2; j++)
			{
				int ti = box_size_i*coords[0] + i;
				int tj = box_size_j*coords[1] + j;
				
				double x = ti * (1.0 / (N + 1));
				double y = tj * (1.0 / (N + 1));
				
				U[i][j][box_size_k+1] = (1. - fabs(2.*x-1.)) * (1. - fabs(2.*y-1.));
	//			printf("F(%d, %d, %d) = %lf\n", ti, tj, box_size_k+1, U[i][j][box_size_k+1]);
			}
	}
	//verify();
}

void recv_top()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_top, coords_top[3];
	coords_top[0] = coords[0] - 1;
	coords_top[1] = coords[1];
	coords_top[2] = coords[2];
	
	MPI_Cart_rank(topology, coords_top, &r_top);
	double* buf = (double*) malloc (((box_size_j+2)*(box_size_k +2)) * sizeof(double));
	MPI_Recv(buf, (box_size_j+2)*(box_size_k +2), MPI_DOUBLE, r_top, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_j+2)*(box_size_k +2))
		printf("Error while recieving from top\n");
	for (j = 0; j < box_size_j+2; j++)
		for (k = 0; k < box_size_k+2; k++)
		{
			U[0][j][k] = buf[j*(box_size_k+2) + k];
		}
	free(buf);
}

void recv_left()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_left, coords_left[3];
	coords_left[0] = coords[0];
	coords_left[1] = coords[1] - 1;
	coords_left[2] = coords[2];
	MPI_Cart_rank(topology, coords_left, &r_left);
	
	double* buf = (double*) malloc (((box_size_i+2)*(box_size_k+2)) * sizeof(double));
	
	MPI_Recv(buf, (box_size_i+2)*(box_size_k+2), MPI_DOUBLE, r_left, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_i+2)*(box_size_k+2))
		printf("Error while recieving from left\n");
	
	for (j = 0; j < box_size_i+2; j++)
		for (k = 0; k < box_size_k+2; k++)
		{
			U[j][0][k] = buf[j*(box_size_k+2) + k];
		}
	free(buf);
}

void recv_back()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_b, coords_back[3];
	coords_back[0] = coords[0];
	coords_back[1] = coords[1];
	coords_back[2] = coords[2] - 1;
	MPI_Cart_rank(topology, coords_back, &r_b);
	
	double* buf = (double*) malloc (((box_size_i+2)*(box_size_j+2)) * sizeof(double));
	
	MPI_Recv(buf, (box_size_i+2)*(box_size_j+2), MPI_DOUBLE, r_b, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_i+2)*(box_size_j+2))
		printf("Error while recieving from left\n");
	
	for (i = 0; i < box_size_i+2; i++)
		for (j = 0; j < box_size_j+2; j++)
		{
			U[i][j][0] = buf[i*(box_size_j+2) + j];
		}
	free(buf);
}

void recv_bottom()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_b, coords_b[3];
	coords_b[0] = coords[0] + 1;
	coords_b[1] = coords[1];
	coords_b[2] = coords[2];
	MPI_Cart_rank(topology, coords_b, &r_b);
	double* buf = (double*) malloc (((box_size_j+2)*(box_size_k+2)) * sizeof(double));
	MPI_Recv(buf, (box_size_j+2)*(box_size_k+2), MPI_DOUBLE, r_b, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_j+2)*(box_size_k+2))
		printf("Error while recieving from left\n");
	for (j = 0 ; j < box_size_j + 2; j++)
		for (k = 0 ; k < box_size_k + 2; k++)
		{
			 U[box_size_i+1][j][k] = buf[ j*(box_size_k + 2) + k ];
		}
	free(buf);
}

void recv_right()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_right, coords_r[3];
	coords_r[0] = coords[0];
	coords_r[1] = coords[1] + 1;
	coords_r[2] = coords[2];
	MPI_Cart_rank(topology, coords_r, &r_right);
	double* buf = (double*) malloc ((box_size_i+2)*(box_size_k+2) * sizeof(double));
	MPI_Recv(buf, (box_size_i+2)*(box_size_k+2), MPI_DOUBLE, r_right, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_i+2)*(box_size_k+2))
		printf("Error while recieving from left\n");
	for (i = 0 ; i < box_size_i + 2; i++)
		for (k = 0 ; k < box_size_k + 2; k++)
		{
			 U[i][box_size_j+1][k] = buf[i *(box_size_k + 2) + k];
		}
	free(buf);
}

void recv_front()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_f, coords_f[3];
	coords_f[0] = coords[0];
	coords_f[1] = coords[1];
	coords_f[2] = coords[2] + 1;
	//printf("Proc %d, recv front, coords %d x %d x %d\n", proc_rank, coords_f[0], coords_f[1], coords_f[2]);
	MPI_Cart_rank(topology, coords_f, &r_f);
	double* buf = (double*) malloc ((box_size_i+2)*(box_size_j+2) * sizeof(double));
	MPI_Recv(buf, (box_size_i+2)*(box_size_j+2), MPI_DOUBLE, r_f, 0,
				 topology, &status);
	MPI_Get_count (&status, MPI_DOUBLE, &count);
	if (count != (box_size_i+2)*(box_size_j+2))
		printf("Error while recieving from left\n");
	for (i = 0 ; i < box_size_i + 2; i++)
		for (j = 0 ; j < box_size_j + 2; j++)
		{
			 U[i][j][box_size_k+1] = buf[i *(box_size_j + 2) + j];
		}
	free(buf);
}

void send_top()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_top, coords_top[3];
	coords_top[0] = coords[0] - 1;
	coords_top[1] = coords[1];
	coords_top[2] = coords[2];
	
	MPI_Cart_rank(topology, coords_top, &r_top);
	double* buf = (double*) malloc (((box_size_j+2)*(box_size_k +2)) * sizeof(double));
	for (j = 0; j < box_size_j+2; j++)
		for (k = 0; k < box_size_k+2; k++)
		{
			buf[j*(box_size_k+2) + k] = U[1][j][k];
		}
	MPI_Send(buf, (box_size_j+2)*(box_size_k +2), MPI_DOUBLE, r_top, 0,
				 topology);
	free(buf);
}

void send_left()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_left, coords_left[3];
	coords_left[0] = coords[0];
	coords_left[1] = coords[1] - 1;
	coords_left[2] = coords[2];
	MPI_Cart_rank(topology, coords_left, &r_left);
	
	double* buf = (double*) malloc (((box_size_i+2)*(box_size_k+2)) * sizeof(double));
	
	for (j = 0; j < box_size_i+2; j++)
		for (k = 0; k < box_size_k+2; k++)
		{
			buf[j*(box_size_k+2) + k] = U[j][1][k];
		}
	MPI_Send(buf, (box_size_i+2)*(box_size_k+2), MPI_DOUBLE, r_left, 0,
				 topology);
	free(buf);
}

void send_back()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_b, coords_back[3];
	coords_back[0] = coords[0];
	coords_back[1] = coords[1];
	coords_back[2] = coords[2] - 1;
	MPI_Cart_rank(topology, coords_back, &r_b);
	
	double* buf = (double*) malloc (((box_size_i+2)*(box_size_j+2)) * sizeof(double));
	
	for (i = 0; i < box_size_i+2; i++)
		for (j = 0; j < box_size_j+2; j++)
		{
			buf[i*(box_size_j+2) + j] = U[i][j][1];
		}
	MPI_Send(buf, (box_size_i+2)*(box_size_j+2), MPI_DOUBLE, r_b, 0,
				 topology);
	free(buf);
}

void send_bottom()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_b, coords_b[3];
	coords_b[0] = coords[0] + 1;
	coords_b[1] = coords[1];
	coords_b[2] = coords[2];
	//printf("Proc %d, send bottom, coords %d x %d x %d\n", proc_rank, coords_b[0], coords_b[1], coords_b[2]);
	MPI_Cart_rank(topology, coords_b, &r_b);
	double* buf = (double*) malloc (((box_size_j+2)*(box_size_k+2)) * sizeof(double));
	for (j = 0 ; j < box_size_j + 2; j++)
		for (k = 0 ; k < box_size_k + 2; k++)
		{
			buf[ j*(box_size_k + 2) + k ] = U[box_size_i][j][k];
		}
	MPI_Send(buf, (box_size_j+2)*(box_size_k+2), MPI_DOUBLE, r_b, 0,
				 topology);
	free(buf);
}

void send_right()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_right, coords_r[3];
	coords_r[0] = coords[0];
	coords_r[1] = coords[1] + 1;
	coords_r[2] = coords[2];
	//printf("Proc %d, send right, coords %d x %d x %d\n", proc_rank, coords_r[0], coords_r[1], coords_r[2]);
	MPI_Cart_rank(topology, coords_r, &r_right);
	double* buf = (double*) malloc ((box_size_i+2)*(box_size_k+2) * sizeof(double));
	for (i = 0 ; i < box_size_i + 2; i++)
		for (k = 0 ; k < box_size_k + 2; k++)
		{
			buf[i *(box_size_k + 2) + k] = U[i][box_size_j][k];
		}
	MPI_Send(buf, (box_size_i+2)*(box_size_k+2), MPI_DOUBLE, r_right, 0,
				 topology);
	free(buf);
}

void send_front()
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	int r_f, coords_f[3];
	coords_f[0] = coords[0];
	coords_f[1] = coords[1];
	coords_f[2] = coords[2] + 1;
	//printf("Proc %d, send front, coords %d x %d x %d\n", proc_rank, coords_f[0], coords_f[1], coords_f[2]);
	MPI_Cart_rank(topology, coords_f, &r_f);
	double* buf = (double*) malloc ((box_size_i+2)*(box_size_j+2) * sizeof(double));
	for (i = 0 ; i < box_size_i + 2; i++)
		for (j = 0 ; j < box_size_j + 2; j++)
		{
			buf[i *(box_size_j + 2) + j] = U[i][j][box_size_k];
		}
	MPI_Send(buf, (box_size_i+2)*(box_size_j+2), MPI_DOUBLE, r_f, 0,
				 topology);
	free(buf);
}

// parameter: iteration number
void relax(int iter)
{
	int coords[3], r;
	MPI_Comm_rank(topology, &r);
	MPI_Cart_coords (topology, r, 3, coords);
	MPI_Status status;
	
	int i, j, k;
	int count;
	// Recieve current iteration data from top, left and back processes
	
	if (coords[0] - 1 >= 0) // top
		recv_top();
	if (coords[1] - 1 >= 0) // left
		recv_left();
	if (coords[2] - 1 >= 0) // back
		recv_back();

	// ************************OMP SECTION***********************************
	// diagonals of 2D side
	// each diagonal can run in parallel, max threads_num blocks in parallel
	int diag, block;
	double loc_eps = 0.;
	double om = omega(iter);
	for(diag = 1; diag <= (2*threads_num-1); diag++)
	{
		int diag_size = diag <= threads_num ? diag : 2*threads_num - diag;
		//printf("Diagonal #%d, diag_size %d\n", diag, diag_size);
		//printf("Block size %lf X %lf\n", (double)box_size_i/(double)threads_num , (double)box_size_j/(double)threads_num);
		omp_set_num_threads(diag_size);
		int bl_coords[2];
#pragma omp parallel for schedule(static, 1) default(none) private(block, bl_coords, i, j, k) shared(U, loc_eps, om,  box_size_i, box_size_j, box_size_k, diag, threads_num, diag_size)  
		for(block = 0; block < diag_size; block++)
		{
			//block = omp_get_thread_num();
			// count the inside of the block on the current diagonal
			bl_coords[0] = Min((diag-1), threads_num-1)- block;
			bl_coords[1] = (diag-1) - bl_coords[0];
			//printf("Thr %d, Block_coords %d X %d\n", block,  bl_coords[0], bl_coords[1]);
			int ti, tj;
			double loc_loc_eps = 0.;
			for(ti=0; ti < box_size_i / threads_num; ti++)
				for(tj=0; tj< box_size_j / threads_num; tj++)
				{
					i = bl_coords[0]*(box_size_i/threads_num) + ti + 1;
					j = bl_coords[1]*(box_size_j/threads_num) + tj + 1;
					for(k=1; k< box_size_k+1; k++)
					{
						double U_dop;
						U_dop = (U[i-1][j][k] + U[i+1][j][k] + U[i][j-1][k] + U[i][j+1][k] + U[i][j][k-1] + U[i][j][k+1])/6.;
						double e;
						e = U[i][j][k];
						U[i][j][k] = om * U_dop + (1. - om) * U[i][j][k];
						if (fabs(e - U[i][j][k]) > loc_loc_eps)
							loc_loc_eps = fabs(e - U[i][j][k]);
					}
				}
			#pragma omp critical	
			{
				loc_eps = Max(loc_eps, loc_loc_eps);
			}
		}
		#pragma omp barrier
	}
	// ***********************************************************************
	
	// Send new data to bottom, right and front processes
	if (coords[0] + 1 <= (N / box_size_i) - 1) // bottom
		send_bottom();
	if (coords[1] + 1 <= (N / box_size_j) - 1) // right
		send_right();
	if (coords[2] + 1 <= (N / box_size_k) - 1) // front
		send_front();

	// Send data to top, left and back processes for future use 
	if (coords[0] - 1 >= 0) // top
		send_top();
	if (coords[1] - 1 >= 0) // left
		send_left();
	if (coords[2] - 1 >= 0) // back
		send_back();
	
	// Recieve data from right, bottom and front processes for future use
	if (coords[0] + 1 <= (N / box_size_i) - 1) // bottom
		recv_bottom();
	if (coords[1] + 1 <= (N / box_size_j) - 1) // right
		recv_right();
	if (coords[2] + 1 <= (N / box_size_k) - 1) // front
		recv_front();
	
	//	eps from all procs
	MPI_Allreduce(&loc_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, topology);

}

void work()
{
	float float_size = pow(proc_size, 1./3.);
	float float_size_double = pow(2 * proc_size, 1./3.);
	float float_size_ddouble = pow(4 * proc_size, 1./3.);
	if (float_size - floor(float_size) == 0)
	{
		box_size_i = N / (int) float_size;
		box_size_j = box_size_i;
		box_size_k = box_size_i;
	}
	else if (float_size_double - floor(float_size_double) == 0)
	{
		box_size_i = N / (int) float_size_double;
		box_size_j = box_size_i;
		box_size_k = box_size_i * 2;
	}
	else
	{
		box_size_i = N / (int) float_size_ddouble;
		box_size_j = box_size_i * 2;
		box_size_k = box_size_i * 2;
	}
	if (proc_rank == 0)
	{
		//printf("Box size is %d x %d x %d\n", box_size_i, box_size_j, box_size_k);
		//printf("Dimentions %d x %d x %d\n", N/box_size_i, N/box_size_j,N/box_size_k);
		printf("N is %d\n", N);
		printf("Eps is %lf\n", maxeps);
		printf("Proc is %d\n", proc_size);
	#pragma omp parallel
		{
			printf("Treads are %d, %d\n", omp_get_num_threads(), threads_num);
		}
	}

	int dims[3], periods[3], reorder;
	dims[0] = N / box_size_i;
	dims[1] = N / box_size_j;
	dims[2] = N / box_size_k;
	periods[0] = 0;
	periods[1] = 0;
	periods[2] = 0;
	reorder = 1;

	int err = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &topology);
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
		//if (proc_rank == 0)
		//printf( "ALL: it=%4i eps=%f\n", it, eps);
		if (eps < maxeps || it > itmax) 
		{
			final_it = it;
			break;
		}
	}
	if (proc_rank == 0)
	{
		printf("Resulting eps: %lf\n", eps);
		printf("Iterations: %d\n", final_it);
	}
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
	if (argc > 3)
	{
		sscanf(argv[3], "%d", &threads_num);
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
	// ************************OMP SECTION***********************************
	omp_set_num_threads(threads_num);
	// **********************************************************************
	
	double start, fin;
	if (proc_rank == 0)
		wtime(&start);

	work();

	if (proc_rank == 0)
	{
		wtime(&fin);
		printf("Time in seconds=%gs\n", fin - start);
	}

	verify();

	err = MPI_Finalize();
	if (err)
	if (err != MPI_SUCCESS)
	{
		fprintf(stdout, "Error: MPI_Finalize. Aborting...\n");
		return 2;
	}
	int i, j;
	for (i = 0; i < box_size_i + 2; i++) 
	{
		for (j = 0; j < box_size_j + 2; j++) 
		{
			free(U[i][j]);
		}
		free(U[i]);
	}
	free(U);
	return 0;
}
