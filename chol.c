/* Cholesky decomposition.
 * Compile as follows:
 * 						gcc -fopenmp -o chol chol.c chol_gold.c -lpthread -lm -std=c99
 */

// includes, system
#define _XOPEN_SOURCE 600 //For barriers
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "chol.h"
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

Matrix allocate_matrix(int num_rows, int num_columns, int init);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern int chol_gold(const Matrix, Matrix);
extern int check_chol(const Matrix, const Matrix);
void chol_using_pthreads(const Matrix, Matrix);
void chol_using_openmp(const Matrix, Matrix);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Matrices for the program
	Matrix A; // The N x N input matrix
	Matrix reference; // The upper triangular matrix computed by the CPU
	Matrix U_pthreads; // The upper triangular matrix computed by the pthread implementation
	Matrix U_openmp; // The upper triangular matrix computed by the openmp implementation 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));

	// Create the positive definite matrix. May require a few tries if we are unlucky
	int success = 0;
	while(!success){
		A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		if(A.elements != NULL)
				  success = 1;
	}
	// print_matrix(A);
	// getchar();


	reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
	U_pthreads =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the pthread result
	U_openmp =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the openmp result

	struct timeval start;
	struct timeval stop;
	
	// Start the timer here. 	
	gettimeofday(&start, NULL);

	// compute the Cholesky decomposition on the CPU; single threaded version	
	printf("Performing Cholesky decomposition on the CPU using the single-threaded version. \n");
	int status = chol_gold(A, reference);
	if(status == 0){
			  printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
			  exit(0);
	}
	
	// Stop timer here and determine the elapsed time. 	
	gettimeofday(&stop, NULL);
	float cpu_time = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("CPU Overall execution time = %fs. \n", cpu_time);
	
	printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, reference) == 0){
		printf("Error performing Cholesky decomposition on the CPU. Try again. Exiting. \n");
		exit(0);
	}
	printf("Cholesky decomposition on the CPU was successful. \n");
	

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using pthreads. The resulting upper triangular matrix should be returned in 
	 U_pthreads */
	printf("Performing Cholesky decomposition on the CPU using the PTHREAD version. \n");
	/* Start the timer here. */
	gettimeofday(&start, NULL);
	chol_using_pthreads(A, U_pthreads);
	/* Stop timer here and determine the elapsed time. */	
	gettimeofday(&stop, NULL);
	float pthread_time = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("PTHREADS Overall execution time = %fs. \n", pthread_time);
	
	// Check if the pthread and openmp results are equivalent to the expected solution
	printf("Double checking for correctness by recovering the original matrix. \n");
	
	if(check_chol(A, U_pthreads) == 0) 
			  printf("Error performing Cholesky decomposition using pthreads. \n");
	else
			  printf("Cholesky decomposition using pthreads was successful. \n");
	

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using openmp. The resulting upper traingular matrix should be returned in U_openmp */
	printf("Performing Cholesky decomposition on the CPU using the OPENMP version. \n");
	/* Start the timer here. */
	gettimeofday(&start, NULL);
	chol_using_openmp(A, U_openmp);
	/* Stop timer here and determine the elapsed time. */	
	gettimeofday(&stop, NULL);
	float omp_time=(float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("OPENMP Overall execution time = %fs. \n", omp_time);

	
	printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, U_openmp) == 0) 
			  printf("Error performing Cholesky decomposition using openmp. \n");
	else	
			  printf("Cholesky decomposition using openmp was successful. \n");


	// Free host matrices
	//free(A.elements); 	
	//free(U_pthreads.elements);	
	//free(U_openmp.elements);
	//free(reference.elements); 
	return 1;
}

/* Write code to perform Cholesky decopmposition using openmp. */
void chol_using_openmp(const Matrix A, Matrix U)
{
	unsigned int i, j, k; 
	unsigned int size = A.num_rows * A.num_columns;

	// Copy the contents of the A matrix into the working matrix U
	//Parallelize this too?
	//#pragma omp parallel for 
	for (i = 0; i < size; i ++)
		U.elements[i] = A.elements[i];

	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++){
			// Take the square root of the diagonal element
			U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
			if(U.elements[k * U.num_rows + k] <= 0){
					 printf("Cholesky decomposition failed. \n");
			}

			// Division step
			//Parallelize this - seems like declaring private iterator 
			//values slows it down....?
			#pragma omp parallel for
			for(j = (k + 1); j < U.num_rows; j++)
					 U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step

			// Elimination step
			float * elems = U.elements;
			omp_set_num_threads(NUM_OMPTHREADS); // Set the number of threads
			
			//Parallelize this
			#pragma omp parallel for private(i,j)
			for(i = (k + 1); i < U.num_rows; i++)
			{
				for(j = i; j < U.num_rows; j++)
				{
						U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];
				}
			}
	}

	// As the final step, zero out the lower triangular portion of U
	for(i = 0; i < U.num_rows; i++)
			  for(j = 0; j < i; j++)
						 U.elements[i * U.num_rows + j] = 0.0;
}

//Range splitter helper function
void range_splitter(int size, int num_threads, int * items_per_thread, int * items_last_thread)
{
	//Divide up total size by number of threads
	//How many are left over?
	int elems_left_over = size%num_threads;
	int elements_per_thread = size/num_threads;
	int last_thread_elements = elements_per_thread;
	if(elems_left_over !=0)
	{ 
		int not_total = elements_per_thread*(num_threads-1);
		last_thread_elements = size - not_total;
		
		if(last_thread_elements<0)
		{
			//Tooo much now
			elements_per_thread-=2;
			not_total = elements_per_thread*(num_threads-1);
			last_thread_elements = size - not_total;
		}
	}
	
	//Double check because math is hard
	if( (((num_threads-1)*elements_per_thread) + last_thread_elements) != size || (last_thread_elements<0))
	{
		printf("AH! MATH! threads:%d elementsperthread:%d lastthreadelm:%d size:%d leftover:%d\n", num_threads,elements_per_thread,last_thread_elements,size,elems_left_over);
		exit(-1);
	}
	*items_per_thread = elements_per_thread;
	*items_last_thread = last_thread_elements;
}

typedef struct chol_pthread_args
{
	//Matricies
	Matrix A;
	Matrix U;
	//Copy work
	int copyi_start;
	int copyi_end;
	//Zero out work
	int zeroi_start;
	int zeroi_end;
	//Each thread does all of j loop as 'written originally'
	//Range is recalculated
	//Barrier to use (pointer, shared among threads)
	pthread_barrier_t * barrier;
	//Thread id
	int id;
}chol_pthread_args;


void range_maker(int items_per_thread,int items_last_thread, int num_threads, int index, int offset, int is_last_thread, int * start, int * end)
{
	if(is_last_thread==1)
	{
		//Last thread
		*start=items_per_thread*index + offset;
		*end = items_per_thread*index + offset + items_last_thread;
	}
	else
	{
		//Regular threads
		*start=items_per_thread*index + offset;
		*end = items_per_thread*(index+1) -1 + offset;
	}	
}

void populate_thread_args(chol_pthread_args * arg_list,Matrix A, Matrix U, pthread_barrier_t * barrier)
{	
	//Matrix size
	unsigned int size = A.num_rows * A.num_columns;
	
	//Copy
	int copyisize = size-0;
	int copyi_items_per_thread, copyi_items_last_thread;
	range_splitter(copyisize, NUM_PTHREADS, &copyi_items_per_thread, &copyi_items_last_thread);
	
	//Zero out
	int zeroisize = U.num_rows - 0;
	int zeroi_items_per_thread, zeroi_items_last_thread;
	range_splitter(zeroisize, NUM_PTHREADS, &zeroi_items_per_thread, &zeroi_items_last_thread);
	
	//Zero offset for both sets of work
	int offset = 0;
	
	//Loop through threads
	int i;
	for(i=0;i< NUM_PTHREADS; i++)
	{
		//Easy ones for all threads
		arg_list[i].A=A;
		arg_list[i].U=U;
		arg_list[i].barrier = barrier;
		arg_list[i].id = i;
		
		if(i == (NUM_PTHREADS-1))
		{
			//Last thread
			range_maker(copyi_items_per_thread,copyi_items_last_thread, NUM_PTHREADS, i, offset,1, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread, NUM_PTHREADS, i, offset,1, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}
		else
		{
			//Regular threads
			range_maker(copyi_items_per_thread,copyi_items_last_thread, NUM_PTHREADS, i, offset,0, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread, NUM_PTHREADS, i, offset,0, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}	
	}
}

void sync_pthreads(pthread_barrier_t * barrier, int thread_id)
{
	// Synchronization point
    int rc = pthread_barrier_wait(barrier);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier.\n");
        exit(-1);
    } 
}


void * chol_pthread(void * arg)
{
	//Get arg as struct
	chol_pthread_args * args = (chol_pthread_args *)arg;
	//Matrices
	Matrix A = args->A;
	Matrix U = args->U;
	//Copy work
	int copyi_start = args->copyi_start;
	int copyi_end = args->copyi_end;
	//Zero out work
	int zeroi_start = args->zeroi_start;
	int zeroi_end = args->zeroi_end;
	//Barrier to sync
	pthread_barrier_t * barrier = args->barrier;
	//Id
	int id = args->id;
	
	//Iterators
	unsigned int i, j, k;
	unsigned int size = A.num_rows * A.num_columns;
	
	//Copy the contents of the A matrix into the working matrix U
	for (i = copyi_start; i <= copyi_end; i ++)
	{
		U.elements[i] = A.elements[i];
	}
	
	//Sync threads!!!
	sync_pthreads(barrier, id);
	
	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++)
	{
		//Only one thread does squre root and division
		if(id==0)
		{
			// Take the square root of the diagonal element
			U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
			if(U.elements[k * U.num_rows + k] <= 0){
					 printf("Cholesky decomposition failed. \n");
					 return 0;
			}
		
			// Division step
			for(j = (k + 1); j < U.num_rows; j++)
			{
				U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step
			}
		}
		
		//Sync threads!!!!!
		sync_pthreads(barrier, id);

		//For this k iteration, split up i
		//Size of i range originally
		int isize = U.num_rows - (k + 1);
		int items_per_thread, items_last_thread;
		range_splitter(isize, NUM_PTHREADS, &items_per_thread, &items_last_thread);
		//Divy up work
		//Elim work
		int elimi_start, elimi_end;
		int offset = (k + 1); //To account for not starting at i=0 each time
		if(id == (NUM_PTHREADS-1))
		{
			//Last thread
			elimi_start=items_per_thread*id + offset;
			elimi_end = items_per_thread*id + offset + items_last_thread;
		}
		else
		{
			//Regular threads
			elimi_start=items_per_thread*id + offset;
			elimi_end = items_per_thread*(id+1) -1 + offset;
		}	

		// Elimination step
		//printf("K Loop. I range %d to %d\n",(k + 1),U.num_rows-1);
		for(i = elimi_start; i <=elimi_end; i++)
		{
			for(j = i; j < U.num_rows; j++)
			{
				U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];
			}
		}
		
		//Sync threads!!!!!
		sync_pthreads(barrier, id);
	}

	//Sync threads!!!!!
	sync_pthreads(barrier, id);
	
	// As the final step, zero out the lower triangular portion of U
	for(i = zeroi_start; i <=zeroi_end; i++)
	{
		for(j = 0; j < i; j++)
		{
			U.elements[i * U.num_rows + j] = 0.0;
		}
	}
	
	//Don't sync, will join outside here
}

/* Write code to perform Cholesky decopmposition using pthreads. */
void chol_using_pthreads(const Matrix A, Matrix U)
{
	//Array of pthreads
	pthread_t threads[NUM_PTHREADS];
	
	//Initialize barrier
	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL, NUM_PTHREADS);
	
	//Array of arguments
	chol_pthread_args args[NUM_PTHREADS];
	
	//Populate thread args
	populate_thread_args(&args[0],A,U,&barrier);
	
	//Launch pthreads
	int i;
	for(i=0; i < NUM_PTHREADS; i++)
	{
		pthread_create(&(threads[i]),NULL,chol_pthread,&(args[i]));
	}
	
	//Join threads
	for(i=0; i < NUM_PTHREADS; i++)
	{
		pthread_join(threads[i],NULL);
	}
}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float *) malloc(size * sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = (float)rand()/(float)RAND_MAX;
	}
    return M;
}	




