#include<cuda.h>
#include<bits/stdc++.h>
#include<curand.h>
#include<curand_kernel.h>
#define PS 100
#define SUB_PS 4
#define CITIES 15
#define CYCLES 10
using namespace std;

__device__ int *best_sol;
__device__ int best_sol_dis;

__device__ int* mutation(int *tour,curandState *state ){
	int result[CITIES];
	int i,j;
	int crossleft, crossright;
	float randf = curand_uniform(&state[0]);
		
	crossleft = ((int)randf*100)%CITIES;
	randf = curand_uniform(&state[1]);
	crossright = ((int)randf*100)%CITIES;
	if(crossleft > crossright)
	{
		int tmp = crossleft;
		crossleft = crossright;
		crossright = tmp;
	}
	while(crossleft >= crossright)
	{
		randf = curand_uniform(&state[1]);
		crossleft = ((int)randf*100)%CITIES;
		randf = curand_uniform(&state[1]);
		crossright = ((int)randf*100)%CITIES;
	}
	for(int i = 0; i < CITIES; i++)
	{
		result[i] = tour[i];
	}
	for(i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
	{
		result[i]=tour[j];
	}
	for(int i = 0; i < CITIES; i++)
	{
		tour[i] = result[i];
	}
	return tour;
	
}
__device__ int* viability_op(int *tour)
{
	int tempA[CITIES], tempB[CITIES], tempC[CITIES];
	memset(tempA, -1, CITIES*sizeof(int));
	memset(tempB, -1, CITIES*sizeof(int));
	memset(tempC, -1, CITIES*sizeof(int));

	int count[CITIES] = {0};
	for(int i = 0; i < CITIES; i++)
		count[tour[i]]++;

	for(int i = CITIES-1; i >= 0; i--)
	{
		if(count[tour[i]] > 1)
		{
			tempA[i] = tour[i];
			count[tour[i]] = -1;
		}
		if(count[tour[i]] == 1)
		{
			tempC[i] = tour[i];
			count[tour[i]] = -1;
		}
	}	
	for(int i = 0; i < CITIES; i++)
	{
		if(count[i] == 0)
			tempB[i] = i;
	}
	int *result;//[CITIES] = {-1};
	result = (int*)malloc(CITIES*sizeof(int));
	int i = 0;
	while(i < CITIES)
	{
		result[i] = tempA[i];
		if(result[i] == -1) result[i] = tempC[i];
		//	result[i] = tempB[i];
		i++;
	}
	int j = 0;
	i = 0;
	while( i < CITIES)
	{
		if(tempB[i] == -1)
			i++;
		else
		{
			if(result[j] == -1)
			{
				result[j] = tempB[i];
				j++; i++;
			}
			else
			{
				j++;
			}
		}
	}
	for(int i = 0; i < CITIES; i++)
		printf("%d ", result[i]);		
//tour[i] = result[i];
	return result;
}

__device__ int* crossover(int *A,int *B, curandState *state){
	int *C;

	int crossleft, crossright;
	float randf = curand_uniform(&state[0]);
		
	//int ind = ((int)(randf*100))%CITIES;
	crossleft = ((int)randf*100)%CITIES;
	randf = curand_uniform(&state[1]);
	crossright = ((int)randf*100)%CITIES;
	if(crossleft > crossright)
	{
		int tmp = crossleft;
		crossleft = crossright;
		crossright = tmp;
	}
	while(crossleft >= crossright)
	{
		randf = curand_uniform(&state[1]);
		crossleft = ((int)randf*100)%CITIES;
		randf = curand_uniform(&state[1]);
		crossright = ((int)randf*100)%CITIES;
	}
	for(int i=crossleft;i<=crossright;i++)
	A[i]=B[i];
	
        printf("\nAfter crossover \n" );
	for(int i = 0; i < CITIES; i++)
		printf("%d ", A[i]);	
		
        C=viability_op(A);
        printf("\nAfter viability \n");
	for(int i = 0; i < CITIES; i++)
		printf("%d ", C[i]);
		
        A=mutation(C, state);
        printf("\nAfter mutation \n");
	for(int i = 0; i < CITIES; i++)
		printf("%d ", A[i]);	
		
	return A;	
}
__global__ void setup_kernel(curandState *state)
{
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void tlboKernel(int *gpupopulation, int *gpuDistanceMat, int numberOfCities, curandState *state)
{
	__shared__ int subPop[SUB_PS][CITIES];
	__shared__ int fitness[SUB_PS];
	__shared__ int mean;
	__shared__ int sum;
	__shared__ int *block_teacher;
	__shared__ int block_teacher_dis;

	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int j = 0; j < CITIES ; j++)
		subPop[threadIdx.x][j] = gpupopulation[id * CITIES + j];

	if(blockIdx.x == 0 && threadIdx.x == 0)
		best_sol_dis = INT_MAX;
	block_teacher_dis = INT_MAX;
	__syncthreads();
	
	//Calculate fitness
	int dis = 0;
	for(int i = 0; i < CITIES-1 ; i++)
	{
		dis += gpuDistanceMat[subPop[threadIdx.x][i] * CITIES + subPop[threadIdx.x][i+1]];
	}
	fitness[threadIdx.x] = dis;

	//Global Teacher
	int old = atomicMin(&best_sol_dis, fitness[threadIdx.x]);
	if( old != best_sol_dis )
	{
		best_sol = subPop[threadIdx.x];
	}	
	if(threadIdx.x == 0)
		printf("best sol = %d, old = %d\n", best_sol_dis, old);
	
	//Subpopulation Teacher
	old = atomicMin(&block_teacher_dis, fitness[threadIdx.x]);
	if( old != block_teacher_dis )
	{
		block_teacher = subPop[threadIdx.x];
	}
	if(threadIdx.x == 0)
		printf("Block = %d : Block Teacher = %d\n",blockIdx.x, block_teacher_dis);
	
	for(int i = 0; i < CYCLES; i++)
	{
		/*TEACHER PHASE*/
		//1. Calculate Mean
		sum = 0;
		__syncthreads();
		atomicAdd((int*)&sum, fitness[threadIdx.x]);
		if(threadIdx.x == 0)
			mean = sum/SUB_PS;
		__syncthreads();
	}
	printf("mean = %d\n", mean);
}


void createPopulation(int *population)
{
	for(int j = 0; j < PS; j++)
	{
		int count[CITIES];
		memset(count, 0, CITIES*sizeof(int));
		for(int i = 0; i < CITIES; i++)
		{	
			int ind = rand()%CITIES;
			while(count[ind])
			{
				ind = rand()%CITIES;
			}
			population[j*CITIES+i] = ind;
			count[ind] = 1; 
		}
	
	}
	
}
int main()
{
	srand(time(NULL));
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	
	int numberOfCities;
	scanf("%d", &numberOfCities);
	int *distanceMat ;//= (int*)malloc(numberOfCities*numberOfCities*sizeof(int));
	cudaHostAlloc(&distanceMat, numberOfCities*numberOfCities*sizeof(int), cudaHostAllocMapped);
	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
			scanf("%d", &distanceMat[i*numberOfCities+j]);
	}
	//int *gpuDistanceMat;
	//cudaMalloc(&gpuDistanceMat, numberOfCities*numberOfCities*sizeof(int));
	//cudaMemcpy(gpuDistanceMat, distanceMat, numberOfCities*numberOfCities*sizeof(int), cudaMemcpyHostToDevice);
	
	int noOfBlocks = ceil((float)PS/SUB_PS);
	int *population; //= (int*)malloc(PS*CITIES*sizeof(int));
	cudaHostAlloc(&population, PS*CITIES*sizeof(int), cudaHostAllocMapped);
	createPopulation(population);

	//int *gpupopulation;
	//cudaMalloc(&gpupopulation, PS*CITIES*sizeof(int));
	//cudaMemcpy(gpupopulation, population, PS*CITIES*sizeof(int), cudaMemcpyHostToDevice);
	setup_kernel<<<noOfBlocks,SUB_PS>>>(d_state);
	tlboKernel<<<noOfBlocks, SUB_PS>>>(population, distanceMat, numberOfCities, d_state);
		
	cudaDeviceSynchronize();

	return 0;
}
