#include<cuda.h>
#include<bits/stdc++.h>
#include<curand.h>
#include<curand_kernel.h>
#define PS 100
#define SUB_PS 4
#define CITIES 42
#define CYCLES 10
#define CROSS_GLOBALTEACHER 0
#define CROSS_LOCALTEACHER 1
#define CROSS_MEAN 2
#define CROSS_LOCALMEAN 3

using namespace std;

__device__ volatile int *best_sol;
__device__ volatile int best_sol_dis;
__device__ volatile int var = 100;
__device__ volatile int itr = 100;
__device__ int* mutation(int *tour,curandState *state ){
	int *result;//[CITIES];
	int i,j;
	int crossleft, crossright;
	result = (int*)malloc(CITIES*sizeof(int));
	float randf = curand_uniform(&state[0]);
		
	crossleft = ((int)randf*100)%CITIES;
	randf = curand_uniform(&state[0]);
	crossright = ((int)randf*100)%CITIES;
	if(crossleft > crossright)
	{
		int tmp = crossleft;
		crossleft = crossright;
		crossright = tmp;
	}
	while(crossleft >= crossright)
	{
		randf = curand_uniform(&state[0]);
		crossleft = ((int)randf*100)%CITIES;
		randf = curand_uniform(&state[0]);
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
	return result;
	
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
	//for(int i = 0; i < CITIES; i++)
	//	printf("%d ", result[i]);		
//tour[i] = result[i];
	return result;
}

__device__ void crossover(int *A,int *B, curandState *state){
	int *C;

	int crossleft, crossright;
	float randf = curand_uniform(&state[0]);
		
	crossleft = ((int)randf*100)%CITIES;
	randf = curand_uniform(&state[0]);
	crossright = ((int)randf*100)%CITIES;
	if(crossleft > crossright)
	{
		int tmp = crossleft;
		crossleft = crossright;
		crossright = tmp;
	}
	while(crossleft >= crossright)
	{
		randf = curand_uniform(&state[0]);
		crossleft = ((int)randf*100)%CITIES;
		randf = curand_uniform(&state[0]);
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
	__shared__ int mean[CITIES];
	__shared__ int block_teacher[CITIES];
	__shared__ int block_teacher_dis;
//	__shared__ int global_dis;
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int j = 0; j < CITIES ; j++)
		subPop[threadIdx.x][j] = gpupopulation[id * CITIES + j];

	if(blockIdx.x == 0 && threadIdx.x == 0)
	{
		best_sol_dis = INT_MAX;
		best_sol = (volatile int*)malloc(CITIES*sizeof( volatile int));
	}
//	global_dis = INT_MAX;

	
	if(threadIdx.x == 0)
	{
		block_teacher_dis = INT_MAX;
	}
	__syncthreads();
	
	//Calculate fitness
	int dis = 0;
	for(int i = 0; i < CITIES-1 ; i++)
	{
		dis += gpuDistanceMat[subPop[threadIdx.x][i] * CITIES + subPop[threadIdx.x][i+1]];
	}
	dis += gpuDistanceMat[subPop[threadIdx.x][CITIES-1] * CITIES + subPop[threadIdx.x][0]];
	fitness[threadIdx.x] = dis;
	__syncthreads();
	//Global Teacher
	atomicMin((int*)&best_sol_dis, fitness[threadIdx.x]);
	/*if( old != global_dis )
	{
		//best_sol = subPop[threadIdx.x];
		best_sol_dis = global_dis;
		printf("%d\n", best_sol_dis);
		for(int i = 0; i < CITIES; i++)
			best_sol[i] = subPop[threadIdx.x][i];
	}*/
	
	__syncthreads();
	if(threadIdx.x == 0 && 0)
		printf("best sol = %d\n", best_sol_dis);
	
	//Subpopulation Teacher
	int old = atomicMin(&block_teacher_dis, fitness[threadIdx.x]);
	if( old != block_teacher_dis )
	{
		for(int i = 0; i < CITIES; i++)
			block_teacher[i] = subPop[threadIdx.x][i];
	}
	//PUT BARRIER HERE
	atomicDec((unsigned int*)&var, 0);
	while( var != 0 );
	//
	if(threadIdx.x == 0 && best_sol_dis == block_teacher_dis)
	{
		
		printf("%d\n", best_sol_dis);
		for(int i = 0; i < CITIES; i++)
			best_sol[i] = block_teacher[i];
		var = 100;
	}
	if(threadIdx.x == 0 )
		printf("Block = %d : Block Teacher = %d, Global teacher = %d\n",blockIdx.x, block_teacher_dis, best_sol_dis);
	for(int c = 0; c < 100; c++)
	{
	//	TEACHER PHASE
		//1. Calculate Mean
		memset(mean, 0, CITIES*sizeof(int));
		for(int j = 0; j < CITIES; j++)
			atomicAdd(&mean[j], subPop[threadIdx.x][j]);
		__syncthreads();
		if(threadIdx.x == 0 )
		{
			for(int j = 0; j < CITIES; j++)
				mean[j] = mean[j]/SUB_PS;
			int *mean_v = viability_op(mean);
			for(int j = 0; j < CITIES; j++)
				mean[j] = mean_v[j];
		}
	
		//2. Teacher Iteration
		int *newA;
		int  *C;
		int crossleft, crossright;
		float randf = curand_uniform(&state[threadIdx.x]);
		int newA_dis = 0;;
		int *result;//[CITIES];
		__syncthreads();
		switch(threadIdx.x) //threadIdx.x == CROSS_GLOBALTEACHER)
		{
			case CROSS_GLOBALTEACHER:
					{
						//printf("Block %d : subPop x globalTeacher\n", blockIdx.x);
						//2.1 CROSSOVER
						newA = (int*)malloc(CITIES*sizeof(int));
						for(int j = 0; j < CITIES; j++)
							newA[j] = subPop[threadIdx.x][j];

						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j=crossleft;j <= crossright;j++)
							newA[j] = (int)best_sol[j];
						C = viability_op(newA);
						
						//2.2 MUTATION	
						result = (int*)malloc(CITIES*sizeof(int));
						
						randf = curand_uniform(&state[threadIdx.x]);
						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j = 0; j < CITIES; j++)
						{
							result[j] = C[j];
						}
						for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
						{
							result[i]=C[j];
						}

						for(int i = 0; i < CITIES; i++)
							newA[i] = result[i];
						
					
						break;
					}
			case CROSS_LOCALTEACHER:
					{
						//printf("Block %d : subPop x localTeacher\n", blockIdx.x);
						//2.1 CROSSOVER
						newA = (int*)malloc(CITIES*sizeof(int));
						for(int j = 0; j < CITIES; j++)
							newA[j] = subPop[threadIdx.x][j];

						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j=crossleft;j <= crossright;j++)
							newA[j] = block_teacher[j];
						C = viability_op(newA);
						
						//2.2 MUTATION	
						result = (int*)malloc(CITIES*sizeof(int));
						
						randf = curand_uniform(&state[threadIdx.x]);
						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j = 0; j < CITIES; j++)
						{
							result[j] = C[j];
						}
						for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
						{
							result[i]=C[j];
						}

						for(int i = 0; i < CITIES; i++)
							newA[i] = result[i];

						break;
					}
				case CROSS_MEAN:
					{
						//printf("Block %d : subPop x mean\n", blockIdx.x);
						//2.1 CROSSOVER
						newA = (int*)malloc(CITIES*sizeof(int));
						for(int j = 0; j < CITIES; j++)
							newA[j] = subPop[threadIdx.x][j];

						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j=crossleft;j <= crossright;j++)
							newA[j] = mean[j];
						C = viability_op(newA);
						
						//2.2 MUTATION	
						result = (int*)malloc(CITIES*sizeof(int));
						
						randf = curand_uniform(&state[threadIdx.x]);
						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j = 0; j < CITIES; j++)
						{
							result[j] = C[j];
						}
						for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
						{
							result[i]=C[j];
						}

						for(int i = 0; i < CITIES; i++)
							newA[i] = result[i];
						break;
					}
				case CROSS_LOCALMEAN:
					{
						//printf("Block %d : localTeacher x mean\n", blockIdx.x);
						//2.1 CROSSOVER
						newA = (int*)malloc(CITIES*sizeof(int));
						for(int j = 0; j < CITIES; j++)
							newA[j] = mean[j];

						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j=crossleft;j <= crossright;j++)
							newA[j] = block_teacher[j];
						C = viability_op(newA);
						
						//2.2 MUTATION	
						result = (int*)malloc(CITIES*sizeof(int));
						
						randf = curand_uniform(&state[threadIdx.x]);
						crossleft = ((int)(randf*100))%CITIES;
						randf = curand_uniform(&state[threadIdx.x]);
						crossright = ((int)(randf*100))%CITIES;
						if(crossleft > crossright)
						{
							int tmp = crossleft;
							crossleft = crossright;
							crossright = tmp;
						}
						while(crossleft >= crossright)
						{
							randf = curand_uniform(&state[threadIdx.x]);
							crossleft = ((int)(randf*100))%CITIES;
							randf = curand_uniform(&state[threadIdx.x]);
							crossright = ((int)(randf*100))%CITIES;
						}
						for(int j = 0; j < CITIES; j++)
						{
							result[j] = C[j];
						}
						for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
						{
							result[i]=C[j];
						}

						for(int i = 0; i < CITIES; i++)
							newA[i] = result[i];
						break;
					}
		
		}
		for(int i = 0; i < CITIES-1; i++)
			newA_dis += gpuDistanceMat[newA[i] * CITIES + newA[i+1]];

		newA_dis += gpuDistanceMat[newA[CITIES-1] * CITIES + newA[0]];
		if(fitness[threadIdx.x] > newA_dis)
		{
			fitness[threadIdx.x] = newA_dis;
			for(int i = 0; i < CITIES; i++)
				subPop[threadIdx.x][i] = newA[i];
		}
		
		/* LEARNER PHASE */
		
		randf = curand_uniform(&state[threadIdx.x]);
		int randomK = ((int)(randf*100))%4;
		while( randomK == threadIdx.x )
		{
			randf = curand_uniform(&state[threadIdx.x]);
			randomK = ((int)(randf*100))%4;
		}

		for(int i = 0; i < CITIES; i++)
			newA[i] = subPop[threadIdx.x][i];
		randf = curand_uniform(&state[threadIdx.x]);
		crossleft = ((int)(randf*100))%CITIES;
		randf = curand_uniform(&state[threadIdx.x]);
		crossright = ((int)(randf*100))%CITIES;
		if(crossleft > crossright)
		{
			int tmp = crossleft;
			crossleft = crossright;
			crossright = tmp;
		}
		while(crossleft >= crossright)
		{
			randf = curand_uniform(&state[threadIdx.x]);
			crossleft = ((int)(randf*100))%CITIES;
			randf = curand_uniform(&state[threadIdx.x]);
			crossright = ((int)(randf*100))%CITIES;
		}
		for(int j=crossleft;j <= crossright;j++)
			newA[j] = subPop[randomK][j];
		C = viability_op(newA);
		
		//2.2 MUTATION	
		
		randf = curand_uniform(&state[threadIdx.x]);
		crossleft = ((int)(randf*100))%CITIES;
		randf = curand_uniform(&state[threadIdx.x]);
		crossright = ((int)(randf*100))%CITIES;
		if(crossleft > crossright)
		{
			int tmp = crossleft;
			crossleft = crossright;
			crossright = tmp;
		}
		while(crossleft >= crossright)
		{
			randf = curand_uniform(&state[threadIdx.x]);
			crossleft = ((int)(randf*100))%CITIES;
			randf = curand_uniform(&state[threadIdx.x]);
			crossright = ((int)(randf*100))%CITIES;
		}
		for(int j = 0; j < CITIES; j++)
		{
			result[j] = C[j];
		}
		for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
		{
			result[i]=C[j];
		}

		for(int i = 0; i < CITIES; i++)
			newA[i] = result[i];

		newA_dis = 0;
		for(int i = 0; i < CITIES-1; i++)
			newA_dis += gpuDistanceMat[newA[i] * CITIES + newA[i+1]];

		newA_dis += gpuDistanceMat[newA[CITIES-1] * CITIES + newA[0]];
		if(fitness[threadIdx.x] > newA_dis)
		{
			fitness[threadIdx.x] = newA_dis;
			for(int i = 0; i < CITIES; i++)
				subPop[threadIdx.x][i] = newA[i];
		}

		
		atomicMin((int*)&best_sol_dis, fitness[threadIdx.x]);
		old = atomicMin(&block_teacher_dis, fitness[threadIdx.x]);
		if( old != block_teacher_dis )
		{
			for(int i = 0; i < CITIES; i++)
				block_teacher[i] = subPop[threadIdx.x][i];
		}
		//PUT BARRIER HERE
		atomicDec((unsigned int*)&var, 0);
		while( var != 0 );
		//
		if(threadIdx.x == 0 && best_sol_dis == block_teacher_dis)
		{
		
			printf("%d\n", best_sol_dis);
			for(int i = 0; i < CITIES; i++)
				best_sol[i] = block_teacher[i];
			var = 100;
		}

		//printf("All operations performed : %d\n", id);
		atomicDec((unsigned int*)&itr, 0);
		while(itr != 0);
	}

	if(blockIdx.x == 0 && threadIdx.x == 0)
	{
		printf("Best Solution :: %d\n", best_sol_dis);
		for(int i = 0; i < CITIES; i++)
			printf("%d ", best_sol[i]);
		printf("\n");
	}
	
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
