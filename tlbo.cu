#include<cuda.h>
#include<bits/stdc++.h>
#include<curand.h>
#include<curand_kernel.h>
#define PS 100
#define SUB_PS 4
#define CITIES 15
using namespace std;


void viability_op(int *tour)
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
	int result[CITIES] = {-1};
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
	printf("\n\n");
}

__global__ void setup_kernel(curandState *state)
{
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void tlboKernel(int *gpupopulation, int *gpuDistanceMat, int numberOfCities, curandState *state)
{
	__shared__ int subPop[SUB_PS][CITIES];
//	__shared__ int fitness[SUB_PS];
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	for(int j = 0; j < CITIES ; j++)
		subPop[threadIdx.x][j] = gpupopulation[id * CITIES + j];

	__syncthreads();
	/*int count[CITIES] = {0};
	for(int i = 0; i < CITIES; i++)
	{
		float randf = curand_uniform(&state[id]);
		
		int ind = ((int)(randf*100))%CITIES;
		while(count[ind])
		{
			randf = curand_uniform(&state[id]);
		
			ind = ((int)(randf*100))%CITIES;
		
		}
		subPop[id][i] = ind;
		count[ind] = 1; 
	}*/
	//Calculate fitness
	int dis = 0;
	for(int i = 0; i < CITIES-1 ; i++)
	{
		//printf("%d\n",gpuDistanceMat[subPop[threadIdx.x][i] * CITIES + subPop[threadIdx.x][i+1]]);
		//printf("%d, %d\n", subPop[threadIdx.x][i], subPop[threadIdx.x][i+1]);
		dis += gpuDistanceMat[subPop[threadIdx.x][i] * CITIES + subPop[threadIdx.x][i+1]];
	}

	printf("dis = %d\n", dis);
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
