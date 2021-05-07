#include<cuda.h>
#include<bits/stdc++.h>
#include<curand.h>
#include<curand_kernel.h>
#define PS 100
#define SUB_PS 4
#define CITIES 1002
#define CYCLES 2000
#define CROSS_GLOBALTEACHER 0
#define CROSS_LOCALTEACHER 1
#define CROSS_MEAN 2
#define CROSS_LOCALMEAN 3

using namespace std;

__device__ volatile int *best_sol;
__device__ volatile int best_sol_dis;
__device__ volatile unsigned int var = 100;
__device__ volatile unsigned int itr = 100;

__device__ void viability_op(int *tour)
{
	int tempA[CITIES], tempB[CITIES], tempC[CITIES];
	memset(tempA, -1, CITIES*sizeof(int));
	memset(tempB, -1, CITIES*sizeof(int));
	memset(tempC, -1, CITIES*sizeof(int));

	int count[CITIES];
	memset(count, 0, CITIES*sizeof(int));
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
	int result[CITIES];//[CITIES] = {-1};
	int i = 0;
	while(i < CITIES)
	{
		result[i] = tempA[i];
		if(result[i] == -1) result[i] = tempC[i];
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
		tour[i] = result[i];
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
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int j = 0; j < CITIES ; j++)
		subPop[threadIdx.x][j] = gpupopulation[id * CITIES + j];

	if(blockIdx.x == 0 && threadIdx.x == 0)
	{
		best_sol_dis = INT_MAX;
		best_sol = (volatile int*)malloc(CITIES*sizeof( volatile int));
	}

	if(threadIdx.x == 0)
	{
		block_teacher_dis = INT_MAX;
	}
	
	//Calculate fitness
	int dis = 0;
	for(int i = 0; i < CITIES-1 ; i++)
	{
		dis += gpuDistanceMat[subPop[threadIdx.x][i] * CITIES + subPop[threadIdx.x][i+1]];
	}
	dis += gpuDistanceMat[subPop[threadIdx.x][CITIES-1] * CITIES + subPop[threadIdx.x][0]];
	fitness[threadIdx.x] = dis;
	//Global Teacher
	atomicMin((int*)&best_sol_dis, fitness[threadIdx.x]);
	
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
	//atomicDec((unsigned int*)&var, 0);
	atomicAdd((int*)&var, -1);
//	printf("var = %d ", var);
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
	for(int c = 0; c < CYCLES; c++)
	{
	//	TEACHER PHASE
		//1. Calculate Mean
		if(blockIdx.x == 0 && threadIdx.x == 0 )
			itr = 100;
		while( itr != 100 );
		memset(mean, 0, CITIES*sizeof(int));
		for(int j = 0; j < CITIES; j++)
			atomicAdd(&mean[j], subPop[threadIdx.x][j]);
		__syncthreads();
		if(threadIdx.x == 0 )
		{
			for(int j = 0; j < CITIES; j++)
				mean[j] = mean[j]/SUB_PS;
			viability_op(mean);
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
						viability_op(newA);
						C = newA;
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
						viability_op(newA);
						C = newA;
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
						viability_op(newA);
						C = newA;
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
						viability_op(newA);
						C = newA;
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
		
	//	 LEARNER PHASE 	
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
		if(newA == NULL)
			printf("-----newA is null-----\n");
		for(int j=crossleft;j <= crossright;j++)
			newA[j] = subPop[randomK][j];
	//	viability_op(newA);
	//	C = newA;
		int tempA[CITIES], tempB[CITIES], tempC[CITIES];
		memset(tempA, -1, CITIES*sizeof(int));
		memset(tempB, -1, CITIES*sizeof(int));
		memset(tempC, -1, CITIES*sizeof(int));

		int count[CITIES];
		memset(count, 0, CITIES*sizeof(int));
		for(int i = 0; i < CITIES; i++)
			count[newA[i]]++;

		for(int i = CITIES-1; i >= 0; i--)
		{
			if(count[newA[i]] > 1)
			{
				tempA[i] = newA[i];
				count[newA[i]] = -1;
			}
			if(count[newA[i]] == 1)
			{
				tempC[i] = newA[i];
				count[newA[i]] = -1;
			}
		}	
		for(int i = 0; i < CITIES; i++)
		{
			if(count[i] == 0)
				tempB[i] = i;
		}
		int result2[CITIES];//[CITIES] = {-1};
		//result2 = (int*)malloc(CITIES*sizeof(int));
		int i = 0;
		while(i < CITIES)
		{
			result2[i] = tempA[i];
			if(result2[i] == -1) result2[i] = tempC[i];
			//	result[i] = tempB[i];
			i++;
		}
		int j = 0, k = 0;
		while( k < CITIES && j < CITIES)
		{
			if(tempB[k] == -1)
				k++;
			else
			{
				if(result2[j] == -1)
				{
					result2[j] = tempB[k];
					j++; k++;
				}
				else
				{
					j++;
				}
			}
		}
		/*for(int i = 0, j = 0; j < CITIES && i < CITIES;)
		{
			if(result2[j] == -1)
			{
				if(tempB[i] == -1)
					i++;
				else
				{
					result[j] = tempB[i];
					i++;
					j++;
				}
			}
			else
			{
				j++;
			}
		}*/

		for(int j = 0; j < CITIES; j++)
			newA[j] = result2[j];
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
		int CC[CITIES];
		for(int j = 0; j < CITIES; j++)
			CC[j] = newA[j];
		for(int j = 0; j < CITIES; j++)
		{
			result[j] = CC[j];
		}
		for(int i=crossleft,j=crossright;i<=crossright&&j>=crossleft;i++,j--)
		{
			result[i]=CC[j];
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
		//atomicDec((unsigned int*)&var, 0);
		atomicAdd((int*)&var, -1);
		while( var != 0 );
		//
		if(threadIdx.x == 0 && best_sol_dis == block_teacher_dis)
		{
		
		//	printf("%d : %d\n",c, best_sol_dis);
			for(int i = 0; i < CITIES; i++)
				best_sol[i] = block_teacher[i];
			var = 100;
		}

		//printf("All operations performed : %d\n", id);
		//atomicDec((unsigned int*)&itr, 0);
		atomicAdd((int*)&itr, -1);
		while(itr != 0);
		free(result);
		//free(result2);
		free(newA);
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
int main(int argc, char **argv)
{
	srand(time(NULL));
	FILE *input;

	input = fopen(argv[1], "r");
	
	if(input == NULL)
		printf("error: failed to open input file\n");

	
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	
	int numberOfCities;
	vector<pair<float, float> > points;
	
	fscanf(input, "%d", &numberOfCities);
	
	for(int i = 0; i < numberOfCities; i++)
	{
		float x , y;
		fscanf(input, "%f", &x);
		fscanf(input, "%f", &y);

		points.push_back(make_pair(x, y));
	}

	int *distanceMat ;//= (int*)malloc(numberOfCities*numberOfCities*sizeof(int));
	cudaHostAlloc(&distanceMat, numberOfCities*numberOfCities*sizeof(int), cudaHostAllocMapped);
/*	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
			scanf("%d", &distanceMat[i*numberOfCities+j]);
	}*/
	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
		{
			int ed;
			int x = (points[j].first - points[i].first)*(points[j].first - points[i].first);
			int y = (points[j].second - points[i].second)*(points[j].second - points[i].second);
			ed = sqrt(x+y);
			distanceMat[i*numberOfCities+j] = ed;
		}
	}
	
	int noOfBlocks = ceil((float)PS/SUB_PS);
	int *population; //= (int*)malloc(PS*CITIES*sizeof(int));
	cudaHostAlloc(&population, PS*CITIES*sizeof(int), cudaHostAllocMapped);
	createPopulation(population);

	setup_kernel<<<noOfBlocks,SUB_PS>>>(d_state);
	tlboKernel<<<noOfBlocks, SUB_PS>>>(population, distanceMat, numberOfCities, d_state);
		
	cudaDeviceSynchronize();

	return 0;
}
