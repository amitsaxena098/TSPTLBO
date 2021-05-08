#include<cuda.h>
#include<bits/stdc++.h>
#include<curand.h>
#include<curand_kernel.h>
#define PS 500
#define SUB_PS 4
#define CITIES 48
#define CROSS_GLOBALTEACHER 0
#define CROSS_LOCALTEACHER 1
#define CROSS_MEAN 2
#define CROSS_LOCALMEAN 3

using namespace std;

__device__ volatile int *best_sol;
__device__ volatile int best_sol_dis;
__device__ volatile unsigned int var = PS;
__device__ volatile unsigned int itr = PS;


__global__ void setup_kernel(curandState *state)
{
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void tlboKernel(int *gpupopulation, int *gpuDistanceMat, int numberOfCities, curandState *state, int CYCLES)
{
	__shared__ int subPop[SUB_PS][CITIES];
	__shared__ int fitness[SUB_PS];
	__shared__ int mean[CITIES];
	__shared__ int block_teacher[CITIES];
	__shared__ int block_teacher_dis;
	
	
	int tempA[CITIES], tempB[CITIES], tempC[CITIES];
	int count[CITIES];
	int vresult[CITIES];
	
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int j = 0; j < CITIES ; j++)
	{
		subPop[threadIdx.x][j] = gpupopulation[id * CITIES + j];
	}
	
		
	//Initialize best solution
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
	{
		printf("best sol = %d\n", best_sol_dis);
	}
	
	
	//Subpopulation Teacher
	int old = atomicMin(&block_teacher_dis, fitness[threadIdx.x]);
	if( old != block_teacher_dis )
	{
		for(int i = 0; i < CITIES; i++)
		{
			block_teacher[i] = subPop[threadIdx.x][i];
		}
	}
	
	
	//PUT BARRIER HERE
	atomicAdd((int*)&var, -1);
	while( var != 0 );
	
	
	if(threadIdx.x == 0 && best_sol_dis == block_teacher_dis)
	{
		printf("%d\n", best_sol_dis);
		for(int i = 0; i < CITIES; i++)
		{
			best_sol[i] = block_teacher[i];
		}
		var = PS;
	}
	
	if(threadIdx.x == 0 )
	{
		printf("Block = %d : Block Teacher = %d, Global teacher = %d\n",blockIdx.x, block_teacher_dis, best_sol_dis);
	}
	for(int c = 0; c < CYCLES; c++)
	{
	
	//	TEACHER PHASE
	
		//1. Calculate Mean
		if(blockIdx.x == 0 && threadIdx.x == 0 )
		{
			itr = PS;
		}
		while( itr != PS );
		memset(mean, 0, CITIES*sizeof(int));
		for(int j = 0; j < CITIES; j++)
		{
			atomicAdd(&mean[j], subPop[threadIdx.x][j]);
		}
		__syncthreads();
		if(threadIdx.x == 0 )
		{
			for(int j = 0; j < CITIES; j++)
			{
				mean[j] = mean[j]/SUB_PS;	
			}
			//viability_op(mean);
			memset(tempA, -1, CITIES*sizeof(int));
			memset(tempB, -1, CITIES*sizeof(int));
			memset(tempC, -1, CITIES*sizeof(int));

			memset(count, 0, CITIES*sizeof(int));
			for(int i = 0; i < CITIES; i++)
			{
				count[mean[i]]++;
			}

			for(int i = CITIES-1; i >= 0; i--)
			{
				if(count[mean[i]] > 1)
				{
					tempA[i] = mean[i];
					count[mean[i]] = -1;
				}
				if(count[mean[i]] == 1)
				{
					tempC[i] = mean[i];
					count[mean[i]] = -1;
				}
			}	
			for(int i = 0; i < CITIES; i++)
			{
				if(count[i] == 0)
				{
					tempB[i] = i;
				}
			}

			int i = 0;
			while(i < CITIES)
			{
				vresult[i] = tempA[i];
				if(vresult[i] == -1) 
				{
					vresult[i] = tempC[i];
				}
				i++;
			}
			int j = 0;
			i = 0;
			while( i < CITIES)
			{
				if(tempB[i] == -1)
				{
					i++;
				}
				else
				{
					if(vresult[j] == -1)
					{
						vresult[j] = tempB[i];
						j++; i++;
					}
					else
					{
						j++;
					}
				}
			}
			for(int i = 0; i < CITIES; i++)
				mean[i] = vresult[i];
			
		}
		
		//2. Teacher Iteration
		int *newA;
		int  *C;
		int crossleft, crossright;
		float randf = curand_uniform(&state[threadIdx.x]);
		int newA_dis = 0;;
		int *result;
		__syncthreads();
		randf = curand_uniform(&state[threadIdx.x]);
		int crossType = ((int)(randf*100))%4;
		switch(crossType) 
		{
			case CROSS_GLOBALTEACHER:
					{

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

						memset(tempA, -1, CITIES*sizeof(int));
						memset(tempB, -1, CITIES*sizeof(int));
						memset(tempC, -1, CITIES*sizeof(int));

						memset(count, 0, CITIES*sizeof(int));
						
						for(int i = 0; i < CITIES; i++)
						{
							count[newA[i]]++;
						}
						
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
							{
								tempB[i] = i;
							}
						}

						int i = 0;
						while(i < CITIES)
						{
							vresult[i] = tempA[i];
							if(vresult[i] == -1) 
							{
								vresult[i] = tempC[i];
							}
							i++;
						}
						
						int j = 0;
						i = 0;
						while( i < CITIES)
						{
							if(tempB[i] == -1)
							{
								i++;
							}
							else
							{
								if(vresult[j] == -1)
								{
									vresult[j] = tempB[i];
									j++; i++;
								}
								else
								{
									j++;
								}
							}
						}
						
						for(int i = 0; i < CITIES; i++)
						{
							newA[i] = vresult[i];
							
						}
						
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
						{
							newA[i] = result[i];
						}
					
						break;
					}
			case CROSS_LOCALTEACHER:
					{

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

						memset(tempA, -1, CITIES*sizeof(int));
						memset(tempB, -1, CITIES*sizeof(int));
						memset(tempC, -1, CITIES*sizeof(int));

						memset(count, 0, CITIES*sizeof(int));
						for(int i = 0; i < CITIES; i++)
						{
							count[newA[i]]++;
						}
						
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
							{
								tempB[i] = i;
							}
						}

						int i = 0;
						while(i < CITIES)
						{
							vresult[i] = tempA[i];
							if(vresult[i] == -1) 
							{
								vresult[i] = tempC[i];
							}
							i++;
						}
						
						int j = 0;
						i = 0;
						while( i < CITIES)
						{
							if(tempB[i] == -1)
							{
								i++;
							}
							else
							{
								if(vresult[j] == -1)
								{
									vresult[j] = tempB[i];
									j++; i++;
								}
								else
								{
									j++;
								}
							}
						}
						
						for(int i = 0; i < CITIES; i++)
						{
							newA[i] = vresult[i];
							
						}
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
						{
							newA[i] = result[i];
						}

						break;
					}
				case CROSS_MEAN:
					{

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

						memset(tempA, -1, CITIES*sizeof(int));
						memset(tempB, -1, CITIES*sizeof(int));
						memset(tempC, -1, CITIES*sizeof(int));

						memset(count, 0, CITIES*sizeof(int));
						
						for(int i = 0; i < CITIES; i++)
						{
							count[newA[i]]++;
						}
						
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
							{
								tempB[i] = i;
							}
						}

						int i = 0;
						while(i < CITIES)
						{
							vresult[i] = tempA[i];
							if(vresult[i] == -1) 
							{
								vresult[i] = tempC[i];
							}
							i++;
						}
						
						int j = 0;
						i = 0;
						while( i < CITIES)
						{
							if(tempB[i] == -1)
							{
								i++;
							}
							else
							{
								if(vresult[j] == -1)
								{
									vresult[j] = tempB[i];
									j++; i++;
								}
								else
								{
									j++;
								}
							}
						}
						
						for(int i = 0; i < CITIES; i++)
						{
							newA[i] = vresult[i];
							
						}
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
						{
							newA[i] = result[i];
						}
						break;
					}
				case CROSS_LOCALMEAN:
					{

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

						memset(tempA, -1, CITIES*sizeof(int));
						memset(tempB, -1, CITIES*sizeof(int));
						memset(tempC, -1, CITIES*sizeof(int));

						memset(count, 0, CITIES*sizeof(int));
						
						
						for(int i = 0; i < CITIES; i++)
						{
							count[newA[i]]++;
						}
						
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
							{
								tempB[i] = i;
							}
						}

						int i = 0;
						while(i < CITIES)
						{
							vresult[i] = tempA[i];
							if(vresult[i] == -1) 
							{
								vresult[i] = tempC[i];
							}
							i++;
						}
						
						int j = 0;
						i = 0;
						while( i < CITIES)
						{
							if(tempB[i] == -1)
							{
								i++;
							}
							else
							{
								if(vresult[j] == -1)
								{
									vresult[j] = tempB[i];
									j++; i++;
								}
								else
								{
									j++;
								}
							}
						}
						
						for(int i = 0; i < CITIES; i++)
						{
							newA[i] = vresult[i];
							
						}
						
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
						{
							newA[i] = result[i];
						}
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
		
		
		//LEARNER PHASE 	

		int randomK = -1; 
		
		for(int i = 0; i < 4; i++)
		{
			if(i != threadIdx.x && fitness[i] <= fitness[threadIdx.x])
			{
				randomK = i;
				break;
			}
		}
		if(randomK == -1)
		{
			randomK = threadIdx.x;
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
			{
				tempB[i] = i;
			}
		}
		int result2[CITIES];

		int i = 0;
		while(i < CITIES)
		{
			result2[i] = tempA[i];
			if(result2[i] == -1) 
			{
				result2[i] = tempC[i];
			}

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

		atomicAdd((int*)&var, -1);
		while( var != 0 );
		
		if(threadIdx.x == 0 && best_sol_dis == block_teacher_dis)
		{
		

			for(int i = 0; i < CITIES; i++)
				best_sol[i] = block_teacher[i];
			var = PS;
		}

		atomicAdd((int*)&itr, -1);
	
		while(itr != 0);
		free(result);

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
	{
		printf("error: failed to open input file\n");
		return 0;
	}

	int CYCLES;
	
	sscanf(argv[2], "%d", &CYCLES);
	printf("Number of iterations of DTLBO = %d\n", CYCLES);
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	
	int numberOfCities;
	vector<pair<float, float> > points;
	
	fscanf(input, "%d", &numberOfCities);
	printf("Number of Cities = %d\n", numberOfCities);
	
	for(int i = 0; i < numberOfCities; i++)
	{
		float x , y;
		fscanf(input, "%f", &x);
		fscanf(input, "%f", &y);

		points.push_back(make_pair(x, y));
	}

	int *distanceMat ;
	cudaHostAlloc(&distanceMat, numberOfCities*numberOfCities*sizeof(int), cudaHostAllocMapped);
	
	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
		{
			float ed;
			float x = (points[j].first - points[i].first)*(points[j].first - points[i].first);
			float y = (points[j].second - points[i].second)*(points[j].second - points[i].second);
			ed = sqrt(x+y);
			distanceMat[i*numberOfCities+j] = floor(ed);
		}
	}
	
	printf("Generated distance matrix successfully...\n");
	int noOfBlocks = ceil((float)PS/SUB_PS);
	int *population;
	cudaHostAlloc(&population, PS*CITIES*sizeof(int), cudaHostAllocMapped);
		
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
     	float milliseconds = 0;
        cudaEventRecord(start,0);
 
	createPopulation(population);
	printf("Generated random population...\n");
	printf("Starting kernel now...Please wait\n");
	setup_kernel<<<noOfBlocks,SUB_PS>>>(d_state);
	tlboKernel<<<noOfBlocks, SUB_PS>>>(population, distanceMat, numberOfCities, d_state, CYCLES);
		
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
	
	return 0;
}
