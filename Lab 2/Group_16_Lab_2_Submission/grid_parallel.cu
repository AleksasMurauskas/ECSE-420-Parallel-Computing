#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#define n 0.0002
#define p 0.5
#define G 0.75
#define N 4
__global__ void interior(float* u, float* u1, float* u2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (((blockIdx.x > 0) && (blockIdx.x < blockDim.x-1)) && (threadIdx.x > 0 && threadIdx.x < blockDim.x-1)) {
		u[i] = (p * (u1[i-blockDim.x] + u1[i+blockDim.x] + u1[i-1] + u1[i+1] - 4*u1[i]) + 2*u1[i] - (1 - n)*(u2[i]))/(1+n);
	}
}

__global__ void side(float*u) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x == 0 && (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1)) {
		u[i] = G * u[blockDim.x+i];
	}
	else if (blockIdx.x == blockDim.x - 1 && (threadIdx.x != 0 && threadIdx.x != blockDim.x - 1)) {
		u[i] = G * u[i-blockDim.x];
	}
	else if (threadIdx.x == 0 && (blockIdx.x > 0 && blockIdx.x < blockDim.x - 1)) {
		u[i] = G * u[i+1];
	}
	else if (threadIdx.x == blockDim.x - 1 && (blockIdx.x > 0 && blockIdx.x < blockDim.x - 1)) {
		u[i] = G * u[i-1];
	}
}

__global__ void corners(float* u) {
	u[0] = G * u[1];
	u[N-1] = G * u[N - 2];
	u[N*(N-1)] = G * u[N*(N-2)];
	u[N*(N-1)+(N-1)] = G * u[N*(N-1)+(N-2)];

}
int main(int argc, char* argv[]) {
 if(argc<2) {
		printf("Not enough arguments.\n");
		return -1;
	}
	int T = atoi(argv[1]);

  float* cuda_u, * cuda_u1, * cuda_u2;
  cudaMallocManaged(&cuda_u, N * N * sizeof(float));
  cudaMallocManaged(&cuda_u1, N * N * sizeof(float));
  cudaMallocManaged(&cuda_u2, N * N * sizeof(float));
  
  cuda_u1[((N * N)/ 2 + N / 2)]=1; // Drum hit	
	
  GpuTimer timer;
  timer.Start();
	for (int iter = 0; iter < T; iter++) {
      interior<<<4,4>>>(cuda_u, cuda_u1, cuda_u2);
      cudaDeviceSynchronize();
      side<<<4,4>>>(cuda_u);
      cudaDeviceSynchronize();
      corners<<<4,4>>>(cuda_u);
      cudaDeviceSynchronize();
      
      // Print result
      printf("Iteration %d | u(N/2, N/2): %f\n", iter, cuda_u[(N*(N/2))+N/2]);
        
    	// Update u1 and u2
      for (int i=0; i<N*N; i++) {
          cuda_u2[i]=cuda_u1[i];
          cuda_u1[i] = cuda_u[i];
      }
      cudaFree(cuda_u);
      cudaMallocManaged(&cuda_u, N*N*sizeof(float));
	}
  
  timer.Stop();
  printf("Time elapsed for %d iterations: %lf\n", T, timer.Elapsed());

  cudaFree(cuda_u);
  cudaFree(cuda_u1);
  cudaFree(cuda_u2);
	return 0;
}