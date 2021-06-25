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
#define N 512

__global__ void synthesis(float* u, float* u1, float* u2, int numBlocks, int threadsPerBlock) {
  for (int i = threadIdx.x * numBlocks + blockIdx.x; i < N*N; i+= threadsPerBlock*numBlocks) {
    // corners
    if (i == 0 || i == N-1 || i == (N-1) * N || i == (N*N-1)) {
      u[i] = G * G * ((p * (u1[i+1] + u1[i+2*N+1] + u1[i+N] + u1[i+N+2] - 4 * u1[i+N+1]) + 2 * u1[i+N+1] - (1-n) * u2[i+N+1]) / (1+n));
    }
    // borders
    else if (i < N) {
      u[i] = G * ((p * (u1[i] + u1[i+2*N] + u1[i+N-1] + u1[i+N+1] - 4 * u1[i+N]) + 2 * u1[i+N] - (1-n) * u2[i+N]) / (1+n));
    }
    else if (i > N * N - N) {
      u[i] = G * ((p * (u1[i-2*N] + u1[i] + u1[i-N-1] + u1[i-N+1] - 4 * u1[i-N]) + 2 * u1[i-N] - (1-n) * u2[i-N]) / (1+n));
    }
    else if (i % N == 0) {
      u[i] = G * ((p * (u1[i+1-N] + u1[i+1+N] + u1[i] + u1[i+2] - 4 * u1[i+1]) + 2 * u1[i+1] - (1-n) * u2[i+1]) / (1 + n));
    }
    else if (i % N == N - 1) {
      u[i] = G * ((p * (u1[i-1-N] + u1[i-1+N] + u1[i-2] + u1[i] - 4 * u1[i-1]) + 2 * u1[i-1] - (1-n) * u2[i-1]) / (1+n));
    }
    // interior
    else {
      u[i] = (p * (u1[i-N] + u1[i+N] + u1[i-1] + u1[i+1] - 4 * u1[i]) + 2 * u1[i] - (1-n) * u2[i]) / (1+n);
    }
  }    
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
  
  cuda_u1[((N * N)/ 2 + N / 2)]=1.0; // Drum hit	
	int numBlocks = 128;
  int threadsPerBlock = 1024;
  GpuTimer timer;
  timer.Start();
	for (int iter = 0; iter < T; iter++) {
      synthesis<<<numBlocks,threadsPerBlock>>>(cuda_u, cuda_u1, cuda_u2, numBlocks, threadsPerBlock);
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