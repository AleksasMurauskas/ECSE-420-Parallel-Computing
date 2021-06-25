#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

int main(int argc, char* argv[]) {
 if(argc<2) {
		printf("Not enough arguments.\n");
		return -1;
	}
	int T = atoi(argv[1]);
	float n = 0.0002;
	float p = 0.5;
	float G = 0.75;
	int N = 4;
 
	float* u = (float*)calloc(N*N, N * N * sizeof(float));
	float* u1 = (float*)calloc(N*N, N * N * sizeof(float));
	float* u2 = (float*)calloc(N*N, N * N * sizeof(float));

  // Drum hit
	u1[((N * N)/ 2 + N / 2)] = 1;
	
	clock_t start = clock();
	for (int iter = 0; iter < T; iter++) {
	
    	// Interior elements
    	for (int row = 1; row <= N-2; row++) {
    		for (int col = 1; col <= N-2; col++) {
    			u[N*row + col] = (p * (u1[N*(row-1)+col] + u1[N*(row+1)+col] + u1[row*N+(col-1)] + u1[(N*row)+col+1] - 4*u1[N*row+col]) 
                              + 2*u1[N*row+col] - (1-n)*(u2[N*row+col]))/(1+n);
    		}
    	}
        
      // Side elements
			for (int u_row = 1; u_row <= N-2; u_row++){
    	    u[N*u_row] = G*u[N*u_row+1];
    	    u[(N*u_row)+(N-1)] = G*u[(N*u_row)+(N-2)];
    	}
    	for (int col=1; col <= N-2; col++){
    	    u[col] = G*u[N+col];
    	    u[(N*(N-1))+col] = G*u[(N*(N-2))+col];
    	}
    	
      // Corners
    	u[0] = G*u[1];
    	u[N-1] = G*u[N-2];
    	u[N*(N-1)] = G*u[N*(N-2)];
    	u[N*(N-1)+(N-1)] = G*u[N*(N-1)+(N-2)];
        
      // Print result
      printf("Iteration %d | u(N/2, N/2): %f\n", iter, u[(N*(N/2))+N/2]);
        
    	// Update u1 and u2
    	u2 = u1;
    	u1 = u;
    	u = (float*)calloc(N * N, N * N * sizeof(float));
    	
	}
  
  clock_t end = clock();
  printf("Time elapsed for %d iterations: %lf\n", T, (double)(end-start)/1000);

  free(u);
  free(u1);
  free(u2);
	return 0;
}