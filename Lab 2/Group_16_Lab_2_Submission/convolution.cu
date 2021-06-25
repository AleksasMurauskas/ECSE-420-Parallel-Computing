#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gputimer.h"
#include "lodepng.h"
#include "wm.h"

__global__ void convolve(unsigned char* input, unsigned char* output, unsigned width, unsigned height, int cycle, float* weights, int nThreads) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float temp[3 * 3];
	float conv;
	unsigned offset;
	if (i < nThreads) {
		for (int k = 0; k < 4; k++) {
			offset = cycle * nThreads + i;

			if ((offset % width) < (width - 3 + 1) && offset < width * (height - 3 + 1)) {
				conv = 0;
				for (int j = 0; j < 3 * 3; j++) {
					temp[j] = input[(offset + width * (j / 3) + (j - 3 * (j / 3))) * 4 + k];
					temp[j] = temp[j] * weights[j];
					conv += temp[j];
				}
				
				if (conv < 0.0) conv = 0;
				if (conv > 255.0) conv = 255;
				if (k == 3) conv = input[offset * 4 + k];

				output[(offset - (offset / width) * (3 - 1)) * 4 + k] = round(conv);
			}
		}
	}
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
    printf("Run the following command:\n ./convolve <input png> <output png> <# threads>");
    return -1;
  }

  int nThreads = atoi(argv[3]); // number of threads
  if (nThreads<=0){
      nThreads= 1;
  }

  char* input_filename = argv[1];
  char* output_filename = argv[2];unsigned error;
	unsigned char* input, * output;
	unsigned width, height;
	error = lodepng_decode32_file(&input, &width, &height, input_filename);
	if (error) {
      printf("error %u: %s\n", error, lodepng_error_text(error));
      return -1;
  }
	output = (unsigned char*)malloc((width - 2) * (height - 2) * sizeof(unsigned char));

	unsigned char* input_cuda;
	cudaMallocManaged((void**)&input_cuda, width * height * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)&output, (width - 2) * (height - 2) * 4 * sizeof(unsigned char));
	for (int i = 0; i < width * height * 4; i++) {
		input_cuda[i] = input[i];
	}
	for (int i = 0; i < (width - 2) * (height - 2) * 4; i++) output[i] = 0;
	
	float* weights;
	cudaMallocManaged((void**)&weights, 3 * 3 * sizeof(float));
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (3 == 3) weights[i * 3 + j] = w[i][j];
		}
	}
  GpuTimer timer;
	int cycle = 0;
 timer.Start();
	while (cycle < width * height / nThreads) {
		convolve <<<1, nThreads >>> (input_cuda, output, width, height, cycle, weights, nThreads);
		cycle++;
	}
	cudaDeviceSynchronize();
	timer.Stop();
	printf("Time elapsed for %d threads: %f\n", nThreads, timer.Elapsed());
	
	lodepng_encode32_file(output_filename, output, (width - 2), (height - 2));
	cudaFree(input); 
  cudaFree(output);
  cudaFree(input_cuda);
  cudaFree(weights);
	free(input);

	return 0;
}
#