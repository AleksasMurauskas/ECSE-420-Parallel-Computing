#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*

*/


__global__ void rectify(unsigned const char* input, unsigned char* output, int nBytes, int nThreads) {
  for (int i=threadIdx.x; i<nBytes; i+=nThreads) {
		if (input[i] < 127) {
			output[i] = 127;
		} else {
			output[i] = input[i];
		}
	}
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
      printf("Run the following command:\n ./rectify <input png> <output png> <# threads>");
      //exit(0)
  }

  int nThreads = atoi(argv[3]); // number of threads
  if (nThreads<=0){
      nThreads= 1;
  }
  if (nThreads > 1024) {
      printf("Max number of threads if 1024.");
      nThreads = 1024;
  }

  char* input_filename = argv[1];
  char* output_filename = argv[2];
  unsigned char* image, *output_image;
  unsigned height, width;
  unsigned error;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  unsigned int image_size = width * height * 4 * sizeof(unsigned char);
  
  unsigned char* cuda_input, *cuda_output;
  cudaMalloc((void**) & cuda_input, image_size);
  cudaMalloc((void**) & cuda_output, image_size);

  cudaMemcpy(cuda_input, image, image_size, cudaMemcpyHostToDevice);

  unsigned char * cuda_inputCpy = cuda_input,
    * cuda_outputCpy = cuda_output;
    
  int nBytes = width * height * 4;
  //while (pxLeft > nThreads) {
    rectify <<< 1, nThreads >>> (cuda_inputCpy, cuda_outputCpy, nBytes, nThreads);

		//cuda_inputCpy += nThreads;
		//cuda_outputCpy += nThreads;
		//pxLeft -= nThreads;
  //}
  //rectify<<<1, pxLeft>>>(cuda_inputCpy, cuda_outputCpy);
  cudaDeviceSynchronize();

  output_image = (unsigned char*)malloc(image_size);
  cudaMemcpy(output_image, cuda_output, image_size, cudaMemcpyDeviceToHost);
  lodepng_encode32_file(output_filename, output_image, width, height);
  
  cudaFree(cuda_input);
  cudaFree(cuda_output);
  free(output_image);

  return 0;
}