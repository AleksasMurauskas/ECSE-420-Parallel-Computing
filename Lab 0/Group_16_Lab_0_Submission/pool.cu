#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void pool_process(unsigned char* input, unsigned char* output, int block_per_thread, int width,int loop_num, int numthreads) {
  int index = threadIdx.x;
  int offset= index* block_per_thread;
  if(loop_num!=1){
      offset= ((index+numthreads)* block_per_thread);
  }
  unsigned found_max;
  unsigned char* active_ptr;
  unsigned active_val;
  for(int x=offset;x<offset+block_per_thread;x++){//Iterates through output pixels 
      for(int y=0; y<4;y++){//Cycle through RGBA
          if(y<3){
              found_max=0;
              for(int z=0;z<2;z++){//Process Pixels
                  //Get offset
                  int block_offset = 8* (width * (x / (width / 2)) +  (x % (width / 2)));
                  active_ptr = input+ block_offset+(4*width*z);
                  for(int a=0; a<2;a++){
                      active_ptr += 4*a;
                      active_val= (int)active_ptr[y];
                      if(found_max < active_val){
                          found_max=active_val;
                      }
                  }
              }
              output[4*x+y] = (unsigned char) found_max;
          } else{
            //Fix alpha Channel
            output[4*x+y] =(unsigned char) 225;
          }
      }
  }	
}

int main(int argc, char *argv[]) {
  if (argc < 4) { 
      printf("Run the following command:\n ./pool <input png> <output png> <# threads>");
  }
  int nThreads = atoi(argv[3]); // number of threads
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
  int output_pixels=image_size/4;
  cudaMalloc((void**) & cuda_input, image_size);
  cudaMalloc((void**) & cuda_output, output_pixels);

  cudaMemcpy(cuda_input, image, image_size, cudaMemcpyHostToDevice);

  unsigned char * cuda_inputCpy = cuda_input,
  * cuda_outputCpy = cuda_output;
  //int nPixels = width * height;
  //int pxLeft = nPixels/4;
  int blocks_per_thread =  output_pixels/ nThreads;
  int leftover = output_pixels/nThreads;
  int round;
  if(blocks_per_thread==0){
      blocks_per_thread=1;
      round=2;
  }
  else{
      round=1;
      pool_process <<<1,nThreads>>> (cuda_inputCpy, cuda_outputCpy, blocks_per_thread, width, round,nThreads);
  }
  if(leftover!=0){
  round =2; 
  blocks_per_thread=1;
  pool_process <<<1,(leftover/blocks_per_thread)>>> (cuda_inputCpy, cuda_outputCpy, blocks_per_thread, width, round, nThreads);
  }

  cudaDeviceSynchronize();
  //Encode as PNG 
  output_image = (unsigned char*)malloc(image_size/4);
  cudaMemcpy(output_image, cuda_output, image_size/4, cudaMemcpyDeviceToHost);
  lodepng_encode32_file(output_filename, output_image, width/2, height/2);

  cudaFree(cuda_input);
  cudaFree(cuda_output);
  free(output_image);
  return 0; 
}