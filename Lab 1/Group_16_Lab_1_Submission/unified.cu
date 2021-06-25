#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#define T 1024

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5
__global__ void gate(int *num1, int *num2, int *op, int *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int operation = op[i];
    int in1 = num1[i];
    int in2 = num2[i];

    if (operation == AND) res[i] = in1 & in2;
    else if (operation == OR) res[i] = in1 | in2;
    else if (operation == NAND) res[i] = !(in1 & in2);
    else if (operation == NOR) res[i] = !(in1 | in2);
    else if (operation == XOR) res[i] = in1 ^ in2;
    else if (operation == XNOR) res[i] = !(in1 ^ in2);
    else res[i] = -1;
}

int main(int argc, char *argv[]) {
  FILE *fp, *out;
  char* input_filename = argv[1];
  int lines = atoi(argv[2]);
  char* output_filename = argv[3];

  // Open input file and create output file
  fp = fopen(input_filename, "r");
  out = fopen(output_filename, "w");
  if (fp == NULL){
      printf("Could not open file %s",input_filename);
      return 1;
  }

  // Unified memory allocation
  char * line = NULL;
  int size = lines*sizeof(int);
  int *p_bool1, *p_bool2, *p_op, *p_results;

  cudaMallocManaged(&p_bool1, size);
  cudaMallocManaged(&p_bool2, size);
  cudaMallocManaged(&p_op, size);
  cudaMallocManaged(&p_results, size);

  // Read the data in the file
  for (int i = 0; i < lines; i++) {
    line = NULL;
    size_t n = 0;
    getline(&line, &n, fp);
    
    p_bool1[i] = (int)line[0]-48;
    p_bool2[i] = (int)line[2]-48;
    p_op[i] = (int)line[4]-48;
  }

  GpuTimer timer;
  timer.Start();
  gate <<<lines/T+1,T>>> (p_bool1, p_bool2, p_op, p_results);
  cudaDeviceSynchronize();
  timer.Stop();
  printf("Time elapsed: %f\n", timer.Elapsed());

  // Write data to new output file
  for (int i = 0; i < lines; i++) {
    //printf("result: %d\n", p_results[i]);
    fprintf(out, "%d\n", p_results[i]);
    //printf("===============\n");
  }
  
  cudaFree(p_bool1);
  cudaFree(p_bool2);
  cudaFree(p_op);
  cudaFree(p_results);
  // Close the files
  fclose(fp);
  fclose(out);
  return 0;
}