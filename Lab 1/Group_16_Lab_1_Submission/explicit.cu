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
  
  //printf("lines in file: %d\n", lines);
  char* output_filename = argv[3];

  // Open input file and create output file
  fp = fopen(input_filename, "r");
  out = fopen(output_filename, "w");
  if (fp == NULL){
      printf("Could not open file %s",input_filename);
      return 1;
  }
  
  // Read the data in the file into an array
   char * line = NULL;
  int * bool1 = new int[lines];
  int * bool2 = new int[lines];
  int * op = new int[lines];
  int * results = new int[lines];

  for (int i = 0; i < lines; i++) {
    line = NULL;
    size_t n = 0;
    getline(&line, &n, fp);
    
    bool1[i] = (int)line[0]-48;
    bool2[i] = (int)line[2]-48;
    op[i] = (int)line[4]-48;
  }

  int *dev_a, *dev_b, *dev_c, *dev_res;
  int size = lines*sizeof(int);
  
  cudaMalloc((void**)&dev_a, size);
  cudaMalloc((void**)&dev_b, size);
  cudaMalloc((void**)&dev_c, size);
  cudaMalloc((void**)&dev_res, size);
  
  GpuTimer timer1, timer2;
  timer1.Start();
  cudaMemcpy(dev_a, bool1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, bool2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, op, size, cudaMemcpyHostToDevice);
  timer1.Stop();
  printf("Time elapsed for data migration: %f\n", timer1.Elapsed());
  
  // allocate memory
  timer2.Start();
  gate <<<lines/T+1,T>>> (dev_a, dev_b, dev_c, dev_res);
  timer2.Stop();
  printf("Time elapsed for execution: %f\n", timer2.Elapsed());
  cudaMemcpy(results, dev_res, size,cudaMemcpyDeviceToHost);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_res);


  
  // Write data to new output file
for (int i = 0; i < lines; i++) {
    //printf("result: %d\n", results[i]);
    fprintf(out, "%d\n", results[i]);
    //printf("===============\n");
  }
  // Close the files
  fclose(fp);
  fclose(out);
  return 0;
}