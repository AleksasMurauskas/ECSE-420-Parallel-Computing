#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

int read_input_one_two_four(int **input1, char* filepath){
 FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    } 
    
    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = ( int *)malloc(len * sizeof(int));

    int temp1;

    while (fscanf(fp, "%d", &temp1) == 1) {
        (*input1)[counter] = temp1;

        counter++;
    }

    fclose(fp);
    return len;
}

int read_input_three(int** input1, int** input2, int** input3, int** input4,char* filepath){
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    } 
    
    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = ( int *)malloc(len * sizeof(int));
    *input2 = ( int *)malloc(len * sizeof(int));
    *input3 = ( int *)malloc(len * sizeof(int));
    *input4 = ( int *)malloc(len * sizeof(int));

    int temp1;
    int temp2;
    int temp3;
    int temp4;

    while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
        (*input1)[counter] = temp1;
        (*input2)[counter] = temp2;
        (*input3)[counter] = temp3;
        (*input4)[counter] = temp4;
        counter++;
    }
    
    fclose(fp);
    return len;
    
}
__device__ int gate_solver(int gate, int prev_output, int input) {
  int result;
  if (gate == AND) result = prev_output & input;
  else if (gate == OR) result = prev_output | input;
  else if (gate == NAND) result = !(prev_output & input);
  else if (gate == NOR) result = !(prev_output | input);
  else if (gate == XOR) result = prev_output ^ input;
  else if (gate == XNOR) result = !(prev_output ^ input);
  else result = -1;
  
  return result;
}

__global__ void global_queuing_kernel(int numBlocks, int threadsPerBlock, int numCurrLevelNodes, int *currLevelNodes_h, int *nodePtrs_h, int *nodeNeighbors_h, int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h, int *numNextLevelNodes_h){
  int id=blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads(); 
  for (int i=id;  i<numCurrLevelNodes; i+=threadsPerBlock*numBlocks) {
  //Loop over all nodes in the current level 
      int node = currLevelNodes_h[i];
      //Loop over all neighbors of the node 
      for (int j=nodePtrs_h[node]; j<nodePtrs_h[node+1]; j++) {
          int neighbor = nodeNeighbors_h[j];
          //If the neighbor hasn’t been visited yet 
          if (nodeVisited_h[neighbor] == 0) {
              nodeVisited_h[neighbor] = 1;
              //Update node output
              nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
              
              //Add it to the global queue
              int position = atomicAdd(numNextLevelNodes_h, 1);
              atomicExch(&nextLevelNodes_h[position], neighbor);

          }
      }
  } 
   __syncthreads(); 
}
int main(int argc, char *argv[]) {
  FILE * output1, * output2;

  // Variables
  int i=0;
  int numNodePtrs;
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors_h;
  int *currLevelNodes_h;
  int numCurrLevelNodes;
  int *numNextLevelNodes_h=&i;
  int *nodeGate_h;
  int *nodeInput_h;
  int *nodeOutput_h;

  int *dev_numNextLevelNodes_h, *dev_nodePtrs_h, *dev_nodeNeighbors_h, *dev_nodeVisited_h, *dev_nodeGate_h, *dev_nodeInput_h, *dev_nodeOutput_h, *dev_currLevelNodes_h, *dev_nextLevelNodes_h;
  numNodePtrs = read_input_one_two_four(&nodePtrs_h, argv[1]);
  numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, argv[2]);
  numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h,argv[3]);
  numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, argv[4]);
  
  //output
  int *nextLevelNodes_h = new int[numTotalNeighbors_h];

  cudaMalloc((void**)&dev_numNextLevelNodes_h, sizeof(int));
  cudaMalloc((void**)&dev_nodePtrs_h, numNodePtrs*sizeof(int));
  cudaMalloc((void**)&dev_nodeNeighbors_h, numTotalNeighbors_h*sizeof(int));
  cudaMalloc((void**)&dev_nodeVisited_h, numNodes*sizeof(int));
  cudaMalloc((void**)&dev_nodeGate_h, numNodes*sizeof(int));
  cudaMalloc((void**)&dev_nodeInput_h, numNodes*sizeof(int));
  cudaMalloc((void**)&dev_nodeOutput_h, numNodes*sizeof(int));
  cudaMalloc((void**)&dev_currLevelNodes_h, numCurrLevelNodes*sizeof(int));
  cudaMalloc((void**)&dev_nextLevelNodes_h, numTotalNeighbors_h*sizeof(int));
  
  cudaMemcpy(dev_nodePtrs_h, nodePtrs_h, numNodePtrs*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nodeNeighbors_h, nodeNeighbors_h, numTotalNeighbors_h*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nodeVisited_h, nodeVisited_h, numNodes*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nodeGate_h, nodeGate_h, numNodes*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nodeInput_h, nodeInput_h, numNodes*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nodeOutput_h, nodeOutput_h, numNodes*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_currLevelNodes_h, currLevelNodes_h, numCurrLevelNodes*sizeof(int), cudaMemcpyHostToDevice);

  int numBlocks=10;
  int threadsPerBlock=32;
  GpuTimer timer;
  timer.Start();
  global_queuing_kernel<<<numBlocks,threadsPerBlock>>>(numBlocks, threadsPerBlock, numCurrLevelNodes, dev_currLevelNodes_h,dev_nodePtrs_h, dev_nodeNeighbors_h, dev_nodeVisited_h, dev_nodeOutput_h, dev_nodeGate_h, dev_nodeInput_h, dev_nextLevelNodes_h, dev_numNextLevelNodes_h);
  timer.Stop();

  cudaMemcpy(numNextLevelNodes_h, dev_numNextLevelNodes_h, sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(nextLevelNodes_h, dev_nextLevelNodes_h, numTotalNeighbors_h*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(nodeOutput_h, dev_nodeOutput_h, numNodes*sizeof(int),cudaMemcpyDeviceToHost);

  // Create output files
  output1 = fopen(argv[5], "w");
  output2 = fopen(argv[6], "w");
  fprintf(output1, "%d\n", numNodes);
  for (int i=0; i<numNodes; i++) {
      fprintf(output1, "%d\n", nodeOutput_h[i]);
  }

  fprintf(output2, "%d\n", *numNextLevelNodes_h);
  for (int i=0; i<(*numNextLevelNodes_h); i++) {
       fprintf(output2, "%d\n", nextLevelNodes_h[i]);
  }
  printf("Time elapsed for execution: %f\n", timer.Elapsed());
  
  // Close and free pointers
  fclose(output1);
  fclose(output2);
  cudaFree(nextLevelNodes_h);
  cudaFree(nodePtrs_h);
  cudaFree(nodeNeighbors_h);
  cudaFree(nodeVisited_h);
  cudaFree(currLevelNodes_h);
  cudaFree(nodeGate_h);
  cudaFree(nodeInput_h);
  cudaFree(nodeOutput_h);

  return 0;
}