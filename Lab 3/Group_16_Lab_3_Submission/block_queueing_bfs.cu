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

__device__ int globalcounter;
__shared__ int blockcounter;
__shared__ int currentcounter;

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

__global__ void block_queuing_kernel(int round, int numRounds, int numBlocks, int threadsPerBlock, int queueSize, int numCurrLevelNodes, int *currLevelNodes_h, int *nodePtrs_h, int *nodeNeighbors_h,int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h, int *numNextLevelNodes_h) {
    extern __shared__ int queue[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    int index=round*threadsPerBlock*numBlocks+id;
    
    if (id < numBlocks*threadsPerBlock) {
        int node = currLevelNodes_h[index];

        for (int j = nodePtrs_h[node]; j < nodePtrs_h[node+1]; j++) {
            int neighbor = nodeNeighbors_h[j];

            if (nodeVisited_h[neighbor] == 0) {
                nodeVisited_h[neighbor] = 1;
                nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);

                int position = atomicAdd(&blockcounter, 1);
                if (position < queueSize) {
                    atomicAdd(&currentcounter, 1);
                    queue[position] = neighbor;
                } else {
                    int global_position = atomicAdd(&globalcounter, 1);
                    atomicExch(&nextLevelNodes_h[global_position], neighbor);
                }
            }
        }
    }

    __syncthreads();
    if ((round == numRounds)&& (threadIdx.x == 0)) {
        for (int j=0; j<currentcounter; j++) {
            int position = atomicAdd(&globalcounter, 1);
            atomicExch(&nextLevelNodes_h[position], queue[j]);
        }
    }
    numNextLevelNodes_h[0] = globalcounter;
}

int main(int argc, char *argv[]) {
    FILE * output1, * output2;
    if (argc != 10) {
        printf("Error - too few arguments.\n");
        return 1;
    }

    // Variables
    int numNodePtrs;
    int numNodes;
    int *nodePtrs_h;
    int *nodeNeighbors_h;
    int *nodeGate_h;
    int *nodeInput_h;
    int *nodeOutput_h;
    int *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;

    //output
    int *nextLevelNodes_h;
    int numNextLevelNodes_h = 0;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, argv[4]);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, argv[5]);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, argv[6]);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, argv[7]);
    nextLevelNodes_h = (int *)malloc(numTotalNeighbors_h*sizeof(int));
    
      
    int *d_numNextLevelNodes_h, *d_nodePtrs_h, *d_nodeNeighbors_h, *d_nodeVisited_h, *d_nodeGate_h, *d_nodeInput_h, *d_nodeOutput_h, *d_currLevelNodes_h, *d_nextLevelNodes_h;

    cudaMalloc((void **)&d_numNextLevelNodes_h, sizeof(int));
    cudaMalloc((void **)&d_nodePtrs_h, numNodePtrs * sizeof(int));
    cudaMalloc((void **)&d_nodeNeighbors_h, numTotalNeighbors_h * sizeof(int));
    cudaMalloc((void **)&d_nodeVisited_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeGate_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeInput_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeOutput_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_currLevelNodes_h, numCurrLevelNodes * sizeof(int));
    cudaMalloc((void **)&d_nextLevelNodes_h, numTotalNeighbors_h * sizeof(int));

    cudaMemcpy(d_nodePtrs_h, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeNeighbors_h, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeVisited_h, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeGate_h, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeInput_h, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeOutput_h, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_currLevelNodes_h, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int queueSize = atoi(argv[3]);
    int round=0;
    int numRounds = numCurrLevelNodes/(numBlocks*threadsPerBlock);

    GpuTimer timer;
    timer.Start();
    while (round <= numRounds) {
        block_queuing_kernel<<<numBlocks, threadsPerBlock,queueSize>>>(round, numRounds, numBlocks, threadsPerBlock, queueSize, numCurrLevelNodes, d_currLevelNodes_h, d_nodePtrs_h, d_nodeNeighbors_h, d_nodeVisited_h, d_nodeOutput_h, d_nodeGate_h, d_nodeInput_h, d_nextLevelNodes_h, d_numNextLevelNodes_h);
        round++;
    }
    timer.Stop();

    cudaMemcpy(&numNextLevelNodes_h, d_numNextLevelNodes_h, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextLevelNodes_h, d_nextLevelNodes_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeOutput_h, d_nodeOutput_h, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    output1 = fopen(argv[8], "w");
    output2 = fopen(argv[9], "w");
    fprintf(output1, "%d\n", numNodes);
    for (int i=0; i<numNodes; i++) {
        fprintf(output1, "%d\n", nodeOutput_h[i]);
    }

    fprintf(output2, "%d\n", numNextLevelNodes_h);
    for (int i=0; i<numNextLevelNodes_h; i++) {
        fprintf(output2, "%d\n", nextLevelNodes_h[i]);
    }
    printf("Time elapsed for execution: %f\n", timer.Elapsed());
  
    free(nodePtrs_h);
    free(nodeNeighbors_h);
    free(nodeGate_h);
    free(nodeInput_h);
    free(nodeOutput_h);
    free(nodeVisited_h);
    free(currLevelNodes_h);
    free(nextLevelNodes_h);
    cudaFree(d_nextLevelNodes_h);
    cudaFree(d_nodePtrs_h);
    cudaFree(d_nodeNeighbors_h);
    cudaFree(d_nodeVisited_h);
    cudaFree(d_currLevelNodes_h);
    cudaFree(d_nodeGate_h);
    cudaFree(d_nodeInput_h);
    cudaFree(d_nodeOutput_h);

    return 0;
}