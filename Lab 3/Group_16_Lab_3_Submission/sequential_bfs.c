#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    //printf("Read input\n");
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
    //printf("Read input 3\n");
    fclose(fp);
    return len;
    
}
int gate_solver(int gate, int prev_output, int input) {
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

int main(int argc, char *argv[]) {
  FILE * output1, * output2;

  // Variables
  int numNodePtrs;
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors_h;
  int *currLevelNodes_h;
  int numCurrLevelNodes;
  int numNextLevelNodes_h;
  int *nodeGate_h;
  int *nodeInput_h;
  int *nodeOutput_h;

  //output
  int *nextLevelNodes_h;

  numNodePtrs = read_input_one_two_four(&nodePtrs_h, argv[1]);
  numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, argv[2]);
  numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h,argv[3]);
  numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, argv[4]);
  output1 = fopen(argv[5], "w");
  output2 = fopen(argv[6], "w");
  nextLevelNodes_h = (int *)malloc(numTotalNeighbors_h*sizeof(int)); // allocating too much space ...
  
  //printf("numNodePtrs: %d\n", numNodePtrs);
  //printf("numCurrLevelNodes: %d\n", numCurrLevelNodes);
  //printf("numTotalNeighbors_h: %d\n", numTotalNeighbors_h);
  //printf("numNodes: %d\n", numNodes);

  clock_t start= clock();
  for (int i=0; i<numCurrLevelNodes; i++) {
      int node = currLevelNodes_h[i];
      for (int j=nodePtrs_h[node]; j<nodePtrs_h[node+1]; j++) {
          int neighbor = nodeNeighbors_h[j];
          if (nodeVisited_h[neighbor] == 0) {
              nodeVisited_h[neighbor] = 1;
              nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
              
              nextLevelNodes_h[numNextLevelNodes_h]=neighbor;
              ++(numNextLevelNodes_h);
          }
      }
  }
  clock_t end = clock();
  printf("Time elapsed: %lf\n", (double)(end-start)/1000);

  // Create output files
  fprintf(output1, "%d\n", numNodes);
  for (int i=0; i<numNodes; i++) {
      fprintf(output1, "%d\n", nodeOutput_h[i]);
  }

  fprintf(output2, "%d\n", numNextLevelNodes_h);
  for (int i=0; i<numNextLevelNodes_h; i++) {
       fprintf(output2, "%d\n", nextLevelNodes_h[i]);
  }

  // Close and free pointers
  fclose(output1);
  fclose(output2);
  free(nextLevelNodes_h);
  free(nodePtrs_h);
  free(nodeNeighbors_h);
  free(nodeVisited_h);
  free(currLevelNodes_h);
  free(nodeGate_h);
  free(nodeInput_h);
  free(nodeOutput_h);

  return 0;
}