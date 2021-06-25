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

int gate(int in1, int in2, int operation) {
  int result;

  if (operation == AND) result = in1 & in2;
  else if (operation == OR) result = in1 | in2;
  else if (operation == NAND) result = !(in1 & in2);
  else if (operation == NOR) result = !(in1 | in2);
  else if (operation == XOR) result = in1 ^ in2;
  else if (operation == XNOR) result = !(in1 ^ in2);
  else result = -1;
  
  return result;
}
int main(int argc, char * argv[]) {
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

  // Read the data in the file into an array
  char * line = NULL;
  int *bool1 = malloc(lines*sizeof(int));
  int *bool2 = malloc(lines*sizeof(int));
  int *op = malloc(lines*sizeof(int));
  size_t n;

  for (int i = 0; i < lines; i++) {
    line = NULL;
    size_t n = 0;
    getline(&line, &n, fp);
    
    bool1[i] = (int)line[0]-48;
    bool2[i] = (int)line[2]-48;
    op[i] = (int)line[4]-48;
  }
  
  // Calculate the result for each line
  clock_t start = clock();
  for (int i = 0; i < lines; i++) {
    int result = gate(bool1[i], bool2[i], op[i]);
    //printf("result: %d\n", result);
    fprintf(out, "%d\n", result);
    //printf("===============\n");
  }
  clock_t end = clock();
  printf("Time elapsed: %lf\n", (double)(end-start)/1000);

  // Close the files
  fclose(fp);
  fclose(out);
  return 0;
}