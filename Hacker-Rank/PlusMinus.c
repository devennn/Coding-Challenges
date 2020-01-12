//Input
//6 numbers in array
//-4 3 -9 0 4 1

//Output
//0.500000,0.333333,0.166667
//POsitive number: 3/6
//Negative number: 2/6
//Zeros: 1/6

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
  int *input,i,n,p = 0,z = 0,m = 0;
  float d1,d2,d3;

  scanf("%d",&n);
  printf("N: d\n",&n);
  input = malloc(n * sizeof(int));

  for (i = 0; i < n; i++) {
    scanf("%d",&input[i]);
  }

  //Check number
  for (i = 0; i < n; i++) {
    if(input[i] > 0){
      p += 1;
    }
    else if(input[i] < 0){
      m += 1;
    }
    else if(input[i] == 0){
      z += 1;
    }
    else{
      printf("\nNot a number!\n");
    }
  }

  //Calculate fraction decimals
  //Use (float) at denominator to divide
  d1 = p/(float)n;
  d2 = m/(float)n;
  d3 = z/(float)n;

  for (i = 0; i < n; i++) {
    printf("%d ",input[i]);
  }
  printf("\nP: %d M: %d Z: %d",p,m,z);
  printf("\nP: %.6f M: %.6f Z: %.6f",d1,d2,d3); //There will be 6 number after .
  return 0;
}
