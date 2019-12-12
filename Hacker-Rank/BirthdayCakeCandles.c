/*
Input
4
3 1 2 3

Output
2

Search for bigest number
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){

  int count,i,j,k = 0,candle = 0,max=0;
  int *input;

  scanf("%d",&count);
  //printf("N: %d\n",count);
  input = malloc(count * sizeof (int));

  for(i = 0 ; i < count ; i++){
    scanf("%d",&input[i]);
    if( input[i] > max){
      candle = 1 ;
      max = input[i];
    }
    else if (input[i] == max) {
      candle += 1 ;
    }
    else{
      candle = candle;
    }
  }
  printf("%d",candle);
}
