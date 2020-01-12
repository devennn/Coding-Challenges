/*
Input
1 2 3 4 5
//942381765 627450398 954173620 583762094 236817490

Output
10 14

Explain
Min Sum of all: 10
Max Sum of all: 14
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){

  long long int maximum = -9999999999,minimum = 9999999999,summax,summin,sumall = 0;
  int i,n=5;
  int *input;

  //printf("N: %d\n",n);
  input = malloc(n * sizeof(int));

  for (int i = 0; i < n; i++) {
    scanf("%lld",&input[i]);
  }
  for (i = 0 ;i < n ;i++) {
    if (input[i] > maximum) {
      maximum = input[i];
    }
    if(input[i] < minimum){
      minimum = input[i];
    }
    else{
      maximum = maximum;
      minimum = minimum;
    }
  }
  for (i = 0; i < n; i++) {
      sumall += input[i];
  }
  summin = sumall - maximum;
  summax = sumall - minimum;
  printf("%lld %lld",summin,summax);
  return 0;
}
