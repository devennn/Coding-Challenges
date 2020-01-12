/*
5
1 2 1 3 2
3 2

What sum of 2 numbers that will produce 3?
*/

#include <stdio.h>
#include <stdlib.h>

int main() {

  int i,n,*input,testsum,goal,req;

//Requirements
  scanf("%d",&n);
  input = malloc(n * sizeof(int));
  for(i=0 ; i<n ; i++){
    scanf("%d",&input[i]);
  }
  scanf("%d %d",&goal,&req);

//Process


  return 0;
}
