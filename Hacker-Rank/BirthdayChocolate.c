/*
34
4 5 4 5 1 2 1 4 3 2 4 4 3 5 2 2 5 4 3 2 3 5 2 1 5 2 3 1 2 3 3 1 2 5
18 6

Ans: 6
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

int main(){

  int n, *input, req[2], i, j, counter = 0, sum = 0, num_before, update = 1, k, iteration = 0;

  scanf("%d", &n);
  input = malloc(n * sizeof(int));

  for(i = 0 ; i < n ; i ++){
    scanf("%d", &input[i]);
  }

  for(i = 0 ; i < 2 ; i ++){
    scanf("%d", &req[i]);
  }

  // Number in forward is sum with the number backward
  for(i = 0 ; i < n ; i++){
    num_before = input[i];
    k = req[1] + iteration ; // Make sure k always in the range required

    if(k <= n){
      for(j = update ; j < k ; j++){
        sum = sum + input[j];
      }

      sum = sum + num_before;
      if(sum == req[0]){
        counter += 1;
        sum = 0;
      }
      else{
        sum = 0;
      }
    }

  update++;
  iteration++;
  }

printf("%d", counter);
return 0;
}
