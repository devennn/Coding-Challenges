/*
9
10 5 20 20 4 5 2 25 1
*/

#include <stdio.h>
#include <stdlib.h>

int main() {

  int game,*score,i,high,low,breakhigh = 0,breaklow = 0;

  scanf("%d",&game);
  score = malloc(game * sizeof(int));

  for (i=0 ; i<game ; i++){
    scanf("%d",&score[i]);
    high = score[0];
    low = score[0];
  }

  for (i=1 ; i<game ; i++){
    if(score[i] > high){
      high = score[i];
      breakhigh += 1;
    }
    else if(score[i] < low){
      low = score[i];
      breaklow += 1;
    }
  }

  printf("%d %d",breakhigh,breaklow);
}
