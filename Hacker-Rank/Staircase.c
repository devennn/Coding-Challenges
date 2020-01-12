//Input
//6

//Output
/*
     #
    ##
   ###
  ####
 #####
######
*/

#include <stdio.h>
#include <string.h>

#define MAXROW 1000
#define MAXCOL 1000

int main(){

  int n,i,j,k,iterate;
  char matrix[MAXROW][MAXCOL];

  scanf("%d",&n);
  printf("N: %d\n",n);

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      matrix[i][j] = ' ';
    }
  }
  k = n;
  for (i = (n-1); i > -1 ; i--) {//Start from bottom row
    iterate = (n - k) - 1; //Column Update -1 every time triggered
    for (j = (n-1); j > iterate; j--) {
      matrix[i][j] = '#';
    }
  k--;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%c",matrix[i][j]);
    }
    printf("\n");
  }

  return 0;
}
