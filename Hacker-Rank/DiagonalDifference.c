//input
/*
array row[i] and column[j] = 3 //3by3 matrix
11 2 4
4 5 6
10 8 -12
*/

//Output
//Difference ofprimary and secondary diagonals, 19 - 4 = 15
//Primary Diagonals: 11 + 5 - 12 = 4
//Secondary Diagonals: 4 + 5 + 10 = 19

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(){


  int n,i,j,d;
  scanf("%d",&n);
  printf("N: %d\n", n);

  int sizerow = n;
  int sizecol = n;
  int *matrix = (int *)malloc(sizerow * sizecol * sizeof(int));
  int sum=0,row,col,a,b,sum1=0,k=0,m=2; //index start at 0


  for(i = 0 ; i<sizerow ; i++){
    for (j = 0; j < sizecol; j++) {
      scanf("%d",&d);
      *(matrix + i*sizecol + j) = d;
    }
  }
///Diagonal Process
for(i = 0 ; i<sizerow ; i++){
    a = *(matrix + i*sizecol + k); //k is to determine the column position
    sum = sum + a;
  k++;
}
for(i = 0 ; i<sizerow ; i++){
    b = *(matrix + i*sizecol + m); //m is to determine the column position
    sum1 = sum1 + b;
  m--;
}
  printf("Sum: %d",sum);
  printf("\nSum1: %d",sum1);
  int total = sum - sum1 ;
  printf("\nDifference: %d",total);

  int check = 0;
  i = 0;
do {
    for (j = 0; j < sizecol; j++){
      printf("%d ", *(matrix + i*sizecol + j));
    }
    printf("\n");
    i++;
    if ((i == 2) && (j == 2)){
      check = 1;
    }
} while(check == 0);
  return 0;
}
