#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
  int m,n,flag,card,x,i,j,*a,*b;
  scanf("%d %d",&n,&m);

  a = malloc(n * sizeof(int));
  b = malloc(m * sizeof(int));

  for(int i = 0; i < n; i++){
    scanf("%d",&a[i]);
  }
  for(int i = 0; i < m; i++)
  {
    scanf("%d",&b[i]);
  }

  card=0;

  for (x=1 ; x<=100 ; x++){
    flag=1;

    for (i=0;i<n;++i){
      if ((x % a[i]) != 0){
        flag=0;
      }
    }

    for (j=0;j<m;++j){
      if ((b[j] % x) != 0){
        flag=0;
      }
    }

    if (flag == 1){
      card++ ;
    }
  }

  printf("%d\n",card);
  return 0;
}
