/*
Input:
s, t
a, b
m n
a0 a1 am
b0 b1 bn

7 11
5 15
3 2
-2 2 1
5 -6

Output
How many fall in range of s and t
1
1

Output:

*/
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define bool int
#define true 1
#define false 0

//Define how istherange works
bool isInRange(int lowerLimit, int upperLimit, int no)
{
    return (lowerLimit <= no && no <= upperLimit);
}

int main(){
  int s,t,a,b,m,n,i,hita = 0,hitb = 0,wherea,whereb;
  int *apple,*orrange;

  scanf("%d %d %d %d %d %d",&s,&t,&a,&b,&m,&n);

  // Set size
  apple = malloc(m * sizeof(int));
  orrange = malloc(n * sizeof(int));

  for(i = 0 ; i < m ; i++){
    scanf("%d",&apple[i]);
    wherea = a + apple[i];
    if (isInRange(s,t,wherea) == 1){
      hita += 1;
    }
  }
  for(i = 0 ; i < n ; i++){
    scanf("%d",&orrange[i]);
    whereb = b + orrange[i];
    if (isInRange(s,t,whereb) == 1){
      hitb += 1;
    }
  }
  printf("%d\n%d",hita,hitb);
}
