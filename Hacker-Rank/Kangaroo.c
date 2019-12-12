/*
Input: 14 4 98 2 , 4523 8092 9419 8076
Output: YES , YES
*/
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(){
  long long int k1,k2,j1,j2,end1,end2,count = 1000000;
  int status = 1,i = 0;
  scanf("%lld %lld %lld %lld",&k1,&j1,&k2,&j2);
  do {
    end1 = k1 + (i*j1);
    end2 = k2 + (i*j2);
    i++;
    if (end1 == end2){
      printf("YES");
      status = 0;
    }
  } while((status == 1) && (i < count));
  if(end1 != end2){
    printf("NO");
  }
  return 0;
}
