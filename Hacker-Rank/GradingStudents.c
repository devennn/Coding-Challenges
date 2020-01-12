/*
Input: 4
73
67
38
33

Output:
75
67
40
33
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
  int multiple5,n,count,*input,i,a;

  scanf("%d",&count);
  input = malloc(count * sizeof(int));

  for (i = 0; i < count; i++) {
    scanf("%d",&input[i]);
    multiple5=(input[i]/5+1)*5; //Change input number to nearest multiple of 5
    a = multiple5 - input[i];
    
    if ( (a < 3) && (input[i] >= 38) ){
      input[i] = multiple5;
    }
  }
  for (int i = 0; i < count; i++) {
    printf("%d\n",input[i]);
  }
  return 0;
}
