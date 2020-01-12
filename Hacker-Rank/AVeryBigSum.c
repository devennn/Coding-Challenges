//input
//5 , 1000000001 1000000002 1000000003 1000000004 1000000005

//Output
//5000000015

#include <stdio.h>
#include <string.h>

int main(){

  int n;
  long long int *input;
  long long int sum = 0;

  scanf("%d",&n);
  input = malloc(n * sizeof(int)); //Using malloc to dynamically provide array size

  for(int j = 0; j < n ; j++){
    scanf("%lld", &input[j]);
  }
  for(int i = 0 ; i<n ; i++){
    sum = sum + input[i];
  }
  printf("%lld",sum);
  return sum;
}
