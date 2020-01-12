/*
Input: 10,9,15
1 1 3 1 2 1 3 3 3 3
10 20 20 10 10 30 50 10 20
6 5 2 3 5 2 2 1 1 5 1 3 3 3 5

Ouput: 4,3,6
*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int main() {

   int n,i,t;
    scanf("%d",&n);
    int c[101];
    int sizeofdata = sizeof(c);
    //printf("%d",sizeofdata); //Checking the data type size
    memset(c,0,sizeofdata); //Replacing char of 0
    for(i=0;i<n;i++)
        {
        scanf("%d",&t);
        c[t]++;
    }
    long int ans=0;
    for(i=1;i<=100;i++)
        {
       ans+=(c[i]/2);
    }
    printf("%ld",ans);
    return 0;
}
