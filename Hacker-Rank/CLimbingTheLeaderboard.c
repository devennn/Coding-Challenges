#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

int main(int argc, char const *argv[]) {

    int n, i, j, t, score_index, k;
    scanf("%d", &n);

    int *scores = malloc(sizeof(int) * n);
    scanf("%d", &scores[0]);

    for (score_index = 1, k = 1; k < n; k++ ) {
       scanf("%d",&t);
       if(t != scores[score_index - 1]) {
           scores[score_index] = t; //only store value which is not the same
           score_index++;
       }
     }

    n = score_index;
    int m, rank;
    j = n - 1; //will store last index or the last rank
    scanf("%d", &m);
    int *alice = malloc(sizeof(int) * m);

    for (int alice_index = 0; alice_index < m; alice_index++) {
       scanf("%d", &alice[alice_index]);
    }

    for(i = 0; i < m; i++) {
        while(j >= 0 && alice[i] > scores[j]) {
           j--;
        }
        if(j == -1) {
            rank = 1;
        } else if(alice[i] == scores[j]) {
            rank = j + 1;
        } else if(alice[i] < scores[j]) {
            rank = j + 2;
        }
        printf("%d\n",rank);
    }

    return 0;
}
