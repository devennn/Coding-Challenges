#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

int main(int argc, char const *argv[]) {

        int numOfHurdles, maxJump, i, temp, tallest = 0;
        scanf("%d %d", &numOfHurdles, &maxJump);
        for(i = 0; i < numOfHurdles; i++) {
                scanf("%d", &temp);
                if(temp > tallest) {
                        tallest = temp;
                }
        }
        if(maxJump < tallest) {
                printf("%d\n", tallest - maxJump);
        } else {
                printf("0\n");
        }

        return 0;
}
