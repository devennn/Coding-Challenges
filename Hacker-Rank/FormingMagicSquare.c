/*
5 3 4
1 5 8
6 4 2
*/

#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int *num;
int cost[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int ans[8][9] = { {8, 1, 6, 3, 5, 7, 4, 9, 2},
                {6, 1, 8, 7, 5, 3, 2, 9, 4},
                {4, 9, 2, 3, 5, 7, 8, 1, 6},
                {2, 9, 4, 7, 5, 3, 6, 1, 8},
                {8, 3, 4, 1, 5, 9, 6, 7, 2},
                {4, 3, 8, 9, 5, 1, 2, 7, 6},
                {6, 7, 2, 1, 5, 9, 8, 3, 4},
                {2, 7, 6, 9, 5, 1, 4, 3, 8} };

void c_magic(void)
{
        int diff;
        for(int i = 0; i < 9; i++) {
                for(int j = 0; j < 9; j++) {
                        diff = abs(ans[i][j] - num[j]);
                        cost[i] += diff;
                }
        }
}

int main()
{
        int i, ans;

        num = malloc(9 * sizeof(int));

        for(i = 0; i < 9; i++){
                scanf("%d", &num[i]);
        }
        c_magic();
        ans = 100;
        for(i = 0; i < 9; i++) {
                if(cost[i] < ans) {
                        ans = cost[i];
                }
        }
        printf("%d\n", ans);

	return 0;
}
