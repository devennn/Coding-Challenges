#include <stdio.h>
#include <stdbool.h>

int find_number(int n, int in[][2])
{
    int j;
    bool done = false;
    static int index = 0;
    for(j = 0; j < index; j++) {
        if(n == in[j][0]) {
            in[j][1] += 1;
            done = true;
            break;
        }
    }
    if(done == false) {
        in[j][0] = n;
        in[j][1] = 1;
        ++index;
    }
    return index;
}

int rank_highest(int index, int in[][2])
{
    int type, typeFreq = -1;

    for(int i = 0; i < index; ++i) {
        if((in[i][1] > typeFreq) ||
            ((in[i][1] == typeFreq) && (in[i][0] < type))) {
            typeFreq = in[i][1];
            type = in[i][0];
        }
    }
    return type;
}

int main(int argc, char const *argv[])
{

    int num;
    scanf("%d", &num);
    int in[num][2];
    int n;

    for(int i = 0; i < num; i++) {
        in[i][0] = -1;
        in[i][1] = 0;
    }

    int index;
    for(int i = 0; i < num; i++) {
        scanf("%d", &n);
        index = find_number(n, in);
    }

    // printf("\nEnding\n");
    // for(int i = 0; i < index; i++) {
    //     printf("%d , %d\n", in[i][0], in[i][1]);
    // }

    printf("%d\n", rank_highest(index, in));

    return 0;
}
