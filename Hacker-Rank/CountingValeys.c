#include <stdio.h>
#include <string.h>

int level_calculator(int level, char a)
{
    if(a == 'U') {
        level += 1;
    } else if(a == 'D') {
        level -= 1;
    }
    return level;
}

int main(int argc, char const *argv[])
{
    int n, level = 0, prev = 0, valley = 0;
    char a;
    scanf("%d", &n);

    for(int i = 0; i < n; ++i) {
        scanf(" %c", &a);
        level = level_calculator(level, a);
        if(level < 0 && prev == 0) {
            valley += 1;
        }
        prev = level;
    }

    printf("%d\n", valley);

    return 0;
}
