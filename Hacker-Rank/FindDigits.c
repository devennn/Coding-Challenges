#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int find_divisor(char *str)
{
	int len = strlen(str);
	int num = atol(str);
	int div = 0;
	int buf;
	for(int i = 0; i < len; ++i) {
		buf = str[i] - '0';
		if(buf == 0) {
			continue;
		}
		if((num % buf) == 0) {
			div += 1;
		}
	}
	return div;
}


int main(int argc, char *argv[])
{
	int n;
	char **num;
	scanf("%d", &n);
	num = (char **)malloc(n * sizeof(char *));
	for(int i = 0; i < n; ++i) {
		num[i] = (char *)malloc(1024 * sizeof(char));
		memset(num[i], '\0', 1024);}

	// Get input
	int div;
	for(int i = 0; i < n; ++i) {
		scanf("%s", num[i]);
		div = find_divisor(num[i]);
		printf("%d\n", div);
	}

}
