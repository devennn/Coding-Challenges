#include <stdio.h>

int main(int argc, char *argv[])
{
	int num, notEat, toPay = 0, notEatPrice = 0;
	int buf;
	scanf("%d %d", &num, &notEat);

	for(int i = 0; i < num; ++i) {
		scanf("%d", &buf);
		if(i != notEat) {
			toPay += buf;
		} else {
			notEatPrice = buf;
		}
	}

	toPay = toPay / 2;
	int charged;
	scanf("%d", &charged);
	
	if((charged - toPay) <= 0) {
		puts("Bon Appetit");
	} else {
		printf("%d\n", charged - toPay);
	}
}
