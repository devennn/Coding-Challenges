/*
Input: 07:05:45PM, 12:40:22AM

Output: 19:05:45, 00:40:22
*/

#include <stdio.h>
#include <string.h>

int main() {

  int hour,min,sec;
  char string[10];
  char time[5];

  scanf("%s",&string);
  //sscanf is used to separate string components
  sscanf(string , "%d:%d:%d%s" , &hour,&min,&sec,&time);
  printf("Hour: %d, Min: %d, Sec: %d, Time: %s" , hour,min,sec,time);

//strcmp is used to compare string. Value will be 0 if they are same
  if (strcmp(time, "PM") == 0) {
    if(hour == 12){
      printf("\n%.2d:%.2d:%.2d",hour,min,sec);
    }
    else{
      hour = 12 + hour;
      printf("\n%.2d:%.2d:%.2d",hour,min,sec);
    }
  }
  else if (strcmp(time, "AM") == 0){
    if(hour == 12){
      hour = 12 - hour;
      printf("\n%.2d:%.2d:%.2d",hour,min,sec);
    }
    else{
      printf("\n%.2d:%.2d:%.2d",hour,min,sec);
    }
  }

  return 0;
}
