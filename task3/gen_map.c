#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char ** argv)
{ 
	int N = 512;
	srand(time(NULL));
	FILE* map = fopen("mapping.txt", "w");
	int i, x = 0, y = 0, z = 0;
	for (i = 0; i < N; i++)
	{
		if (z == 8)
		{
			z = 0;
			x++;
		}
		if (x == 8)
		{
			x = 0;
			y++;
		}
		if (y == 8)
		{
			printf("Error");
			break;
		}
		fprintf(map, "%d %d %d 0\n", x, y, z);
		z++;
	}
/*	for (i = 0; i < N; i++)
	{

		for (i = 0; i < N; i++)
		{
			x = rand() % 8;
			y = rand() % 8;
			z = rand() % 8;
			fprintf(map, "%d %d %d 0\n", x, y, z);
		}
	}
	fclose(map);
   */
    return 0;
}
