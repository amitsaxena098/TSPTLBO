#include<bits/stdc++.h>

using namespace std;

int main(int argc, char **argv)
{
	FILE *input, *output;

	printf("%s %s\n", argv[1], argv[2]);
	input = fopen(argv[1], "r");
	output = fopen(argv[2], "w+");

	if(input == NULL)
		printf("error: failed to open input file\n");

	if(output == NULL)
		printf("error: failed to open output file\n");

	vector<pair<float, float> > points;

	int numberOfCities;
	fscanf(input, "%d", &numberOfCities);
	for(int i = 0; i < numberOfCities; i++)
	{
		float x , y;
		fscanf(input, "%f", &x);
		fscanf(input, "%f", &y);

		points.push_back(make_pair(x, y));
	}

	int dis[numberOfCities][numberOfCities] = {0};
	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
		{
			int ed;
			int x = (points[j].first - points[i].first)*(points[j].first - points[i].first);
			int y = (points[j].second - points[i].second)*(points[j].second - points[i].second);
			ed = sqrt(x+y);
			dis[i][j] = ed;
		}
	}

	for(int i = 0; i < numberOfCities; i++)
	{
		for(int j = 0; j < numberOfCities; j++)
		{
			fprintf(output, "%d ", dis[i][j]);
		}
		fprintf(output, "\n");
	}
	return 0;

}
