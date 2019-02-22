#ifndef HEAD
#define HEAD

#include "vec.h"
#include <vector>
#include <string>
#include <fstream>
using namespace std;

class Head
{
	vector<Vec> vertices;
public:
	vector<float> depth;
	vector<bool> mask;
	Head(const char* filename);
	void projection(int size, float scale, const Vec& center);
	void visualization(int size);
};

#endif // !HEAD
