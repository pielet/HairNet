#ifndef HAIR
#define HAIR

#include <vector>
#include <string>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <algorithm>
#include <time.h>
#include <random>
#include "Vec.h"
#include "hdf5.h"

using namespace std;

class Hair
{
	string fname, suffix;
	float angle;	// »¡¶È
	float depth_thred;
	bool recompute_curv;
	typedef vector<Vec> strand;
	vector<strand> strands, tangents;
	typedef vector<float> curvature;
	vector<curvature> curvatures;
	typedef vector<bool> strand_visible;
	vector<strand_visible> strands_visible;

	void clear() { strands.clear(); tangents.clear(); }
	size_t size() const { return strands.size(); }

	bool read_bin(const char* filename);
	void compute_tangents();
	bool is_long();
	void output_x(const float* x, const float* y, const bool* mask, int size) const;
	void output_param(int, const vector<int>&, const short*) const;
	void output_y(int grid, const vector<int>& idx, const short* mask) const;

public:
	Hair(const char* filename):angle(0), suffix(""), recompute_curv(true) {
		read_bin(filename);				// fname, strainds
		depth_thred = is_long()? -0.07 : 0;	// depth
	}
	/*
	Vec get_center() const {
		return Vec((bbox2[0] + bbox1[0]) / 2, (bbox1[1] + bbox2[1]) / 2, (bbox1[2] + bbox2[2]) / 2);
	}
	float get_longest_xyz() const {
		float max;
		if (bbox2[0] - bbox1[0] > bbox2[1] - bbox1[1])
			max = bbox2[0] - bbox1[0];
		else max = bbox2[1] - bbox1[1];
		if (max > bbox2[2] - bbox1[2]) return max;
		else return bbox2[2] - bbox1[2];
	}
	*/
	void get_interpolate_param(const Vec& center, float angle, int grid, vector<int>& idx, short*) const;
	void train_x(int size, float scale, const Vec& center); // tangents
	void train_y(int grid, const vector<int>& idx, const short* mask);
	void rotate(float angle);	// suffix, angle, strands
	void flip();				// suffix, strands
};
#endif // !HAIR
