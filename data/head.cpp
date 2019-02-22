#include "head.h"


Head::Head(const char* filename) {
	ifstream f(filename);
	if (f.is_open()) {
		string tmp;

		// skip header
		while ((f >> tmp) && tmp != "v");
		// read vertices position
		double x, y, z;
		do {
			f >> x >> y >> z;
			vertices.push_back(Vec(x, y, z));
		} while ((f >> tmp) && tmp == "v");
	}
}


void Head::projection(int size, float scale, const Vec& center) {
	depth.clear();
	mask.clear();

	int i, j, count = size * size;

	depth.resize(count);
	mask.resize(count);

	for (i = 0; i < count; ++i) {
		mask[i] = false;
		depth[i] = FLT_MIN;
	}

	int sx, sy, idx;

#pragma omp parallel for
	for (i = 0; i < vertices.size(); ++i) {
		sx = (vertices[i][0] - center[0]) * scale + size / 2;
		sy = (center[1] - vertices[i][1]) * scale + size / 2;
		if (sx >= size || sy >= size) continue;
		idx = sy * size + sx;
		if (mask[idx] && depth[idx] > vertices[i][2]) continue;
		mask[idx] = true;
		depth[idx] = vertices[i][2];
	}
}


void Head::visualization(int size) {
	ofstream f("head_model.ppm");
	float alpha;
	Vec color;
	if (f.is_open()) {
		f << "P3\n" << size << " " << size << " 255\n";
		double u, v;
		for (int i = 0; i < size*size; ++i) {
			if (mask[i])
				color = Vec(1, 1, 0);
			else color = Vec(0, 0, 0);
			f << color << endl;
		}
	}
	f.close();
}