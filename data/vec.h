#ifndef VEC
#define VEC

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
using namespace std;

class Vec {
	float v[3];
public:
	Vec(float v0, float v1, float v2) { v[0] = v0;  v[1] = v1; v[2] = v2; }
	Vec() { v[0] = 0; v[1] = 0; v[2] = 0; }
	float operator[](int i) const { return v[i]; }
	float& operator[](int i) { return v[i]; }
	Vec operator-(const Vec& other) const { return Vec(v[0] - other[0], v[1] - other[1], v[2] - other[2]); }
	Vec operator+(const Vec& other) const { return Vec(v[0] + other[0], v[1] + other[1], v[2] + other[2]); }
	Vec& operator=(const Vec& other) {
		v[0] = other[0];
		v[1] = other[1];
		v[2] = other[2];
		return *this;
	}
	Vec& operator+=(const Vec& other) {
		v[0] += other[0];
		v[1] += other[1];
		v[2] += other[2];
		return *this;
	}
	Vec& operator/=(int i){
		v[0] /= i;
		v[1] /= i;
		v[2] /= i;
		return *this;
	}
	Vec operator* (float k) const { return Vec(k*v[0], k*v[1], k*v[2]); }
	Vec operator/ (float k) const { return Vec(v[0]/k, v[1]/k, v[2]/k); }
	Vec cross(const Vec& other) const {
		return Vec(v[1] * other.v[2] - v[2] * other.v[1],
			v[2] * other.v[0] - v[0] * other.v[2],
			v[0] * other.v[1] - v[1] * other.v[0]);
	}
	float len2() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }
	float len() const { return sqrt(len2()); }
};

inline istream& operator>>(istream& in, Vec& t) {
	in >> t[0] >> t[1] >> t[2];
	return in;
}

inline ostream& operator<<(ostream& out, const Vec& t) {
	for (int i = 0; i < 3; ++i)
		out << int(255.99*t[i]) << " ";
	return out;
}

inline Vec operator*(float k, const Vec& v) {
	return Vec(v[0] * k, v[1] * k, v[2] * k);
}

inline void normalize(Vec& vec) {
	float invLength = 1 / sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] *= invLength;
	vec[1] *= invLength;
	vec[2] *= invLength;
}

inline void rotate_x(Vec& v, float angle) {
	float y = v[1], z = v[2];
	v[1] = cos(angle)*y - sin(angle)*z;
	v[2] = sin(angle)*y + cos(angle)*z;
}

inline void rotate_y(Vec&v, float angle) {
	float x = v[0], z = v[2];
	v[0] = cos(angle)*x - sin(angle)*z;
	v[2] = sin(angle)*x + cos(angle)*z;
}

inline void rotate_z(Vec&v, float angle) {
	float x = v[0], y = v[1];
	v[0] = cos(angle)*x - sin(angle)*y;
	v[1] = sin(angle)*x + cos(angle)*y;
}
#endif // !VEC
