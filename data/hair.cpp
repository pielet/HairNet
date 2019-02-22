#include "hair.h"


bool Hair::read_bin(const char *filename)
{
	
	fname = filename;
	fname.erase(fname.end() - 5, fname.end());	// skip .data
	auto pos = fname.find("strands");
	if (pos != string::npos)
		fname = fname.substr(pos);
	
	FILE *f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return false;
	}

	int nstrands = 0;
	if (!fread(&nstrands, 4, 1, f)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(f);
		return false;
	}
	strands.resize(nstrands);

	for (int i = 0; i < nstrands; i++) {
		int nverts = 0;
		if (!fread(&nverts, 4, 1, f)) {
			fprintf(stderr, "Couldn't read number of vertices\n");
			fclose(f);
			return false;
		}
		strands[i].resize(nverts);

		for (int j = 0; j < nverts; j++) {
			if (!fread(&strands[i][j][0], 12, 1, f)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return false;
			}
		}
	}

	fclose(f);
	return true;
}


void Hair::compute_tangents()
{
	int ns = strands.size();
	tangents.clear();
	tangents.resize(ns);

#pragma omp parallel for
	for (int i = 0; i < ns; i++) {
		int n = strands[i].size();
		if (!n) {
			continue;
		}
		else if (n == 1) {
			tangents[i].push_back(Vec(0, 0, -1));
			continue;
		}
		tangents[i].resize(n);
		for (int j = 0; j < n - 1; j++) {
			Vec t = strands[i][j + 1] - strands[i][j];
			normalize(t);
			tangents[i][j] += t;
			tangents[i][j + 1] += t;
		}
		for (int j = 0; j < n; j++)
			normalize(tangents[i][j]);
	}
}


bool Hair::is_long() {
	float y_max = strands[0][0][1];
	float y_min = y_max;
	float tmp;

#pragma omp parallel for
	for (int i = 0; i < strands.size(); ++i) {
		for (int j = 0; j < strands[i].size(); ++j) {
			tmp = strands[i][j][1];
			if (tmp > y_max) y_max = tmp;
			else if (tmp < y_min) y_min = tmp;
		}
	}

	if (y_max - y_min > 0.4) return true;
	else return false;
}


struct scalp
{
	int uv;
	int idx;
	scalp(int uv, int i) :uv(uv), idx(i) {};
	scalp() {}
};
bool cmp(const scalp& a, const scalp& b) { return a.uv < b.uv; }

void Hair::get_interpolate_param(const Vec& center, float angle, int grid, vector<int>& selected_idx, short* root_mask) const {
	selected_idx.clear();
	angle = angle / 180 * M_PI;
	vector<Vec> roots;
	int i, j, num = strands.size();
	roots.resize(num);
	for (i = 0; i < num; ++i)
		roots[i] = strands[i][0];
	
	// 计算所有发根的uv坐标
	vector<float> u, v;
	u.resize(num); v.resize(num);
	Vec on_scalp;
#pragma omp parallel for
	for (i = 0; i < num; ++i) {
		on_scalp = roots[i] - center;
		normalize(on_scalp);
		rotate_x(on_scalp, angle);
		u[i] = acos(on_scalp[0] / sqrt(on_scalp[0] * on_scalp[0] + (on_scalp[1] + 1)*(on_scalp[1] + 1)));
		v[i] = acos(on_scalp[2] / sqrt(on_scalp[2] * on_scalp[2] + (on_scalp[1] + 1)*(on_scalp[1] + 1)));
	}
	// 求最值
	float u_max = *max_element(u.begin(), u.end());
	float u_min = *min_element(u.begin(), u.end());
	float v_max = *max_element(v.begin(), v.end());
	float v_min = *min_element(v.begin(), v.end());
	// 变成格点
	vector<scalp> scalp_roots;
	scalp_roots.resize(num);
	int int_u, int_v;
#pragma omp parallel for
	for (i = 0; i < num; ++i) {
		int_u = (u[i] - u_min) / (u_max - u_min) * grid;
		int_v = (v[i] - v_min) / (v_max - v_min) * grid;
		if (int_u == grid) int_u = grid - 1;
		if (int_v == grid) int_v = grid - 1;
		scalp_roots[i] = scalp(int_u*grid + int_v, i);
	}
	// sort according to uv
	sort(scalp_roots.begin(), scalp_roots.end(), cmp);

	int pos = 0, closest_idx;
	double center_u, center_v, dis, closest_dis;
	vector<int> idx;
#pragma omp parallel for
	for (i = 0; i < grid*grid; ++i) {
		idx.clear();
		while (pos < strands.size() && scalp_roots[pos].uv == i) {	// 落在第i个方格里的头发
			idx.push_back(scalp_roots[pos++].idx);
		}
		// 跳过没有头发的情况（可能并没有这种情况）（还是有的）
		if (idx.empty()) continue;
		root_mask[i] = 1;
		// 选择离中心最近的发根作为代表
		center_u = (i / grid + 0.5) / grid * (u_max - u_min) + u_min;
		center_v = (i % grid + 0.5) / grid * (v_max - v_min) + v_min;
		closest_dis = 2;
		for (j = 0; j < idx.size(); ++j) {
			dis = (center_u - u[idx[j]]) * (center_u - u[idx[j]]) +
				(center_v - v[idx[j]]) * (center_v - v[idx[j]]);
			if (dis < closest_dis) {
				closest_dis = dis;
				closest_idx = j;
			}
		}
		selected_idx.push_back(idx[closest_dis]);
	}

	output_param(grid, selected_idx, root_mask);
}

void Hair::output_param(int grid, const vector<int>& idx, const short* mask) const {
	hid_t file_id = H5Fcreate("root_param.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	herr_t status;

	// mask
	unsigned rank = 2;
	hsize_t dims[2];
	dims[0] = dims[1] = grid;
	hid_t msk_space = H5Screate_simple(rank, dims, NULL);
	hid_t msk = H5Dcreate(file_id, "/mask", H5T_NATIVE_SHORT, msk_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(msk, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask);
	status = H5Dclose(msk);
	status = H5Sclose(msk_space);

	// root position
	float* flat_pos = new float[idx.size() * 3];
#pragma omp parallel for
	for (int i = 0; i < idx.size(); ++i) {
		flat_pos[3 * i] = strands[idx[i]][0][0];
		flat_pos[3 * i + 1] = strands[idx[i]][0][1];
		flat_pos[3 * i + 2] = strands[idx[i]][0][2];
	}
	dims[0] = idx.size();
	dims[1] = 3;
	hid_t pos_space = H5Screate_simple(rank, dims, NULL);
	hid_t pos = H5Dcreate(file_id, "/pos", H5T_NATIVE_FLOAT, pos_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(pos, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_pos);
	status = H5Dclose(pos);
	status = H5Sclose(pos_space);

	status = H5Fclose(file_id);
	delete[] flat_pos;
}


void Hair::train_x(int size, float scale, const Vec& center) {
	compute_tangents();
	strands_visible.clear();
	strands_visible.resize(strands.size());
	int i, j, count = size * size;

	float* x = new float[count];
	float* y = new float[count];
	float* depth = new float[count];
	bool* mask = new bool[count];

#pragma omp parallel for
	for (i = 0; i < count; ++i) {
		mask[i] = false;
		depth[i] = FLT_MIN;
	}

	int sx, sy, idx;

#pragma omp parallel for
	for (j = 0; j < strands.size(); ++j) {
		strands_visible[j].resize(strands[j].size());
		for (i = 0; i < strands[j].size(); ++i) {
			if (strands[j][i][2] < depth_thred) {	// 去掉后一半
				strands_visible[j][i] = false;
				continue;
			}
			sx = (strands[j][i][0] - center[0]) * scale + size / 2;
			sy = (center[1] - strands[j][i][1]) * scale + size / 2;
			if (sx >= size || sy >= size || sx < 0 || sy < 0) {
				strands_visible[j][i] = false;
				continue;
			}
			idx = sy * size + sx;
			if (mask[idx] && depth[idx] > strands[j][i][2]) {  // 被别的头发挡住
				strands_visible[j][i] = false;
				continue;
			}
			if (strands[j].size() == 1) {	// 没有头发的点不投影
				strands_visible[j][i] = true;
				continue;
			}
			// undirection [0, PI]
			if (tangents[j][i][1] < 0) {
				x[idx] = -tangents[j][i][0];
				y[idx] = -tangents[j][i][1];
			}
			else {
				x[idx] = tangents[j][i][0];
				y[idx] = tangents[j][i][1];
			}
			mask[idx] = true;
			depth[idx] = strands[j][i][2];
			strands_visible[j][i] = true;
		}
	}

	output_x(x, y, mask, size);

	delete[] x, y, depth, mask;
}

void Hair::output_x(const float* x, const float* y, const bool* mask, int size) const {
	ofstream f("train_data\\x\\" + fname + suffix + ".ppm");
	float alpha;
	Vec color;
	if (f.is_open()) {
		f << "P3\n" << size << " " << size << " 255\n";
		double u, v;
#pragma omp parallel for
		for (int i = 0; i < size*size; ++i) {
			alpha = atan2(y[i], x[i]) / M_PI;
			if (mask[i])
				color = alpha * Vec(1, 0, 0) + (1 - alpha) * Vec(0, 0, 1);
			else color = Vec(0, 0, 0);
			f << color << endl;
		}
	}
	f.close();
}


void Hair::train_y(int grid, const vector<int>& idx, const short* mask) {
	int i, j;
	int num = idx.size();
	
	// 计算样品头发的曲率
	if (recompute_curv) {
		curvatures.clear();
		curvatures.resize(num);
#pragma omp parallel for
		for (i = 0; i < num; ++i) {
			curvatures[i].resize(100);
			if (strands[idx[i]].size() == 1) {
				for (j = 0; j < 100; ++j)
					curvatures[i][j] = 0;
			}
			else {
				for (j = 0; j < 98; ++j) {
					// 计算外接圆半径
					Vec ac = strands[idx[i]][j + 1] - strands[idx[i]][j];
					Vec ab = strands[idx[i]][j + 2] - strands[idx[i]][j];
					Vec abXac = ab.cross(ac);
					if (abXac.len2() == 0)
						curvatures[i][j + 1] = 0;
					else {
						Vec toCircumsphereCenter = (abXac.cross(ab)*ac.len2() + ac.cross(abXac)*ab.len2()) / (2.f*abXac.len2());
						float radius = toCircumsphereCenter.len();
						curvatures[i][j + 1] = 1 / radius;
					}
				}
				// padding
				curvatures[i][0] = curvatures[i][1];
				curvatures[i][99] = curvatures[i][98];
			}
		}
	}
	output_y(grid, idx, mask);
}

void Hair::output_y(int grid, const vector<int>& sample_idx, const short* mask) const {
	// 将samples和curvatures放入连续内存空间
	int num = sample_idx.size();
	float* vis = new float[grid * grid * 100];
	float* smp = new float[grid * grid * 300];
	float* curv = new float[grid * grid * 100];
	strand tmp;
	strand_visible visible;
	int i, j, k, idx, pos = 0;

	// 填vis_data
#pragma omp parallel for
	for (i = 0; i < grid; ++i) {
		for (j = 0; j < grid; ++j) {
			idx = i * grid + j;
			if (mask[idx] == 0) {
				for (k = 0; k < 100; ++k)
					vis[idx * 100 + k] = 0;
			}
			else {
				visible = strands_visible[sample_idx[pos]];
				if (visible.size() == 1) {
					for (k = 0; k < 100; ++k)
						vis[100 * idx + k] = (visible[0] ? 10 : 0.1);
				}
				else {
					for (k = 0; k < 100; ++k)
						vis[100 * idx + k] = (visible[k] ? 10 : 0.1);
				}
				++pos;
			}
		}
	}
	// 填smp_data
	pos = 0;
#pragma omp parallel for
	for (i = 0; i < grid; ++i) {
		for (j = 0; j < grid; ++j) {
			idx = i * grid + j;
			if (mask[idx] == 0) {
				for (k = 0; k < 100; ++k)
					smp[idx * 300 + 3*k] = smp[idx * 300 + 3*k + 1] = smp[idx * 300 + 3*k + 2] = 0;
			}
			else {
				tmp = strands[sample_idx[pos]];
				if (tmp.size() == 1) {
					for (k = 0; k < 100; ++k) 
						smp[idx * 300 + 3*k] = smp[idx * 300 + 3*k + 1] = smp[idx * 300 + 3*k + 2] = 0;
				}
				else {
					for (k = 0; k < 100; ++k) {
						smp[idx * 300 + 3*k] = tmp[k][0] - tmp[0][0];
						smp[idx * 300 + 3*k + 1] = tmp[k][1] - tmp[0][1];
						smp[idx * 300 + 3*k + 2] = tmp[k][2] - tmp[0][2];
					}
				}
				++pos;
			}
		}
	}
	// 填curv_data
	pos = 0;
#pragma omp parallel for
	for (i = 0; i < grid; ++i) {
		for (j = 0; j < grid; ++j) {
			idx = i * grid + j;
			if (mask[idx] == 0) {
				for (k = 0; k < 100; ++k)
					curv[idx * 100 + k] = 0;
			}
			else {
				for (k = 0; k < 100; ++k)
					curv[idx * 100 + k] = curvatures[pos][k];
				++pos;
			}
		}
	}

	// 写入HDf5
	hid_t file_id = H5Fcreate(("train_data\\y\\" + fname + suffix + ".h5").c_str(),
		H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	herr_t status;
	// dataset angle
	hid_t angle_space = H5Screate(H5S_SCALAR);
	hid_t angle_data = H5Dcreate(file_id, "/angle", H5T_NATIVE_FLOAT, angle_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(angle_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &angle);
	status = H5Dclose(angle_data);
	status = H5Sclose(angle_space);
	// dataset visible
	unsigned rank = 3;
	hsize_t vis_dims[3];
	vis_dims[0] = grid; vis_dims[1] = grid; vis_dims[2] = 100;
	hid_t visible_space = H5Screate_simple(rank, vis_dims, NULL);
	hid_t visible_data = H5Dcreate(file_id, "/weight", H5T_NATIVE_FLOAT, visible_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(visible_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vis);
	status = H5Dclose(visible_data);
	status = H5Sclose(visible_space);
	// dataset strands position
	rank = 4;
	hsize_t pos_dims[4];
	pos_dims[0] = grid; pos_dims[1] = grid; pos_dims[2] = 100; pos_dims[3] = 3;
	hid_t pos_space = H5Screate_simple(rank, pos_dims, NULL);
	hid_t pos_data = H5Dcreate(file_id, "/pos", H5T_NATIVE_FLOAT, pos_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(pos_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, smp);
	status = H5Dclose(pos_data);
	status = H5Sclose(pos_space);
	// dataset curvatures
	rank = 3;
	hsize_t curv_dims[3];
	curv_dims[0] = grid; curv_dims[1] = grid; curv_dims[2] = 100;
	hid_t curv_space = H5Screate_simple(rank, curv_dims, NULL);
	hid_t curv_data = H5Dcreate(file_id, "/curv", H5T_NATIVE_FLOAT, curv_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(curv_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, curv);
	status = H5Dclose(curv_data);
	status = H5Sclose(curv_space);

	status = H5Fclose(file_id);

	delete[] vis, smp, curv;
}


void Hair::rotate(float a) {
#pragma omp parallel for
	for (int i = 0; i < strands.size(); ++i) {
		for (int j = 0; j < strands[i].size(); ++j) {
			rotate_y(strands[i][j], a / 180 * M_PI - angle);
		}
	}
	suffix = a > 0 ? "_right" : "_left";
	angle = a / 180 * M_PI - angle;
	recompute_curv = false;
	if(is_long() && abs(a) < 20) depth_thred = -0.06;
	else depth_thred = 0;
}


void Hair::flip() {
#pragma omp parallel for
	for (int i = 0; i < strands.size(); ++i) {
		for (int j = 0; j < strands[i].size(); ++j) {
			strands[i][j][0] = -strands[i][j][0];
		}
	}
	fname += "_flip";
}