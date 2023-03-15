#pragma once

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <thread>
#include <future>
#include "../utils/datatools.hpp"

using namespace std;
using namespace cv;

#define RESIZE_FACTOR 2
#define BLOCK_SIZE 13

tuple<Mat, Mat> get_disparity(Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax);
void get_disparity_t(promise<tuple<Mat, Mat>> &&promise, Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax);
vector<tuple<Mat, Mat>> get_disparity_wrapper(vector<StereoData> stereo_data);
Mat get_pointcloud(Mat left_img, Mat disp, Mat depth, float f, float vmin, float vmax);