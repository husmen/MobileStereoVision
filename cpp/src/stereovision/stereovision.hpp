#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <tuple>
#include <thread>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include "../utils/datatools.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

#define RESIZE_FACTOR 2
#define BLOCK_SIZE 13

tuple<Mat, Mat> get_disparity(Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax);
void get_disparity_t(promise<tuple<Mat, Mat>> &&promise, Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax);
vector<tuple<Mat, Mat>> get_disparity_wrapper(vector<StereoData> stereo_data);
vector<PointCloud<PointXYZRGB>::Ptr> get_pointcloud(vector<StereoData> stereo_data, vector<tuple<Mat, Mat>> disps);