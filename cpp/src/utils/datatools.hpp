#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

// #include "zip.h"
// #include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;
using namespace pcl;

struct StereoData
{
    Mat img_0;
    Mat img_1;
    Mat disp_gt_0;
    Mat disp_gt_1;
    Matx33d cam_0;
    Matx33d cam_1;
    float doffs;
    float baseline;
    int width;
    int height;
    int ndisp;
    int vmin;
    int vmax;
    float f;
    Mat disp_0;
    Mat disp_1;
    Mat depth_0;
    Mat depth_1;
};

// size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream);

// int download_zip_file(const char* url);

// int unzip_file();

// std::vector<std::string> split(const std::string &str, char delimiter);

map<string, vector<StereoData>> load_dataset(vector<string> select_dataset);
void visualize_imgs(vector<Mat> imgs, string title, Size figsize = Size(1280, 720));
void visualize_pcds(vector<PointCloud<PointXYZRGB>::Ptr> pcds, string title);