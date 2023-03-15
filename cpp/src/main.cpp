#pragma once

#include "utils/ScopeBasedTimer.hpp"
#include "stereovision/stereovision.hpp"
#include "utils/datatools.hpp"

#include <iostream>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <tuple>
#include <map>

// #include "zip.h"
// #include <curl/curl.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


vector<tuple<Mat, Mat>> run_stereo_correspondance(vector<StereoData> dataset, string name)
{
    vector<tuple<Mat, Mat>> results;

    {
        Timer timer;
        results = get_disparity_wrapper(dataset);
    }

    plot_imgs({get<0>(results[0]), get<0>(results[1]), get<0>(results[2])}, name + "_disparity");

    return results;
}


int main()
{
    // const char *url = "https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip";
    // cout << "downloading..." << endl;
    // download_zip_file(url);
    // cout << "unzipping..." << endl;
    // unzip_file();

    cout << "Starting..." << endl;

    auto cwd = filesystem::current_path();
    cout << "Current path is " << cwd << endl;

    vector<string> select_dataset = {"chess", "curule", "skiboots"};

    map<string, vector<StereoData>> dataset;
    {
        Timer timer;
        dataset = load_dataset(select_dataset);
    }

    plot_imgs({dataset["chess"][0].img_0, dataset["chess"][0].img_1}, "chess");

    vector<tuple<Mat, Mat>> chess_results = run_stereo_correspondance(dataset["chess"], "chess");
    // vector<tuple<Mat, Mat>> curule_results = run_stereo_correspondance(dataset["curule"], "curule");
    // vector<tuple<Mat, Mat>> skiboots_results = run_stereo_correspondance(dataset["skiboots"], "skiboots");

    get_pointcloud(dataset["chess"][0].img_0, get<0>(chess_results[0]), get<1>(chess_results[0]), dataset["chess"][0].f, dataset["chess"][0].vmin, dataset["chess"][0].vmax);

    return 0;
}
