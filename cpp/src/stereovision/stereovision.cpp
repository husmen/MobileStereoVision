#include "stereovision.hpp"

tuple<Mat, Mat> get_disparity(Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax)
{
    // Resize images to speed up computation
    int new_height = left_img.size().height / RESIZE_FACTOR;
    int new_width = left_img.size().width / RESIZE_FACTOR;

    Mat left_img_d = Mat::zeros(new_height, new_width, left_img.type());
    resize(left_img, left_img_d, Size(new_width, new_height), 0, 0, INTER_AREA);

    Mat right_img_d = Mat::zeros(new_height, new_width, right_img.type());
    resize(right_img, right_img_d, Size(new_width, new_height), 0, 0, INTER_AREA);

    Mat left_gray, right_gray;
    cvtColor(left_img_d, left_gray, COLOR_RGB2GRAY);
    cvtColor(right_img_d, right_gray, COLOR_RGB2GRAY);

    // plot_imgs({left_gray, right_gray}, "disparity");

    // Compute disparity map
    Ptr<StereoSGBM> stereo = StereoSGBM::create(
        vmin / RESIZE_FACTOR,                 // minDisparity
        (vmax - vmin) / RESIZE_FACTOR,                // numDisparities
        BLOCK_SIZE,                   // blockSize
        1 * BLOCK_SIZE * BLOCK_SIZE,       // P1
        2 * BLOCK_SIZE * BLOCK_SIZE,      // P2
        1,                    // disp12MaxDiff
        10,                 // preFilterCap
        10,                   // uniquenessRatio
        100,                  // speckleWindowSize
        2,                   // speckleRange
        StereoSGBM::MODE_HH); // mode

    Mat disp, disp_conv;
    stereo->compute(left_gray, right_gray, disp);

    // Converting disparity values to CV_32F from CV_16S
    // auto tmp = disp.type() == CV_16S;
    disp.convertTo(disp_conv, CV_32F, 1.0);
    // disp_conv = (disp_conv/16.0f - (float)vmin)/((float)MAX_NDISP);
 
    Mat disp_norm;
    normalize(disp_conv, disp_norm, 0, 255, NORM_MINMAX, CV_8U);

    // Convert disparity to depth
    Mat depth = f * baseline / (disp + max(1e-6f, doffs));

    // plot_imgs({disp, disp_conv, disp_norm, depth}, "disparity");

    // Resize disparity and depth maps to original size
    Mat disp_u, depth_u;
    resize(disp_norm, disp_u, left_img.size(), 0, 0, INTER_CUBIC);
    resize(depth, depth_u, left_img.size(), 0, 0, INTER_CUBIC);

    // plot_imgs({disp_u, depth_u}, "chess");

    return make_tuple(disp_u, depth_u);
}

void get_disparity_t(promise<tuple<Mat, Mat>> &&promise, Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax)
{
    auto res = get_disparity(left_img, right_img, f, baseline, doffs, ndisp, vmin, vmax);
    promise.set_value(res);
}

vector<tuple<Mat, Mat>> get_disparity_wrapper(vector<StereoData> stereo_data)
{
    vector<tuple<cv::Mat, cv::Mat>> result_list;
    vector<thread> disparit_threads;

    vector<promise<tuple<Mat, Mat>>> promises = vector<promise<tuple<Mat, Mat>>>(stereo_data.size());
    vector<future<tuple<Mat, Mat>>> futures = vector<future<tuple<Mat, Mat>>>(stereo_data.size());
    for (int i = 0; i < stereo_data.size(); i++)
    {
        futures[i] = promises[i].get_future();
    }

    for (int i = 0; i < stereo_data.size(); i++)
    {
        StereoData sd = stereo_data[i];
        // auto tmp = get_disparity(sd.img_0, sd.img_1, sd.f, sd.baseline, sd.doffs, sd.ndisp, sd.vmin, sd.vmax);
        disparit_threads.push_back(thread(&get_disparity_t, move(promises[i]), sd.img_0, sd.img_1, sd.f, sd.baseline, sd.doffs, sd.ndisp, sd.vmin, sd.vmax));
    }

    for (int i = 0; i < disparit_threads.size(); i++)
    {
        disparit_threads[i].join();
        result_list.push_back(futures[i].get());
    }

    return result_list;
}

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using namespace cv;

Mat get_pointcloud(Mat left_img, Mat disp, Mat depth, float f, float vmin, float vmax)
{
    // Convert depth to point cloud
    int h = left_img.rows;
    int w = left_img.cols;

    Mat Q = (Mat_<float>(4,4) << 1, 0, 0, -0.5*w,
                                 0,-1, 0,  0.5*h, // turn points 180 deg around x-axis,
                                 0, 0, 0,     -f, // so that y-axis looks up
                                 0, 0, 1,      0);

    Mat points_3d;
    reprojectImageTo3D(disp, points_3d, Q);

    vector<Mat> points_3d_channels(3);
    split(points_3d, points_3d_channels);

    Mat points_3d_transposed;
    hconcat(points_3d_channels[0], points_3d_channels[1], points_3d_transposed);
    hconcat(points_3d_transposed, points_3d_channels[2], points_3d_transposed);

    Mat colors;
    cvtColor(left_img, colors, COLOR_BGR2RGB);

    vector<Mat> colors_channels(3);
    split(colors, colors_channels);

    Mat colors_transposed;
    hconcat(colors_channels[0], colors_channels[1], colors_transposed);
    hconcat(colors_transposed, colors_channels[2], colors_transposed);

    // filter noise
    // Mat points_mask = (disp > vmin) & (disp < vmax);
    // points_3d = points_3d.reshape(3, 1);
    // colors = colors.reshape(3, 1);
    // points_3d = points_3d.reshape(1, points_3d.total()).t();
    // colors = colors.reshape(1, colors.total()).t();
    // points_3d = points_3d.reshape(points_3d.total() / 3, 3);
    // colors = colors.reshape(colors.total() / 3, 3);
    // points_3d = points_3d.compress(points_mask, 0);
    // colors = colors.compress(points_mask, 0);

    // save point cloud
    // Mat verts = Mat::zeros(points_3d.rows, points_3d.cols + colors.cols, CV_32FC1);
    // hconcat(points_3d, colors, verts);
    // ofstream outfile("res_cv.ply");
    // outfile << "ply\nformat ascii 1.0\nelement vertex " << verts.rows << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    // for (int i = 0; i < verts.rows; i++)
    //     outfile << verts.at<float>(i, 0) << " " << verts.at<float>(i, 1) << " " << verts.at<float>(i, 2) << " " << static_cast<int>(verts.at<float>(i, 3)) << " " << static_cast<int>(verts.at<float>(i, 

    // save point cloud
    Mat verts;
    hconcat(points_3d, colors, verts);
    ofstream out_file("res_cv.ply");
    out_file << "ply\nformat ascii 1.0\n"
             << "element vertex " << verts.rows << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "property uchar red\n"
             << "property uchar green\n"
             << "property uchar blue\n"
             << "end_header\n";
    out_file << format(verts, Formatter::FMT_PYTHON) << endl;

    return Q;
}