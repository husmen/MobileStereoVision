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

    // visualize_imgs({left_gray, right_gray}, "disparity");

    // Compute disparity map
    Ptr<StereoSGBM> stereo = StereoSGBM::create(
        vmin / RESIZE_FACTOR,          // minDisparity
        (vmax - vmin) / RESIZE_FACTOR, // numDisparities
        BLOCK_SIZE,                    // blockSize
        1 * BLOCK_SIZE * BLOCK_SIZE,   // P1
        2 * BLOCK_SIZE * BLOCK_SIZE,   // P2
        1,                             // disp12MaxDiff
        10,                            // preFilterCap
        10,                            // uniquenessRatio
        100,                           // speckleWindowSize
        2,                             // speckleRange
        StereoSGBM::MODE_HH);          // mode

    Mat disp, disp_conv;
    stereo->compute(left_gray, right_gray, disp);

    // Converting disparity values to CV_32F from CV_16S
    // auto tmp = disp.type() == CV_16S;
    disp.convertTo(disp_conv, CV_32F, 1.0 / 16.0);
    disp_conv = (disp_conv/16.0f - (float)vmin)/((float)(vmax - vmin));

    Mat disp_norm;
    normalize(disp_conv, disp_norm, 0, 255, NORM_MINMAX, CV_8U);

    // Convert disparity to depth
    Mat depth = f * baseline / (disp_conv + max(1e-6f, doffs));

    // visualize_imgs({disp, disp_conv, disp_norm, depth}, "disparity");

    // Resize disparity and depth maps to original size
    Mat disp_u, depth_u;
    resize(disp_norm, disp_u, left_img.size(), 0, 0, INTER_CUBIC);
    resize(depth, depth_u, left_img.size(), 0, 0, INTER_CUBIC);

    // visualize_imgs({disp_u, depth_u}, "chess");

    return make_tuple(disp_u, depth_u);
}

void get_disparity_t(promise<tuple<Mat, Mat>> &&promise, Mat left_img, Mat right_img, float f, float baseline, float doffs, int ndisp, int vmin, int vmax)
{
    auto res = get_disparity(left_img, right_img, f, baseline, doffs, ndisp, vmin, vmax);
    promise.set_value(res);
}

vector<tuple<Mat, Mat>> get_disparity_wrapper(vector<StereoData> stereo_data)
{
    vector<tuple<Mat, Mat>> result_list;
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

vector<PointCloud<PointXYZRGB>::Ptr> get_pointcloud(vector<StereoData> stereo_data, vector<tuple<Mat, Mat>> disps)
{
    vector<PointCloud<PointXYZRGB>::Ptr> pointclouds;
    for (int id = 0; id < stereo_data.size(); id++)
    {
        auto f = stereo_data[id].f;
        auto vmin = stereo_data[id].vmin;
        auto vmax = stereo_data[id].vmax;
        
        Mat img = stereo_data[id].img_0;
        Mat disp = get<0>(disps[id]);

        // Convert depth to point cloud
        int h = img.rows;
        int w = img.cols;

        Mat Q = (Mat_<float>(4, 4) << 1, 0, 0, -0.5 * w,
                 0, -1, 0, 0.5 * h, // turn points 180 deg around x-axis,
                 0, 0, 0, -f,       // so that y-axis looks up
                 0, 0, 1, 0);

        Mat points_3d;
        reprojectImageTo3D(disp, points_3d, Q);
        points_3d = points_3d.reshape(3, h * w);

        Mat colors;
        cvtColor(img, colors, COLOR_BGR2RGB);
        colors = img.reshape(3, h * w);

        // Create the vertices of coloured point cloud
        colors.convertTo(colors, CV_32F, 1.0 / 255.0);
        Mat verts;
        hconcat(points_3d, colors, verts);

        // save point cloud
        // ofstream out_file("res_cv.ply");
        // out_file << "ply\nformat ascii 1.0\n"
        //          << "element vertex " << h * w << "\n"
        //          << "property float x\n"
        //          << "property float y\n"
        //          << "property float z\n"
        //          << "property uchar red\n"
        //          << "property uchar green\n"
        //          << "property uchar blue\n"
        //          << "end_header\n";

        // for (int i = 0; i < verts.rows; i++)
        // {
        //     auto point = verts.at<Vec3f>(i, 0);
        //     auto color = verts.at<Vec3f>(i, 1);
        //     out_file << format("%f %f %f %d %d %d\n", point[0], point[1], point[2], (int)(color[0] * 255), (int)(color[1] * 255), (int)(color[2] * 255));
        // }

        // Create the point cloud with PCL
        PointCloud<PointXYZRGB>::Ptr point_cloud_ptr(new PointCloud<PointXYZRGB>);

        for (int i = 0; i < verts.rows; i++)
        {
            auto point = verts.at<Vec3f>(i, 0);
            auto color = verts.at<Vec3f>(i, 1);

            if (point[0] > vmax || point[0] < vmin || point[1] > vmax || point[1] < vmin)
                continue;

            PointXYZRGB rgb_point;
            rgb_point.x = point[0];
            rgb_point.y = point[1];
            rgb_point.z = point[2];
            rgb_point.r = color[0] * 255;
            rgb_point.g = color[1] * 255;
            rgb_point.b = color[2] * 255;
            point_cloud_ptr->points.push_back(rgb_point);
        }

        pointclouds.push_back(point_cloud_ptr);
    }

    return pointclouds;
}