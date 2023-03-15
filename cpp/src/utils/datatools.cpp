#include "datatools.hpp"

map<string, vector<StereoData>> load_dataset(vector<string> select_dataset)
{
    string dataset_path = "data/";
    map<string, vector<StereoData>> loaded_dataset;

    for (const auto &dataset : select_dataset)
    {
        // vector<string> dataset_folders;
        vector<int> folder_ids = {1, 2, 3};
        // glob(dataset_path + dataset + "*", dataset_folders);

        vector<StereoData> loaded_sub_dataset;

        for (const auto id : folder_ids)
        {
            auto folder = dataset_path + dataset + to_string(id);
            StereoData sd;

            // Load images
            cout << "Loading images from " << folder << endl;
            sd.img_0 = imread(folder + "/im0.png");
            sd.img_1 = imread(folder + "/im1.png");

            // Load disparity maps (ground truth)
            cout << "Loading disparity maps from " << folder << endl;
            sd.disp_gt_0 = imread(folder + "/disp0.pfm");
            sd.disp_gt_1 = imread(folder + "/disp1.pfm");

            // Load camera parameters
            cout << "Loading camera parameters from " << folder << endl;
            ifstream file(folder + "/calib.txt");
            string line;
            getline(file, line);
            float K_0[3][3];
            sscanf(line.c_str(), "cam0=[%f %f %f; %f %f %f; %f %f %f]", &K_0[0][0], &K_0[0][1], &K_0[0][2], &K_0[1][0], &K_0[1][1], &K_0[1][2], &K_0[2][0], &K_0[2][1], &K_0[2][2]);
            sd.cam_0 = Matx33d(K_0[0][0], K_0[0][1], K_0[0][2], K_0[1][0], K_0[1][1], K_0[1][2], K_0[2][0], K_0[2][1], K_0[2][2]);
            sd.f = K_0[0][0];

            getline(file, line);
            float K_1[3][3];
            sscanf(line.c_str(), "cam1=[%f %f %f; %f %f %f; %f %f %f]", &K_1[0][0], &K_1[0][1], &K_1[0][2], &K_1[1][0], &K_1[1][1], &K_1[1][2], &K_1[2][0], &K_1[2][1], &K_1[2][2]);
            sd.cam_1 = Matx33d(K_1[0][0], K_1[0][1], K_1[0][2], K_1[1][0], K_1[1][1], K_1[1][2], K_1[2][0], K_1[2][1], K_1[2][2]);

            getline(file, line);
            sscanf(line.c_str(), "doffs=%f", &sd.doffs);

            getline(file, line);
            sscanf(line.c_str(), "baseline=%f", &sd.baseline);

            getline(file, line);
            sscanf(line.c_str(), "width=%d", &sd.width);

            getline(file, line);
            sscanf(line.c_str(), "height=%d", &sd.height);

            getline(file, line);
            sscanf(line.c_str(), "ndisp=%d", &sd.ndisp);

            getline(file, line);
            sscanf(line.c_str(), "vmin=%d", &sd.vmin);

            getline(file, line);
            sscanf(line.c_str(), "vmax=%d", &sd.vmax);

            loaded_sub_dataset.push_back(sd);
        }
        loaded_dataset[dataset] = loaded_sub_dataset;
    }

    return loaded_dataset;
}


void plot_imgs(vector<Mat> imgs, string title, Size figsize)
{
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        auto win_title = title + to_string(i);
        namedWindow(win_title, WINDOW_NORMAL);
        // resizeWindow(win_title, figsize.width, figsize.height);

        Mat normalized;
        normalize(imgs[i], normalized, 0, 255, NORM_MINMAX, imgs[i].type());
        
        imshow(win_title, normalized);
        waitKey(1);
    }

    waitKey(0);
    destroyAllWindows();
}