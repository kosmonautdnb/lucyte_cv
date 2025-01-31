// Lucyte Created on: 31.01.2025 by Stefan Mader
//
// this is taken from an internet example https://avisingh599.github.io/vision/monocular-vo/
/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencl_refinement.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000
#define KEYPOINTCOUNT (MIN_NUM_FEAT*4)
//#define KLT

using namespace cv;
using namespace std;

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}
std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;
float descriptivity(std::vector<cv::Mat>& mipMaps, const ::KeyPoint& k, const int mipEnd) {
    Descriptor d;
    float h = 0;
    for (int i = mipEnd; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipMaps[i].cols;
        const int height = mipMaps[i].rows;
        h += sampleDescriptor(k, d, mipMaps[i].data, descriptorScale, width, height, mipScale) * mipScale;
    }
    h /= float(mipEnd + 1);
    return h;
}
std::vector<cv::Mat> mipMaps(const cv::Mat& mat) {
    cv::Mat k;
    k = mat.clone();
    //cv::cvtColor(mat, k, cv::COLOR_RGB2GRAY);
    int width = k.cols;
    int height = k.rows;
    std::vector<cv::Mat> mipmaps;
    while (k.cols > 4 && k.rows > 4) {
        cv::Mat clone = k.clone();
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size(k.cols * MIPSCALE, k.rows * MIPSCALE), 0.f, 0.f, cv::INTER_AREA);
    }
    return mipmaps;
}

std::vector<std::vector<Descriptor>> d1;
std::vector<std::vector<Descriptor>> d2;

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, std::vector<cv::Mat>& mips1, std::vector<cv::Mat>& mips2, int mipEnd) {

#ifdef KLT
    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }
#else // KLT
    std::vector<::KeyPoint> p;
    std::vector<float> e;
    p.resize(KEYPOINTCOUNT);
    e.resize(KEYPOINTCOUNT);
    for (int i = 0; i < points1.size(); ++i) {
        p[i].x = points1[i].x;
        p[i].y = points1[i].y;
    }
    uploadKeyPoints_openCL(p);
    uploadMipMaps_openCL(mips1);
    d1.resize(mips1.size());
    for (int i = 0; i <= mipEnd; i++) {
        d1[i].resize(KEYPOINTCOUNT);
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mips1[i].cols;
        const int height = mips1[i].rows;
        sampleDescriptors_openCL(i, d1, descriptorScale, width, height, mipScale);
        uploadDescriptors_openCL(i, d1);
    }
    uploadMipMaps_openCL(mips2);
    refineKeyPoints_openCL(p, e, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
    points2.resize(points1.size());
    for (int i = 0; i < points1.size(); i++) {
        points2[i].x = p[i].x;
        points2[i].y = p[i].y;
    }
    uploadKeyPoints_openCL(p);
    d2.resize(mips1.size());
    for (int i = 0; i <= mipEnd; i++) {
        d2[i].resize(KEYPOINTCOUNT);
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mips2[i].cols;
        const int height = mips2[i].rows;
        sampleDescriptors_openCL(i, d2, descriptorScale, width, height, mipScale);
    }
    int indexCorrection = 0;
    int a = points2.size();
    for (int i = 0; i < a; i++)
    {
        int differing = 0;
        for (int j = 0; j <= mipEnd; j++) {
            for (int k = 0; k < DESCRIPTORSIZE; ++k) {
                differing += ((d1[j][i].bits[k >> 5] ^ d2[j][i].bits[k >> 5]) >> (k & 31)) & 1;
            }
        }
        double diffRatio = double(differing) / double(mipEnd * DESCRIPTORSIZE);
        Point2f pt = points2.at(i - indexCorrection);
        double dx = pt.x - points1.at(i - indexCorrection).x;
        double dy = pt.y - points1.at(i - indexCorrection).y;
        double d = sqrt(dx * dx + dy * dy); // erstmal
        if (e[i]>0.5f || diffRatio > 0.175) {
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }
#endif // KLT
}


void featureDetection(Mat img_1, vector<Point2f>& points1, std::vector<cv::Mat> &mips, int mipEnd) {
    vector<cv::KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints_1, points1, vector<int>());
    if (points1.size() > KEYPOINTCOUNT) points1.resize(KEYPOINTCOUNT);
}

using namespace cv;
using namespace std;

std::string absoluteScaleFileName = "e:/KITTIE/data_odometry_poses/dataset/poses/00.txt";
std::string filenames1 = "e:/KITTIE/data_odometry_color/dataset/sequences/00/image_2/%06d.png";
std::string filenames2 = "e:/KITTIE/data_odometry_color/dataset/sequences/00/image_2/%06d.png";

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!
double absx, absy, absz;
double startx, starty, startz;

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {

    string line;
    int i = 0;
    ifstream myfile(absoluteScaleFileName);
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while ((getline(myfile, line)) && (i <= frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j = 0; j < 12; j++) {
                in >> z;
                if (j == 7) y = z;
                if (j == 3)  x = z;
            }
            if (i == 0) {
                startx = x;
                starty = y;
                startz = z;
            }
            i++;
        }
        absx = x_prev - startx;
        absy = y_prev - starty;
        absz = z_prev - startz;
        //myfile.close();
    }

    else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));

}


int main(int argc, char** argv) {

    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL();

    Mat img_1, img_2;
    Mat R_f, t_f; //the final rotation and tranlation vectors containing the 

    //ofstream myfile;
    //myfile.open("results1_1.txt");

    double scale = 1.00;
    char filename1[200];
    char filename2[200];
    snprintf(filename1, 200, filenames1.c_str(), 0);
    snprintf(filename2, 200, filenames2.c_str(), 1);

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    //read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    mipmaps1 = mipMaps(img_1);
    mipmaps2 = mipMaps(img_2);


    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(img_1, points1, mipmaps1, mipmaps1.size()-1);        //detect features in img_1
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status, mipmaps1, mipmaps2, mipmaps1.size()-1); //track those features to img_2

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    //E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    //recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    R = cv::Mat::zeros(3, 3, CV_64F);
    R.at<double>(0, 0) = 1;
    R.at<double>(1, 1) = 1;
    R.at<double>(2, 2) = 1;
    t = cv::Mat::zeros(3, 1, CV_64F);

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    namedWindow("Road facing camera", WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
        snprintf(filename, 200, filenames2.c_str(), numFrame);
        //cout << numFrame << endl;
        Mat currImage_c = imread(filename);
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        mipmaps2 = mipMaps(currImage);

        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, mipmaps1, mipmaps2, mipmaps1.size()-1);

        if (currFeatures.size() > 3 && prevFeatures.size() > 3) {
            E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
            recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
        }

        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


        for (int i = 0; i < prevFeatures.size(); i++) {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        //cout << "Scale is " << scale << endl;

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)) && (currFeatures.size()>3) && (prevFeatures.size()>3)) {

            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;

        }

        else {
            //cout << "scale below 0.1, or incorrect translation" << endl;
        }

        // lines for printing results
        // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

       // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            //cout << "trigerring redection" << endl;
            featureDetection(prevImage, prevFeatures, mipmaps1, mipmaps1.size()-1);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, mipmaps1, mipmaps2, mipmaps1.size()-1);
        }

        for (int i = 0; i < currFeatures.size(); i++) {
            cv::line(currImage_c, currFeatures[i], prevFeatures[i], cv::Scalar(255, 0, 255));
        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;
        mipmaps1 = mipMaps(prevImage);


        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(absx + 300, absz + 100), 1, CV_RGB(255, 255, 0), 2);
        circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
        snprintf(text, 100, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        imshow("Road facing camera", currImage_c);
        imshow("Trajectory", traj);

        if (waitKey(100) == 27)
            break;

    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;

    //cout << R_f << endl;
    //cout << t_f << endl;

    return 0;
}