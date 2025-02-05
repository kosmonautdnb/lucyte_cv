// Lucyte Created on: 02.02.2025 by Stefan Mader
//
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencl_refinement.hpp"
#include <fstream>
#include <sstream>

std::string absoluteScaleFileName = "g:/KITTIE/data_odometry_poses/dataset/poses/00.txt";
std::string filenamesLeft = "g:/KITTIE/data_odometry_color/dataset/sequences/00/image_2/%06d.png";
std::string filenamesRight = "g:/KITTIE/data_odometry_color/dataset/sequences/00/image_3/%06d.png";
double absx, absy, absz;
double startx, starty, startz;
#define KEYPOINTCOUNT 3000

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


double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {
    std::string line;    int i = 0; std::ifstream myfile(absoluteScaleFileName);
    double x = 0, y = 0, z = 0; double x_prev, y_prev, z_prev;
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
    }
    else {
        return 0;
    }
    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
}

int main(int argc, char** argv) {
    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL();

    cv::Mat E, R, t, mask, R_f, t_f;
    R = cv::Mat::zeros(3, 3, CV_64F);
    R.at<double>(0, 0) = 1;
    R.at<double>(1, 1) = 1;
    R.at<double>(2, 2) = 1;
    t = cv::Mat::zeros(3, 1, CV_64F);
    R_f = R.clone();
    t_f = t.clone();
    std::vector<::KeyPoint> lastFrameLeftImageKeyPoints;
    std::vector<::KeyPoint> leftImageKeyPoints;
    std::vector<::KeyPoint> lastFrameRightImageKeyPoints;
    std::vector<::KeyPoint> rightImageKeyPoints;
    std::vector<float> lastFrameLeftImageErrors;
    std::vector<float> leftImageErrors;
    std::vector<float> lastFrameRightImageErrors;
    std::vector<float> rightImageErrors;
    std::vector<cv::Mat> leftMipMaps;
    std::vector<cv::Mat> rightMipMaps;

    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    char filename[200];
    snprintf(filename, 200, filenamesLeft.c_str(), firstFrame);  
    cv::Mat leftImage = cv::imread(filename);
    cv::Mat lastLeftImage_grey; cv::cvtColor(leftImage, lastLeftImage_grey, cv::COLOR_BGR2GRAY);

    int runs = 0;
    for (int steps = firstFrame; steps < lastFrame; steps++) {
        const int doubleBuffer = runs & 1; runs++;

        snprintf(filename, 200, filenamesLeft.c_str(), steps);  cv::Mat leftImage = cv::imread(filename);
        snprintf(filename, 200, filenamesRight.c_str(), steps); cv::Mat rightImage = cv::imread(filename);
        cv::Mat leftImage_grey; cv::cvtColor(leftImage, leftImage_grey, cv::COLOR_BGR2GRAY);
        cv::Mat rightImage_grey; cv::cvtColor(rightImage, rightImage_grey, cv::COLOR_BGR2GRAY);
        leftMipMaps = mipMaps(leftImage_grey);
        rightMipMaps = mipMaps(rightImage_grey);
        const int mipEnd = leftMipMaps.size() - 1;
        const bool wait = true;
        uploadMipMaps_openCL(0, doubleBuffer * 2 + 0, leftMipMaps);  if (wait) uploadMipMaps_openCL_waitfor(0, doubleBuffer * 2 + 0);
        uploadMipMaps_openCL(0, doubleBuffer * 2 + 1, rightMipMaps); if (wait) uploadMipMaps_openCL_waitfor(0, doubleBuffer * 2 + 1);

        const int missing = KEYPOINTCOUNT - leftImageKeyPoints.size();
        std::vector<cv::KeyPoint> corners;
        cv::FAST(lastLeftImage_grey, corners, 20, true);
        for (int i = 0; i < missing; i++) {
            int b = rand() % corners.size();
            leftImageKeyPoints.push_back({ corners[b].pt.x, corners[b].pt.y });
            rightImageKeyPoints.push_back({ corners[b].pt.x, corners[b].pt.y });
            leftImageErrors.push_back(10);
            rightImageErrors.push_back(10);
            corners.erase(corners.begin() + b); if (corners.empty()) break;
        }
        lastLeftImage_grey = leftImage_grey;

        uploadKeyPoints_openCL(0, 0, leftImageKeyPoints); if (wait) uploadKeyPoints_openCL_waitfor(0, 0);
        uploadKeyPoints_openCL(0, 1, leftImageKeyPoints); if (wait) uploadKeyPoints_openCL_waitfor(0, 1);
        std::vector<std::vector<Descriptor>> lastFrameLeftDescriptors;
        lastFrameLeftDescriptors.resize(leftMipMaps.size());
        for (int i = 0; i <= mipEnd; i++) {
            lastFrameLeftDescriptors[i].resize(leftImageKeyPoints.size());
            const float mipScale = powf(MIPSCALE, float(i));
            const float descriptorScale = 1.f / mipScale;
            const int width = leftMipMaps[i].cols;
            const int height = leftMipMaps[i].rows;
            sampleDescriptors_openCL(0, 0, 0, (doubleBuffer^1) * 2 + 0, i, leftImageKeyPoints.size(), descriptorScale, width, height, mipScale);
        }
        for (int i = 0; i <= mipEnd; i++) {
            sampleDescriptors_openCL_waitfor(0, 0, i, lastFrameLeftDescriptors);
        }
        uploadDescriptors_openCL(0, 0, mipEnd, lastFrameLeftDescriptors);
        if (wait) uploadDescriptors_openCL_waitfor(0, 0, mipEnd);
        lastFrameLeftImageKeyPoints = leftImageKeyPoints;
        lastFrameLeftImageErrors = leftImageErrors;
        lastFrameRightImageKeyPoints = rightImageKeyPoints;
        lastFrameRightImageErrors = rightImageErrors;
        refineKeyPoints_openCL(0, 0, 0, (doubleBuffer) * 2 + 0, leftImageKeyPoints.size(), mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        if (wait) refineKeyPoints_openCL_waitfor(0, 0, leftImageKeyPoints, leftImageErrors);
        refineKeyPoints_openCL(0, 0, 1, (doubleBuffer) * 2 + 1, leftImageKeyPoints.size(), mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        if (!wait) refineKeyPoints_openCL_waitfor(0, 0, leftImageKeyPoints, leftImageErrors);
        refineKeyPoints_openCL_waitfor(0, 1, rightImageKeyPoints, rightImageErrors);
        for (int i = leftImageKeyPoints.size() - 1; i >= 0; i--) {
            const double deltaXToRight = leftImageKeyPoints[i].x - rightImageKeyPoints[i].x;
            const double deltaYToRight = leftImageKeyPoints[i].y - rightImageKeyPoints[i].y;
            if (leftImageErrors[i] > 0.05 || abs(deltaYToRight)>2.0) {
                leftImageKeyPoints.erase(leftImageKeyPoints.begin() + i);
                leftImageErrors.erase(leftImageErrors.begin() + i);
                rightImageKeyPoints.erase(rightImageKeyPoints.begin() + i);
                rightImageErrors.erase(rightImageErrors.begin() + i);
                lastFrameLeftImageKeyPoints.erase(lastFrameLeftImageKeyPoints.begin() + i);
                lastFrameLeftImageErrors.erase(lastFrameLeftImageErrors.begin() + i);
                lastFrameRightImageKeyPoints.erase(lastFrameRightImageKeyPoints.begin() + i);
                lastFrameRightImageErrors.erase(lastFrameRightImageErrors.begin() + i);
            }
        }
        if (leftImageKeyPoints.size() > 10) {
            std::vector<cv::Point2f> thisFeaturesLeft;  thisFeaturesLeft.resize(leftImageKeyPoints.size());
            std::vector<cv::Point2f> lastFeaturesLeft;  lastFeaturesLeft.resize(lastFrameLeftImageKeyPoints.size());
            std::vector<cv::Point2f> thisFeaturesRight; thisFeaturesRight.resize(rightImageKeyPoints.size());
            std::vector<cv::Point2f> lastFeaturesRight; lastFeaturesRight.resize(lastFrameRightImageKeyPoints.size());
            for (int i = 0; i < leftImageKeyPoints.size(); i++) { thisFeaturesLeft[i].x = leftImageKeyPoints[i].x; thisFeaturesLeft[i].y = leftImageKeyPoints[i].y; }
            for (int i = 0; i < lastFrameLeftImageKeyPoints.size(); i++) { lastFeaturesLeft[i].x = lastFrameLeftImageKeyPoints[i].x; lastFeaturesLeft[i].y = lastFrameLeftImageKeyPoints[i].y; }
            for (int i = 0; i < rightImageKeyPoints.size(); i++) { thisFeaturesRight[i].x = rightImageKeyPoints[i].x; thisFeaturesRight[i].y = rightImageKeyPoints[i].y; }
            for (int i = 0; i < lastFrameRightImageKeyPoints.size(); i++) { lastFeaturesRight[i].x = lastFrameRightImageKeyPoints[i].x; lastFeaturesRight[i].y = lastFrameRightImageKeyPoints[i].y; }
            E = findEssentialMat(thisFeaturesLeft, lastFeaturesLeft, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
            recoverPose(E, thisFeaturesLeft, lastFeaturesLeft, R, t, focal, pp, mask);
            const double moveDistance = getAbsoluteScale(steps, 0, t.at<double>(2));
            if ((moveDistance > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
                t_f = t_f + moveDistance * (R_f * t);
                R_f = R * R_f;
            }
            for (int i = 0; i < thisFeaturesLeft.size(); i++) cv::line(leftImage, thisFeaturesLeft[i], lastFeaturesLeft[i], cv::Scalar(255, 0, 255));
            for (int i = 0; i < thisFeaturesRight.size(); i++) cv::line(rightImage, thisFeaturesRight[i], lastFeaturesRight[i], cv::Scalar(255, 0, 255));
        }

        int x = int(t_f.at<double>(0));
        int z = int(t_f.at<double>(2));
        circle(traj, cv::Point(absx + 300, absz + 100), 1, CV_RGB(255, 255, 0), 2);
        circle(traj, cv::Point(x + 300, z + 100), 1, CV_RGB(255, 0, 0), 2);

        imshow("left", leftImage);
        imshow("right", rightImage);
        imshow("trajectory", traj);

        if (cv::waitKey(100) == 27)
            break;
    }

    return 0;
}