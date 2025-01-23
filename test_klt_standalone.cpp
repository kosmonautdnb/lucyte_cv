// Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "refinement.hpp"
#include "opencl_refinement.hpp"

const int SEED = 0x13337;
const float MAXERROR = 3.0;

const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / (float)0x10000;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, const std::vector<cv::Point2f> &kltStartKeyPoints, const std::vector<cv::Point2f>& kltFinishKeyPoints, const std::vector<float> &err) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    if (kltStartKeyPoints.size() == kltFinishKeyPoints.size()) {
        for (int i = 0; i < kltStartKeyPoints.size(); i++) {
            if (err[i] < MAXERROR) {
                validKeyPoints++;
                const float a = atan2(kltFinishKeyPoints[i].x - kltStartKeyPoints[i].x, kltFinishKeyPoints[i].y - kltStartKeyPoints[i].y);
                const float sx = 2.f;
                const float sy = 4.f;
                cv::line(mat, kltFinishKeyPoints[i], cv::Point(kltFinishKeyPoints[i].x - sy * sin(a) - sx * cos(a), kltFinishKeyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, kltFinishKeyPoints[i], cv::Point(kltFinishKeyPoints[i].x - sy * sin(a) + sx * cos(a), kltFinishKeyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, kltStartKeyPoints[i], cv::Point(kltFinishKeyPoints[i].x, kltFinishKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
        }
    }
    imshow(windowName, mat);
    return mat;
}

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    cv::Mat r = cv::imread(buffer, cv::IMREAD_COLOR);
    //cv::resize(r, r, cv::Size(1024, 512));
    return r;
}

int main(int argc, char** argv)
{
    srand(SEED);

    cv::Mat matg1, matg2;

    cv::Mat mat1 = loadImage(firstFrame);
    cv::cvtColor(mat1, matg1, cv::COLOR_RGB2GRAY);
    cv::VideoWriter video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);

    std::vector<cv::Point2f> keyPoints,resultKeyPoints;
    keyPoints.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2(mat1.cols);
        keyPoints[i].y = frrand2(mat1.rows);
    }

    cv::Mat m1, m2;
    long long t0 = _Query_perf_counter();;
    long long t00 = _Query_perf_counter();;
    long long fr = _Query_perf_frequency();
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        cv::cvtColor(mat2, matg2, cv::COLOR_RGB2GRAY);

        cv::Mat status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
        try {
            cv::calcOpticalFlowPyrLK(matg1, matg2, keyPoints, resultKeyPoints, status, err, cv::Size(15, 15), 2, criteria);
        }
        catch (cv::Exception& a) {
            int debug = 1;
        }

        video.write(output("keypoints", mat2, keyPoints, resultKeyPoints, err));
        cv::setWindowTitle("keypoints", std::string("(KLT) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27) 
            break;

        mat1 = mat2.clone();
        matg1 = matg2.clone();
        keyPoints = resultKeyPoints;

        const bool readd = true;
        if (readd) {
            const int width = mat2.cols;
            const int height = mat2.rows;
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                cv::Point2f &k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width-RIGHT) || (k.y < TOP) || (k.y >= height-BOTTOM)) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
                if (err[j]>=MAXERROR) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
            }
        }
    }

    video.release();

    return 0;
}
