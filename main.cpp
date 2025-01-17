// Created on: 17.01.2025 by Stefan Mader
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "refinement.hpp"
#include "fileset.hpp"

const unsigned int SEED = 0x13337;
const float SCALEINVARIANCE = 0.5 / 2.f; // 0.5 / 2 is good
const float ROTATIONINVARIANCE = 20.f / 2.f; // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
const int STEPCOUNT = 64;
const float STEPSIZE = 0.0025f;
const float DESCRIPTORSCALE = 8.f;
const bool BOOLSTEPPING = true;
const float MIPSTART = 0.0;
const float STEPATTENUATION = 0.5f;
const float MIPSCALE = 0.5;
const int KEYPOINTCOUNT = 300;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

float frrand(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad * 2.f - rad;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

cv::Mat output(const std::string& windowName, const cv::Mat& matrix1, std::vector<KeyPoint>& keyPoints1) {
    cv::Mat bg1 = matrix1.clone();
    for (int i = 0; i < keyPoints1.size(); i++) {
        cv::circle(bg1, cv::Point(keyPoints1[i].x, keyPoints1[i].y), 4.f, cv::Scalar(0, 0, 255));
    }
    imshow(windowName, bg1);
    return bg1;
}

std::vector<cv::Mat> mipMaps(const cv::Mat& mat) {
    cv::Mat k;
    cv::cvtColor(mat, k, cv::COLOR_RGB2GRAY);
    int width = k.cols;
    int height = k.rows;
    std::vector<cv::Mat> mipmaps;
    while (k.cols > 4 && k.rows > 4) {
        cv::Mat clone = k.clone();
        cv::resize(clone, clone, cv::Size(width, height), 0.f, 0.f, cv::INTER_LINEAR);
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size(k.cols * MIPSCALE, k.rows * MIPSCALE), 0.f, 0.f, cv::INTER_AREA);
    }
    return mipmaps;
}

void cross(int index, double centerx, double centery, double size) {
    descriptorsX1[index * 2 + 0] = centerx - size * 1.5;
    descriptorsY1[index * 2 + 0] = centery;
    descriptorsX2[index * 2 + 0] = centerx + size;
    descriptorsY2[index * 2 + 0] = centery;
    descriptorsX1[index * 2 + 1] = centerx;
    descriptorsY1[index * 2 + 1] = centery - size;
    descriptorsX2[index * 2 + 1] = centerx;
    descriptorsY2[index * 2 + 1] = centery + size * 1.5;
}

void defaultDescriptorShape(const double rad) {
    int i = 0;
    cross(i, 0, 0, rad);
    i++;
    cross(i, 0 - rad * 0.3, 0 - rad * 0.3, rad * 0.6);
    i++;
    cross(i, 0 + rad * 0.3, 0 + rad * 0.3, rad * 0.6);
    i++;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            cross(i, -rad * 0.5 + rad * x, -rad * 0.5 + rad * y, rad * 0.5);
            i++;
        }
    }
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            cross(i, -rad * 0.75 + rad * 0.5 * x, -rad * 0.75 + rad * 0.5 * y, rad * 0.25);
            i++;
        }
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            cross(i, -rad * 0.25 + rad * 0.25 * x, -rad * 0.25 + rad * 0.25 * y, rad * 0.3);
            i++;
        }
    }
}

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

int main(int argc, char** argv)
{
    srand(SEED);
    omp_set_num_threads(4);

    defaultDescriptorShape(DESCRIPTORSCALE);

    cv::Mat mat1 = loadImage(firstFrame);
    mipmaps1 = mipMaps(mat1);

    cv::VideoWriter video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate, cv::Size(mat1.cols, mat1.rows), true);

    std::vector<KeyPoint> groundTruthKeyPoints;
    groundTruthKeyPoints.resize(KEYPOINTCOUNT);
    for (int i = 0; i < groundTruthKeyPoints.size(); ++i) {
        groundTruthKeyPoints[i].x = frrand2(mipmaps1[0].cols);
        groundTruthKeyPoints[i].y = frrand2(mipmaps1[0].rows);
    }

    std::vector<std::vector<Descriptor>> groundTruthDescriptors;
    groundTruthDescriptors.resize(mipmaps1.size());
    int mipStart = MIPSTART * mipmaps1.size();
    for (int i = mipStart; i < mipmaps1.size(); i++) {
        const float descriptorScale = 1 << i;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        groundTruthDescriptors[i].resize(groundTruthKeyPoints.size());
        for (int j = 0; j < groundTruthKeyPoints.size(); ++j) {
            KeyPoint& kp = groundTruthKeyPoints[j];
            Descriptor& d = groundTruthDescriptors[i][j];
            sampleDescriptor(kp, d, mipmaps1[i].data, descriptorScale, width, height);
        }
    }

    std::vector<KeyPoint> keyPoints = groundTruthKeyPoints;
    for (int steps = firstFrame; steps <= lastFrame; steps++) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        for (int i = mipmaps1.size() - 1; i >= mipStart; i--) {
            const int width = mipmaps2[i].cols;
            const int height = mipmaps2[i].rows;
            std::vector<Descriptor> foundDescriptors;
            foundDescriptors.resize(keyPoints.size());
#pragma omp parallel for
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    for (int k = 0; k < STEPCOUNT; k++) {
                        float descriptorScale = (1 << i);
                        const float step = STEPSIZE * descriptorScale * (1.f-float(k)/STEPCOUNT*STEPATTENUATION);
                        descriptorScale *= (1.0 + frrand(SCALEINVARIANCE));
                        const float angle = frrand(ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                        sampleDescriptor(keyPoints[j], foundDescriptors[j], mipmaps2[i].data, descriptorScale, width, height);
                        refineKeyPoint(BOOLSTEPPING, keyPoints[j], groundTruthDescriptors[i][j], foundDescriptors[j], mipmaps2[i].data, descriptorScale, angle, step, width, height);
                    }
            }
        }
        video.write(output("keypoints", mat2, keyPoints)); cv::waitKey(1);
        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint &k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width-RIGHT) || (k.y < TOP) || (k.y >= height-BOTTOM)) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
            }
        }

        const bool resample = true;
        if (resample) {
            groundTruthKeyPoints = keyPoints;
            for (int i = mipStart; i < mipmaps1.size(); i++) {
                const float descriptorScale = 1 << i;
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
#pragma omp parallel for
                for (int j = groundTruthKeyPoints.size() - 1; j >= 0; j--) {
                    sampleDescriptor(groundTruthKeyPoints[j], groundTruthDescriptors[i][j], mipmaps2[i].data, descriptorScale, width, height);
                }
            }
        }
    }

    video.release();

    while (1) {
        if (cv::waitKey(0) == 27) break;
    }

    return 0;
}
