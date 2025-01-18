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
const bool CHECKVARIANCE = true;
const float MAXVARIANCEPIXELS = 3.0;
const float MIPSTART = 0.0;
const float STEPATTENUATION = 0.5f;
const float MIPSCALE = 0.5;
const int KEYPOINTCOUNT = 600;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

float frrand(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad * 2.f - rad;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& variancePoints, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<KeyPoint>& lastFrameVariancePoints) {
    cv::Mat mat = image.clone();
    for (int i = 0; i < keyPoints.size(); i++) {
        float varianceX = variancePoints[i].x - keyPoints[i].x;
        float varianceY = variancePoints[i].y - keyPoints[i].y;
        float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
        float lastFrameVarianceX = lastFrameVariancePoints[i].x - lastFrameKeyPoints[i].x;
        float lastFrameVarianceY = lastFrameVariancePoints[i].y - lastFrameKeyPoints[i].y;
        float lastFrameVariance = sqrtf(lastFrameVarianceX * lastFrameVarianceX + lastFrameVarianceY * lastFrameVarianceY);
        if (variance < MAXVARIANCEPIXELS && lastFrameVariance < MAXVARIANCEPIXELS) {
            const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
            const float sx = 3.f;
            const float sy = 6.f;
            cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(mat, cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(0, 0, 255));
        }
    }
    imshow(windowName, mat);
    return mat;
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
    omp_set_num_threads(32);

    defaultDescriptorShape(DESCRIPTORSCALE);

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate, cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    int mipStart = MIPSTART * mipmaps1.size();

    std::vector<KeyPoint> keyPoints;
    std::vector<KeyPoint> variancePoints;
    keyPoints.resize(KEYPOINTCOUNT);
    variancePoints.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2(mipmaps1[0].cols);
        keyPoints[i].y = frrand2(mipmaps1[0].rows);
    }

    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());
    for (int i = mipStart; i < mipmaps1.size(); i++) {
        const float descriptorScale = 1 << i;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptors[i].resize(keyPoints.size());
        for (int j = 0; j < keyPoints.size(); ++j) {
            sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps1[i].data, descriptorScale, width, height);
        }
    }

    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<KeyPoint> lastFrameVariancePoints = variancePoints;
        for (int v = 0; v < (CHECKVARIANCE ? 2 : 1); v++) {
#pragma omp parallel for
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint kp = { mipmaps2[0].cols * 0.5, mipmaps2[0].rows * 0.5 };
                for (int i = mipmaps1.size() - 1; i >= mipStart; i--) {
                    const int width = mipmaps2[i].cols;
                    const int height = mipmaps2[i].rows;
                    for (int k = 0; k < STEPCOUNT; k++) {
                        float descriptorScale = (1 << i);
                        const float step = STEPSIZE * descriptorScale * (1.f - float(k) / STEPCOUNT * STEPATTENUATION);
                        descriptorScale *= (1.0 + frrand(SCALEINVARIANCE));
                        const float angle = frrand(ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                        Descriptor foundDescriptor;
                        sampleDescriptor(kp, foundDescriptor, mipmaps2[i].data, descriptorScale, width, height);
                        kp = refineKeyPoint(BOOLSTEPPING, kp, searchForDescriptors[i][j], foundDescriptor, mipmaps2[i].data, descriptorScale, angle, step, width, height);
                    }
                }
                switch (v) {
                case 0: keyPoints[j] = kp; 
                case 1: variancePoints[j] = kp; break;
                }
            }
        }
        video.write(output("keypoints", mat2, keyPoints, variancePoints, lastFrameKeyPoints, lastFrameVariancePoints));
        cv::setWindowTitle("keypoints", std::string("Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame));
        if (cv::waitKey(1) == 27) 
            break;
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
                float varianceX = variancePoints[j].x - keyPoints[j].x;
                float varianceY = variancePoints[j].y - keyPoints[j].y;
                float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                if (variance >= MAXVARIANCEPIXELS) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
            }
        }

        const bool resample = true;
        if (resample) {
            for (int i = mipStart; i < mipmaps1.size(); i++) {
                const float descriptorScale = 1 << i;
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
#pragma omp parallel for
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, width, height);
                }
            }
        }
    }

    video.release();

    return 0;
}
