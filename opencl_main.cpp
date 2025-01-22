// Created on: 21.01.2025 by Stefan Mader
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "refinement.hpp"
#include "fileset.hpp"
#include "opencl_refinement.hpp"

const unsigned int SEED = 0x13337;
const float MIPSCALE = 0.5;
const float SCALEINVARIANCE = 0.5 / 2.f; // 0.5 / 2 is good
const float ROTATIONINVARIANCE = 20.f / 2.f; // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
const int STEPCOUNT = 100;
const float STEPSIZE = 0.002f;
const float DESCRIPTORSCALE = 5.f;
const bool BOOLSTEPPING = false;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
const int KEYPOINTCOUNT = 1000;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / (float)0x10000;
}

float frrand(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad * 2.f - rad;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& variancePoints, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<KeyPoint>& lastFrameVariancePoints) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    for (int i = 0; i < keyPoints.size(); i++) {
        float varianceX = variancePoints[i].x - keyPoints[i].x;
        float varianceY = variancePoints[i].y - keyPoints[i].y;
        float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
        float lastFrameVarianceX = lastFrameVariancePoints[i].x - lastFrameKeyPoints[i].x;
        float lastFrameVarianceY = lastFrameVariancePoints[i].y - lastFrameKeyPoints[i].y;
        float lastFrameVariance = sqrtf(lastFrameVarianceX * lastFrameVarianceX + lastFrameVarianceY * lastFrameVarianceY);
        if (variance < MAXVARIANCEINPIXELS && lastFrameVariance < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            float distanceX = lastFrameVariancePoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameVariancePoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
            const float sx = 2.f;
            const float sy = 4.f;
            if (distance >= MAXVARIANCEINPIXELS) {
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
            else {
                cv::circle(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), MAXVARIANCEINPIXELS, cv::Scalar(255, 255, 255));
            }
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
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size(k.cols * MIPSCALE, k.rows * MIPSCALE), 0.f, 0.f, cv::INTER_AREA);
    }
    return mipmaps;
}

void cross(int index, double centerx, double centery, double size) {
    float a = (float)rand() * 2.f * 3.14159f / RAND_MAX;
    descriptorsX1[index * 2 + 0] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 0] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 0] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 0] = centery - sinf(a) * size;
    a += 3.14159f * 0.5 * 0.75;
    descriptorsX1[index * 2 + 1] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 1] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 1] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 1] = centery - sinf(a) * size;
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
    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL();

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(mipmaps1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

    std::vector<KeyPoint> keyPoints;
    std::vector<KeyPoint> variancePoints;
    keyPoints.resize(KEYPOINTCOUNT);
    variancePoints.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2(mipmaps1[0].cols);
        keyPoints[i].y = frrand2(mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(keyPoints);

    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());
    for (int i = mipEnd; i >= 0; i--) {
        const float descriptorScale = 1 << i;
        const float mipScale = powf(MIPSCALE, float(i));
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptors[i].resize(keyPoints.size());
        uploadDescriptors_openCL(i,searchForDescriptors);
        sampleDescriptors_openCL(i,searchForDescriptors,mipmaps1[i].data, descriptorScale, width, height, mipScale);
    }

    long long t0 = _Query_perf_counter();;
    long long fr = _Query_perf_frequency();
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        uploadMipMaps_openCL(mipmaps2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<KeyPoint> lastFrameVariancePoints = variancePoints;
        const bool openCL = true; 
        if (!openCL) 
        {
            for (int v = 0; v < (CHECKVARIANCE ? 2 : 1); v++) {
#pragma omp parallel for num_threads(32)
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    KeyPoint kp = { mipmaps2[0].cols * 0.5, mipmaps2[0].rows * 0.5 };
                    for (int i = mipEnd; i >= 0; i--) {
                        const int width = mipmaps2[i].cols;
                        const int height = mipmaps2[i].rows;
                        for (int k = 0; k < STEPCOUNT; k++) {
                            float descriptorScale = (1 << i);
                            const float mipScale = powf(MIPSCALE, float(i));
                            const float step = STEPSIZE * descriptorScale;
                            descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.f - SCALEINVARIANCE;
                            const float angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.f - ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                            Descriptor foundDescriptor;
                            sampleDescriptor(kp, foundDescriptor, mipmaps2[i].data, descriptorScale, width, height, mipScale);
                            kp = refineKeyPoint(BOOLSTEPPING, kp, searchForDescriptors[i][j], foundDescriptor, mipmaps2[i].data, descriptorScale, angle, step, width, height, mipScale);
                        }
                    }
                    switch (v) {
                    case 0: keyPoints[j] = kp;
                    case 1: variancePoints[j] = kp; break;
                    }
                }
            }
        }
        else {
            refineKeyPoints_openCL(keyPoints, variancePoints, mipEnd, STEPCOUNT, BOOLSTEPPING ? 1 : 0, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        }
        long long t1 = _Query_perf_counter();
        printf("seconds: % f\n", double(t1 - t0) / fr);
        t0 = _Query_perf_counter();

        video.write(output("keypoints", mat2, keyPoints, variancePoints, lastFrameKeyPoints, lastFrameVariancePoints));
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
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
                if (variance >= MAXVARIANCEINPIXELS) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
                variancePoints[j] = k;
            }
        }

        uploadKeyPoints_openCL(keyPoints);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            resampledDescriptors.resize(mipmaps2.size());
            for (int i = mipEnd; i >= 0; i--) {
                const float descriptorScale = 1 << i;
                const float mipScale = powf(MIPSCALE, float(i));
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
                resampledDescriptors[i].resize(keyPoints.size());
                sampleDescriptors_openCL(i,resampledDescriptors, mipmaps2[i].data, descriptorScale, width, height, mipScale);
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    float varianceX = variancePoints[j].x - keyPoints[j].x;
                    float varianceY = variancePoints[j].y - keyPoints[j].y;
                    float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                    if ((!RESAMPLEONVARIANCE) || (variance < RESAMPLEONVARIANCERADIUS)) {
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
                    }
                }
                uploadDescriptors_openCL(i, searchForDescriptors);
            }
        }
    }

    video.release();

    return 0;
}
