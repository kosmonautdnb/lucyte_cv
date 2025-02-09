// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencl_refinement.hpp"

const int SEED = 0x13337;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
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
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<float>& errors, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<float>& lastFrameErrors) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    for (int i = 0; i < keyPoints.size(); i++) {
        if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
            const float sx = 2.f;
            const float sy = 4.f;
            if (distance >= MAXVARIANCEINPIXELS) {
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) - sx * cosf(a), keyPoints[i].y - sy * cosf(a) + sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) + sx * cosf(a), keyPoints[i].y - sy * cosf(a) - sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
            else {
                cv::circle(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), (int)floorf(MAXVARIANCEINPIXELS), cv::Scalar(255, 255, 255));
            }
        }
    }
    imshow(windowName, mat);
    return mat;
}

std::vector<MipMap> mipMaps(const std::vector<cv::Mat>& m) { std::vector<MipMap> r; r.resize(m.size()); for (int i = 0; i < r.size(); i++) { r[i].width = m[i].cols; r[i].height = m[i].rows; r[i].data = m[i].data; } return r; }
std::vector<cv::Mat> mipMaps(const cv::Mat& mat) {
    cv::Mat k;
    cv::cvtColor(mat, k, cv::COLOR_RGB2GRAY);
    int width = k.cols;
    int height = k.rows;
    std::vector<cv::Mat> mipmaps;
    while (k.cols > 4 && k.rows > 4) {
        cv::Mat clone = k.clone();
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size((int)(floorf((float)k.cols * MIPSCALE)), (int)(floorf((float)k.rows * MIPSCALE))), 0.f, 0.f, cv::INTER_AREA);
    }
    return mipmaps;
}

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

int main(int argc, char** argv)
{
    initOpenCL();
    srand(SEED);

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(mipMaps(mipmaps1));
    int mipLevels = (int)floorf(MIPEND * (float)mipmaps1.size());
    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    std::vector<bool> found;
    float rectangleLeft = 600;
    float rectangleTop = 130;
    float rectangleRight = 680;
    float rectangleBottom = 430;
    keyPoints.resize(KEYPOINTCOUNT);
    found.resize(KEYPOINTCOUNT);
    while (true) {
        int left = 0;
        for (int i = 0; i < KEYPOINTCOUNT; i++) {
            if (!found[i]) {
                left++;
                keyPoints[i].x = frrand2(1.f)*(rectangleRight - rectangleLeft) + rectangleLeft;
                keyPoints[i].y = frrand2(1.f)*(rectangleBottom - rectangleTop) + rectangleTop;
            }
        }
        if (left == 0) break;
        printf("%d,", left);
        uploadKeyPoints_openCL(keyPoints);
        sampleDescriptors_openCL(mipLevels, searchForDescriptors, 1.f, MIPSCALE);
        uploadDescriptors_openCL(mipLevels, searchForDescriptors);
        refineKeyPoints_openCL(keyPoints, errors, mipLevels, STEPCOUNT, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        for (int i = 0; i < KEYPOINTCOUNT; i++) {
            if (!found[i]) {
                if (errors[i] < MAXVARIANCEINPIXELS) {
                    found[i] = true;
                }
            }
        }
    }
    printf("\n");

    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        uploadMipMaps_openCL(mipMaps(mipmaps2));
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;
        refineKeyPoints_openCL(keyPoints, errors, mipLevels, STEPCOUNT, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

        cv::Mat v = output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27) 
            break;

        uploadKeyPoints_openCL(keyPoints);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            sampleDescriptors_openCL(mipLevels, resampledDescriptors, 1.f, MIPSCALE);
            for (int i = mipLevels-1; i >= 0; i--)
                for (int j = (int)keyPoints.size() - 1; j >= 0; j--)
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS))
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
            uploadDescriptors_openCL(mipLevels, searchForDescriptors);
        }
    }

    if (outputVideo) video.release();

    return 0;
}
