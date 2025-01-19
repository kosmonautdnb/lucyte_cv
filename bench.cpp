// Created on: 19.01.2025 by Stefan Mader
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "refinement.hpp"
#include "fileset.hpp"

const float SCALEINVARIANCE = 0.5 / 2.f; // 0.5 / 2 is good
const float ROTATIONINVARIANCE = 20.f / 2.f; // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
const int STEPCOUNT = 64;
const float STEPSIZE = 0.0025f;
const float DESCRIPTORSCALE = 8.f;
const bool BOOLSTEPPING = false;
const float MAXVARIANCEINPIXELS = 3.0;
const float MIPSCALE = 0.5;
const unsigned int KEYPOINTCOUNT = 1000;
const float MARGIN = 0.1;
int randomLikeIndex = 0;

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

const double randomLike(const int index) {
    //return float(rand()) / float(RAND_MAX);
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (double)(b & 0xffff) / 0x10000;
}

cv::Mat drawPoly(cv::Mat& canvas, const cv::Mat& image, const float x0, const float y0, const float x1, const float y1, const float x2, const float y2, const float x3, const float y3) {
    cv::Point2f source[4];
    cv::Point2f dest[4];
    const float sx1 = image.cols;
    const float sy1 = image.rows;
    source[0] = { 0,0 };
    source[1] = { sx1,0 };
    source[2] = { sx1,sy1 };
    source[3] = { 0,sy1 };
    dest[0] = { x0, y0 };
    dest[1] = { x1, y1 };
    dest[2] = { x2, y2 };
    dest[3] = { x3, y3 };
    cv::Mat mat = cv::getPerspectiveTransform(source, dest);
    cv::warpPerspective(image, canvas, mat, cv::Size(canvas.cols, canvas.rows), cv::INTER_AREA);
    return mat;
}

KeyPoint getPointInPoly(float x, float y, const cv::Mat& perspective) {
    float xc = x * perspective.at<double>(0, 0) + y * perspective.at<double>(0, 1) + perspective.at<double>(0, 2);
    float yc = x * perspective.at<double>(1, 0) + y * perspective.at<double>(1, 1) + perspective.at<double>(1, 2);
    float zc = x * perspective.at<double>(2, 0) + y * perspective.at<double>(2, 1) + perspective.at<double>(2, 2);
    KeyPoint r;
    r.x = xc / zc;
    r.y = yc / zc;
    return r;
}

std::pair<KeyPoint,float> trackPoint(const cv::Mat &sourceMat, const cv::Mat& destMat, const KeyPoint &keyPointSource) {
    std::vector<cv::Mat> mipmapsSource = mipMaps(sourceMat);
    std::vector<cv::Mat> mipmapsDest = mipMaps(destMat);

    std::vector<Descriptor> searchForDescriptors;
    searchForDescriptors.resize(mipmapsSource.size());
    for (int i = mipmapsSource.size()-1; i >= 0; i--) {
        const float descriptorScale = 1 << i;
        const float mipScale = powf(MIPSCALE, float(i));
        const int width = mipmapsSource[i].cols;
        const int height = mipmapsSource[i].rows;
        sampleDescriptor(keyPointSource, searchForDescriptors[i], mipmapsSource[i].data, descriptorScale, width, height, mipScale);
    }

    KeyPoint keyPoint;
    KeyPoint varianceKeyPoint;
    for (int v = 0; v < 2; v++) {
        KeyPoint kp = { mipmapsDest[0].cols * 0.5, mipmapsDest[0].rows * 0.5 };
        for (int i = mipmapsDest.size() - 1; i >= 0; i--) {
            const int width = mipmapsDest[i].cols;
            const int height = mipmapsDest[i].rows;
            for (int k = 0; k < STEPCOUNT; k++) {
                float descriptorScale = (1 << i);
                const float mipScale = powf(MIPSCALE, float(i));
                const float step = STEPSIZE * descriptorScale;
                descriptorScale *= 1.0 + randomLike(randomLikeIndex++) * SCALEINVARIANCE * 2.f - SCALEINVARIANCE;
                const float angle = (randomLike(randomLikeIndex++) * ROTATIONINVARIANCE * 2.f - ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                Descriptor foundDescriptor;
                sampleDescriptor(kp, foundDescriptor, mipmapsDest[i].data, descriptorScale, width, height, mipScale);
                kp = refineKeyPoint(BOOLSTEPPING, kp, searchForDescriptors[i], foundDescriptor, mipmapsDest[i].data, descriptorScale, angle, step, width, height, mipScale);
            }
        }
        switch (v) {
        case 0: keyPoint = kp;
        case 1: varianceKeyPoint = kp; break;
        }
    }
    const float varianceX = varianceKeyPoint.x - keyPoint.x;
    const float varianceY = varianceKeyPoint.y - keyPoint.y;
    const float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
    return { keyPoint, variance };
}

void testFrame(cv::Mat &image) {
    cv::Mat leftCanvas;
    cv::Mat rightCanvas;
    cv::Mat trackCanvas;
    leftCanvas.create(cv::Size(500, 500), CV_8UC3);
    rightCanvas.create(cv::Size(500, 500), CV_8UC3);
    trackCanvas.create(cv::Size(500, 500), CV_8UC3);
    const float sx = 512;
    const float sy = 512;
    const float kx = 64;
    const float ky = 64;
    KeyPoint d[4] = {
        {randomLike(randomLikeIndex++) * kx,randomLike(randomLikeIndex++) * ky},
        {sx - randomLike(randomLikeIndex++) * kx,randomLike(randomLikeIndex++) * ky},
        {sx - randomLike(randomLikeIndex++) * kx,sy - randomLike(randomLikeIndex++) * ky},
        {randomLike(randomLikeIndex++) * kx,sy - randomLike(randomLikeIndex++) * ky},
    };
    cv::Mat baseImage = leftCanvas.clone(); cv::resize(image,baseImage,cv::Size(sx,sy));
    cv::Mat leftMat = drawPoly(leftCanvas, baseImage, 0, 0, sx, 0, sx, sy, 0, sy);
    cv::Mat rightMat = drawPoly(rightCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);
    cv::Mat trackMat = drawPoly(trackCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);

    std::vector<KeyPoint> keyPointsSource;
    std::vector<KeyPoint> keyPointsGroundTruth;
    keyPointsSource.resize(KEYPOINTCOUNT);
    keyPointsGroundTruth.resize(KEYPOINTCOUNT);
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        keyPointsSource[i].x = randomLike(randomLikeIndex++) * baseImage.cols * (1.f - MARGIN * 2.f) + MARGIN * baseImage.cols;
        keyPointsSource[i].y = randomLike(randomLikeIndex++) * baseImage.rows * (1.f - MARGIN * 2.f) + MARGIN * baseImage.rows;
        KeyPoint k = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, leftMat);
        cv::circle(leftCanvas, cv::Point(k.x, k.y), 2, cv::Scalar(0, 0, 255));
    }
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        KeyPoint k = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, rightMat);
        keyPointsGroundTruth[i] = k;
        cv::circle(rightCanvas, cv::Point(k.x, k.y), 2, cv::Scalar(0, 0, 255));
    }
    int validKeyPoints = 0;
    double error = 0;
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        std::pair<KeyPoint, float> kpv = trackPoint(baseImage, trackCanvas, keyPointsSource[i]);
        KeyPoint k = kpv.first;
        float variance = kpv.second;
        if (variance < MAXVARIANCEINPIXELS) {
            cv::circle(trackCanvas, cv::Point(k.x, k.y), 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(trackCanvas, cv::Point(k.x, k.y), 3, cv::Scalar(255, 255, 255));
            KeyPoint k1 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, leftMat);
            cv::circle(leftCanvas, cv::Point(k1.x, k1.y), 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(leftCanvas, cv::Point(k1.x, k1.y), 3, cv::Scalar(255, 255, 255));
            KeyPoint k2 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, rightMat);
            cv::circle(rightCanvas, cv::Point(k2.x, k2.y), 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(rightCanvas, cv::Point(k2.x, k2.y), 3, cv::Scalar(255, 255, 255));
            cv::circle(rightCanvas, cv::Point(k.x, k.y), 2, cv::Scalar(255, 0, 0), -1);
            validKeyPoints++;
            double pixelErrorX = k2.x - k.x;
            double pixelErrorY = k2.y - k.y;
            double distance = sqrt(pixelErrorX * pixelErrorX + pixelErrorY * pixelErrorY);
            error += sqrt(pixelErrorX * pixelErrorX + pixelErrorY * pixelErrorY);
        }
    }
    cv::Mat canvas;
    cv::hconcat( leftCanvas, rightCanvas, canvas);
    cv::hconcat( canvas, trackCanvas, canvas);
    imshow("canvas", canvas);
    error /= double(validKeyPoints);
    cv::setWindowTitle("canvas", "Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT) + ", Average misplacing in pixels: " + std::to_string(error));
}

int main(int argc, char** argv)
{
    defaultDescriptorShape(DESCRIPTORSCALE);

    cv::Mat benchImage = cv::imread("bench.png", cv::IMREAD_COLOR);
    while (cv::waitKey(100) != 27) {
        testFrame(benchImage);
        cv::waitKey(1);
    }

    return 0;
}
