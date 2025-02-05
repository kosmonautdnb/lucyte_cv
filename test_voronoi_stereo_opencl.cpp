// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencl_refinement.hpp"

const int SEED = 0x13337;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 2.5;
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 2.5f;
const float OUTLIERPERCENTAGE = 20.f;
const float OUTLIEREXPAND = 200.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;
std::vector<cv::Mat> mipmaps3;

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

float nonOutlierLength(std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& lastFrameKeyPoints, const float outlierPercent = OUTLIERPERCENTAGE, const float expand = OUTLIEREXPAND) {
    float maxLength = 0.f;
    if (!keyPoints.empty()) {
        std::vector<float> lengths; lengths.resize(keyPoints.size());
        for (int i = 0; i < keyPoints.size(); i++) {
            float dx = keyPoints[i].x - lastFrameKeyPoints[i].x;
            float dy = keyPoints[i].y - lastFrameKeyPoints[i].y;
            lengths[i] = sqrt(dx * dx + dy * dy);
        }
        std::sort(lengths.begin(), lengths.end(), [](const float& a, const float& b)->bool {return a < b; });
        const int cut = int(lengths.size() * (1.0 - outlierPercent * 0.01f));
        maxLength = lengths[cut] + expand;
    }
    return maxLength;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& limage, const cv::Mat& rimage,
    std::vector<KeyPoint>& keyPointsLeft, std::vector<float>& errorsLeft, std::vector<KeyPoint>& lastFrameKeyPointsLeft, std::vector<float>& lastFrameErrorsLeft,
    std::vector<KeyPoint>& keyPointsRight, std::vector<float>& errorsRight, std::vector<KeyPoint>& lastFrameKeyPointsRight, std::vector<float>& lastFrameErrorsRight
    ) {
    cv::Mat mat = limage.clone();
    cv::Subdiv2D subdiv = cv::Subdiv2D(cv::Rect(0, 0, limage.cols, limage.rows));
    validKeyPoints = 0;
    const float maxLength = nonOutlierLength(keyPointsLeft, lastFrameKeyPointsLeft);
    for (int i = 0; i < keyPointsLeft.size(); i++) {
        float distanceX = keyPointsRight[i].x - keyPointsLeft[i].x;
        float distanceY = keyPointsRight[i].y - keyPointsLeft[i].y;
        float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
        if (errorsLeft[i] < MAXVARIANCEINPIXELS && lastFrameErrorsLeft[i] < MAXVARIANCEINPIXELS && errorsRight[i] < MAXVARIANCEINPIXELS && lastFrameErrorsRight[i] < MAXVARIANCEINPIXELS && distance < maxLength) {
            validKeyPoints++;
            if (keyPointsLeft[i].x >= 0 && keyPointsLeft[i].y >= 0 && keyPointsLeft[i].x < limage.cols && keyPointsLeft[i].y < limage.rows) subdiv.insert(cv::Point2f(keyPointsLeft[i].x, keyPointsLeft[i].y));
        }
    }
    std::vector<std::vector<cv::Point2f>> facetList;
    std::vector<cv::Point2f> facetCenters;
    subdiv.getVoronoiFacetList(std::vector<int>(), facetList, facetCenters);
    std::vector<cv::Point> ifacet;
    cv::Mat lbottom = limage.clone();
    cv::Mat rbottom = rimage.clone();
    for (int i = 0; i < facetList.size(); i++) {
        ifacet.resize(facetList[i].size());
        for (size_t j = 0; j < facetList[i].size(); j++)
            ifacet[j] = facetList[i][j]; 
        int j = 0;
        for (; j < keyPointsLeft.size(); ++j) 
            if ((facetCenters[i].x - keyPointsLeft[j].x) * (facetCenters[i].x - keyPointsLeft[j].x) + (facetCenters[i].y - keyPointsLeft[j].y) * (facetCenters[i].y - keyPointsLeft[j].y) < 1.f)
                break;
        float distanceX = keyPointsRight[j].x - keyPointsLeft[j].x;
        float distanceY = keyPointsRight[j].y - keyPointsLeft[j].y;
        float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
        float l = distance*255.f*0.015f;
        if (l > 255.f) l = 255.f;
        float r = l;
        float g = l;
        float b = l;
        cv::fillConvexPoly(mat, ifacet, cv::Scalar(r, g, b));
        {
            float distanceX = lastFrameKeyPointsLeft[j].x - keyPointsLeft[j].x;
            float distanceY = lastFrameKeyPointsLeft[j].y - keyPointsLeft[j].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPointsLeft[j].x - lastFrameKeyPointsLeft[j].x, keyPointsLeft[j].y - lastFrameKeyPointsLeft[j].y);
            const float sx = 2.f;
            const float sy = 4.f;
            cv::line(lbottom, cv::Point(keyPointsLeft[j].x, keyPointsLeft[j].y), cv::Point(keyPointsLeft[j].x - sy * sin(a) - sx * cos(a), keyPointsLeft[j].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(lbottom, cv::Point(keyPointsLeft[j].x, keyPointsLeft[j].y), cv::Point(keyPointsLeft[j].x - sy * sin(a) + sx * cos(a), keyPointsLeft[j].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(lbottom, cv::Point(keyPointsLeft[j].x, keyPointsLeft[j].y), cv::Point(lastFrameKeyPointsLeft[j].x, lastFrameKeyPointsLeft[j].y), cv::Scalar(255, 255, 255));
        }
        {
            float distanceX = lastFrameKeyPointsRight[j].x - keyPointsRight[j].x;
            float distanceY = lastFrameKeyPointsRight[j].y - keyPointsRight[j].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPointsRight[j].x - lastFrameKeyPointsRight[j].x, keyPointsRight[j].y - lastFrameKeyPointsRight[j].y);
            const float sx = 2.f;
            const float sy = 4.f;
            cv::line(rbottom, cv::Point(keyPointsRight[j].x, keyPointsRight[j].y), cv::Point(keyPointsRight[j].x - sy * sin(a) - sx * cos(a), keyPointsRight[j].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(rbottom, cv::Point(keyPointsRight[j].x, keyPointsRight[j].y), cv::Point(keyPointsRight[j].x - sy * sin(a) + sx * cos(a), keyPointsRight[j].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
            cv::line(rbottom, cv::Point(keyPointsRight[j].x, keyPointsRight[j].y), cv::Point(lastFrameKeyPointsRight[j].x, lastFrameKeyPointsRight[j].y), cv::Scalar(255, 255, 255));
        }
    }
    cv::Mat image = limage.clone();
    cv::Scalar color = cv::Scalar(255, 255, 255);
    const int font = cv::HersheyFonts::FONT_HERSHEY_DUPLEX;
    const double fontScale = 0.7;
    const double yp = 30;
    const double xp = 5;
    cv::putText(lbottom, "Left camera", cv::Point(xp, yp), font, fontScale, color);
    cv::putText(rbottom, "Right camera", cv::Point(xp, yp), font, fontScale, color);
    cv::putText(mat, "Depth from features", cv::Point(xp, yp), font, fontScale, color);
    cv::putText(image, "Video", cv::Point(xp, yp), font, fontScale, color);
    cv::hconcat(image, mat, mat);
    cv::hconcat(lbottom, rbottom, lbottom);
    cv::vconcat(mat, lbottom, mat);
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

cv::Mat loadImageLeft(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, stereoFileNamesLeft, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

cv::Mat loadImageRight(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, stereoFileNamesRight, frame);
    cv::Mat r =  cv::imread(buffer, cv::IMREAD_COLOR);
    return r;
}

int main(int argc, char** argv)
{
    initOpenCL();
    srand(SEED);

    cv::Mat mat1 = loadImageLeft(stereoFirstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), stereoOutputVideoFrameRate / double(stereoFrameStep), cv::Size(mat1.cols * 2, mat1.rows * 2), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(mipmaps1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

    std::vector<KeyPoint> keyPointsLeft;
    std::vector<float> errorsLeft;
    std::vector<KeyPoint> keyPointsRight;
    std::vector<float> errorsRight;
    keyPointsLeft.resize(KEYPOINTCOUNT);
    errorsLeft.resize(KEYPOINTCOUNT);
    keyPointsRight.resize(KEYPOINTCOUNT);
    errorsRight.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPointsLeft.size(); ++i) {
        keyPointsLeft[i].x = frrand2(mipmaps1[0].cols);
        keyPointsLeft[i].y = frrand2(mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(keyPointsLeft);

    std::vector<std::vector<Descriptor>> searchForDescriptorsLeft;
    searchForDescriptorsLeft.resize(mipmaps1.size());
    for (int i = mipEnd; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptorsLeft[i].resize(keyPointsLeft.size());
        sampleDescriptors_openCL(i, searchForDescriptorsLeft, descriptorScale, width, height, mipScale);
    }
    uploadDescriptors_openCL(mipEnd, searchForDescriptorsLeft);
    uploadMipMaps_openCL(mipmaps3);
    refineKeyPoints_openCL(keyPointsRight, errorsRight, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
    uploadMipMaps_openCL(mipmaps2);
    refineKeyPoints_openCL(keyPointsLeft, errorsLeft, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

    for (int steps = stereoFirstFrame+1; steps <= stereoLastFrame; steps += stereoFrameStep) {
        std::vector<KeyPoint> lastFrameKeyPointsLeft = keyPointsLeft;
        std::vector<float> lastFrameErrorsLeft = errorsLeft;
        std::vector<KeyPoint> lastFrameKeyPointsRight = keyPointsRight;
        std::vector<float> lastFrameErrorsRight = errorsRight;

        
        cv::Mat mat2 = loadImageLeft(steps);
        cv::Mat mat3 = loadImageRight(steps);
        mipmaps2 = mipMaps(mat2);
        mipmaps3 = mipMaps(mat3);

        uploadMipMaps_openCL(mipmaps3);
        refineKeyPoints_openCL(keyPointsRight, errorsRight, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        uploadMipMaps_openCL(mipmaps2);
        refineKeyPoints_openCL(keyPointsLeft, errorsLeft, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

        cv::Mat v = output("keypoints", mat2, mat3, keyPointsLeft, errorsLeft, lastFrameKeyPointsLeft, lastFrameErrorsLeft,
            keyPointsRight, errorsRight, lastFrameKeyPointsRight, lastFrameErrorsRight);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - stereoFirstFrame) + " of " + std::to_string(stereoLastFrame - stereoFirstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));

        if (cv::waitKey(1) == 27)
            break;

        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = keyPointsLeft.size() - 1; j >= 0; j--) {
                KeyPoint& k = keyPointsLeft[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                const float MARGIN = 50;
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM) || (errorsLeft[j] >= MAXVARIANCEINPIXELS)) {
                    k.x = frrand2(width - MARGIN * 2) + MARGIN;
                    k.y = frrand2(height - MARGIN * 2) + MARGIN;
                    keyPointsRight[j] = k;
                    errorsRight[j] = 0;
                    errorsLeft[j] = 0;
                }
            }
        }

        uploadKeyPoints_openCL(keyPointsLeft);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            resampledDescriptors.resize(mipmaps2.size());
            for (int i = mipEnd; i >= 0; i--) {
                const float mipScale = powf(MIPSCALE, float(i));
                const float descriptorScale = 1.f / mipScale;
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
                resampledDescriptors[i].resize(keyPointsLeft.size());
                sampleDescriptors_openCL(i, resampledDescriptors, descriptorScale, width, height, mipScale);
                for (int j = keyPointsLeft.size() - 1; j >= 0; j--) {
                    if ((!RESAMPLEONVARIANCE) || (errorsLeft[j] < RESAMPLEONVARIANCERADIUS)) {
                        searchForDescriptorsLeft[i][j] = resampledDescriptors[i][j];
                    }
                }
            }
            uploadDescriptors_openCL(mipEnd, searchForDescriptorsLeft);
        }
    }

    if (outputVideo) video.release();

    return 0;
}
