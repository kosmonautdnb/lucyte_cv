// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "refinement.hpp"
#include "opencl_refinement.hpp"

const int SEED = 0x13337;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;

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

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, 
    std::vector<KeyPoint>& keyPointsLeft, std::vector<KeyPoint>& variancePointsLeft, std::vector<KeyPoint>& lastFrameKeyPointsLeft, std::vector<KeyPoint>& lastFrameVariancePointsLeft,
    std::vector<KeyPoint>& keyPointsRight, std::vector<KeyPoint>& variancePointsRight, std::vector<KeyPoint>& lastFrameKeyPointsRight, std::vector<KeyPoint>& lastFrameVariancePointsRight
    ) {
    cv::Mat mat = image.clone();
    cv::Subdiv2D subdiv = cv::Subdiv2D(cv::Rect(0, 0, image.cols, image.rows));
    validKeyPoints = 0;
    for (int i = 0; i < keyPointsLeft.size(); i++) {
        float varianceLeftX = variancePointsLeft[i].x - keyPointsLeft[i].x;
        float varianceLeftY = variancePointsLeft[i].y - keyPointsLeft[i].y;
        float varianceLeft = sqrtf(varianceLeftX * varianceLeftX + varianceLeftY * varianceLeftY);
        float lastFrameVarianceLeftX = lastFrameVariancePointsLeft[i].x - lastFrameKeyPointsLeft[i].x;
        float lastFrameVarianceLeftY = lastFrameVariancePointsLeft[i].y - lastFrameKeyPointsLeft[i].y;
        float lastFrameVarianceLeft = sqrtf(lastFrameVarianceLeftX * lastFrameVarianceLeftX + lastFrameVarianceLeftY * lastFrameVarianceLeftY);
        float varianceRightX = variancePointsRight[i].x - keyPointsRight[i].x;
        float varianceRightY = variancePointsRight[i].y - keyPointsRight[i].y;
        float varianceRight = sqrtf(varianceRightX * varianceRightX + varianceRightY * varianceRightY);
        float lastFrameVarianceRightX = lastFrameVariancePointsRight[i].x - lastFrameKeyPointsRight[i].x;
        float lastFrameVarianceRightY = lastFrameVariancePointsRight[i].y - lastFrameKeyPointsRight[i].y;
        float lastFrameVarianceRight = sqrtf(lastFrameVarianceRightX * lastFrameVarianceRightX + lastFrameVarianceRightY * lastFrameVarianceRightY);
        if (varianceLeft < MAXVARIANCEINPIXELS && lastFrameVarianceLeft < MAXVARIANCEINPIXELS && varianceRight < MAXVARIANCEINPIXELS && lastFrameVarianceRight < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            if (keyPointsLeft[i].x >= 0 && keyPointsLeft[i].y >= 0 && keyPointsLeft[i].x < image.cols && keyPointsLeft[i].y < image.rows) subdiv.insert(cv::Point2f(keyPointsLeft[i].x, keyPointsLeft[i].y));
        }
    }
    std::vector<std::vector<cv::Point2f>> facetList;
    std::vector<cv::Point2f> facetCenters;
    subdiv.getVoronoiFacetList(std::vector<int>(), facetList, facetCenters);
    std::vector<cv::Point> ifacet;
    for (int i = 0; i < facetList.size(); i++) {
        ifacet.resize(facetList[i].size());
        for (size_t j = 0; j < facetList[i].size(); j++)
            ifacet[j] = facetList[i][j]; 
        int j = 0;
        for (; j < keyPointsLeft.size(); ++j) 
            if ((facetCenters[i].x - keyPointsLeft[j].x) * (facetCenters[i].x - keyPointsLeft[j].x) + (facetCenters[i].y - keyPointsLeft[j].y) * (facetCenters[i].y - keyPointsLeft[j].y) < 1.f)
                break;
        float distanceX = keyPointsRight[j].x - keyPointsRight[j].x;
        float distanceY = keyPointsRight[j].y - keyPointsRight[j].y;
        float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
        float l = distance*255.f*0.1f;
        if (l > 255.f) l = 255.f;
        float r = l;
        float g = l;
        float b = l;
        cv::fillConvexPoly(mat, ifacet, cv::Scalar(r, g, b));
    }
    mat += image;
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
    //cv::Mat affine;
    //affine.create(2, 3, CV_32F);
    //affine = cv::Mat::zeros(2, 3, CV_32F);
    //float disparity = 20;
    //affine.at<float>(0, 0) = 1;
    //affine.at<float>(0, 1) = 0;
    //affine.at<float>(0, 2) = disparity;
    //affine.at<float>(1, 0) = 0;
    //affine.at<float>(1, 1) = 1;
    //affine.at<float>(1, 2) = 0;
    //cv::warpAffine(r, r, affine, cv::Size(r.cols, r.rows));
    return r;
}

int main(int argc, char** argv)
{
    srand(SEED);
    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL();

    cv::Mat mat1 = loadImageLeft(stereoFirstFrame);
    cv::VideoWriter video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(stereoFrameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(mipmaps1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

    std::vector<KeyPoint> keyPointsLeft;
    std::vector<KeyPoint> variancePointsLeft;
    std::vector<KeyPoint> keyPointsRight;
    std::vector<KeyPoint> variancePointsRight;
    keyPointsLeft.resize(KEYPOINTCOUNT);
    variancePointsLeft.resize(KEYPOINTCOUNT);
    keyPointsRight.resize(KEYPOINTCOUNT);
    variancePointsRight.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPointsLeft.size(); ++i) {
        keyPointsLeft[i].x = frrand2(mipmaps1[0].cols);
        keyPointsLeft[i].y = frrand2(mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(keyPointsLeft);

    std::vector<std::vector<Descriptor>> searchForDescriptorsLeft;
    searchForDescriptorsLeft.resize(mipmaps1.size());
    for (int i = mipEnd; i >= 0; i--) {
        const float descriptorScale = 1 << i;
        const float mipScale = powf(MIPSCALE, float(i));
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptorsLeft[i].resize(keyPointsLeft.size());
        uploadDescriptors_openCL(i, searchForDescriptorsLeft);
        sampleDescriptors_openCL(i, searchForDescriptorsLeft, mipmaps1[i].data, descriptorScale, width, height, mipScale);
    }

    long long t0 = _Query_perf_counter();;
    long long t00 = _Query_perf_counter();;
    long long fr = _Query_perf_frequency();
    for (int steps = stereoFirstFrame; steps <= stereoLastFrame; steps += stereoFrameStep) {
        std::vector<KeyPoint> lastFrameKeyPointsLeft = keyPointsLeft;
        std::vector<KeyPoint> lastFrameVariancePointsLeft = variancePointsLeft;
        std::vector<KeyPoint> lastFrameKeyPointsRight = keyPointsRight;
        std::vector<KeyPoint> lastFrameVariancePointsRight = variancePointsRight;

        
        cv::Mat mat2 = loadImageLeft(steps);
        cv::Mat mat3 = loadImageRight(steps);
        mipmaps2 = mipMaps(mat2);
        mipmaps3 = mipMaps(mat3);

        t00 = _Query_perf_counter();
        uploadMipMaps_openCL(mipmaps3);
        refineKeyPoints_openCL(keyPointsRight, variancePointsRight, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        uploadMipMaps_openCL(mipmaps2);
        refineKeyPoints_openCL(keyPointsLeft, variancePointsLeft, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
       
        long long t1 = _Query_perf_counter();
        printf("Overall seconds: %f; Feature refinement seconds: %f\n", double(t1 - t0) / fr, double(t1 - t00) / fr);
        t0 = _Query_perf_counter();

        video.write(output("keypoints", mat2, keyPointsLeft, variancePointsLeft, lastFrameKeyPointsLeft, lastFrameVariancePointsLeft, keyPointsRight, variancePointsRight, 
            lastFrameKeyPointsRight, lastFrameVariancePointsRight));
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
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM)) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
                float varianceX = variancePointsLeft[j].x - keyPointsLeft[j].x;
                float varianceY = variancePointsLeft[j].y - keyPointsLeft[j].y;
                float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                if (variance >= MAXVARIANCEINPIXELS) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                }
                variancePointsLeft[j] = k;
            }
        }

        uploadKeyPoints_openCL(keyPointsLeft);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            resampledDescriptors.resize(mipmaps2.size());
            for (int i = mipEnd; i >= 0; i--) {
                const float descriptorScale = 1 << i;
                const float mipScale = powf(MIPSCALE, float(i));
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
                resampledDescriptors[i].resize(keyPointsLeft.size());
                sampleDescriptors_openCL(i, resampledDescriptors, mipmaps2[i].data, descriptorScale, width, height, mipScale);
                for (int j = keyPointsLeft.size() - 1; j >= 0; j--) {
                    float varianceX = variancePointsLeft[j].x - keyPointsLeft[j].x;
                    float varianceY = variancePointsLeft[j].y - keyPointsLeft[j].y;
                    float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                    if ((!RESAMPLEONVARIANCE) || (variance < RESAMPLEONVARIANCERADIUS)) {
                        searchForDescriptorsLeft[i][j] = resampledDescriptors[i][j];
                    }
                }
                uploadDescriptors_openCL(i, searchForDescriptorsLeft);
            }
        }
    }

    video.release();

    return 0;
}
