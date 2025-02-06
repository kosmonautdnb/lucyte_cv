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
    cv::Subdiv2D subdiv = cv::Subdiv2D(cv::Rect(0, 0, image.cols, image.rows));
    validKeyPoints = 0;
    for (int i = 0; i < keyPoints.size(); i++) {
        if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            if (keyPoints[i].x >= 0 && keyPoints[i].y >= 0 && keyPoints[i].x < image.cols && keyPoints[i].y < image.rows) subdiv.insert(cv::Point2f(keyPoints[i].x, keyPoints[i].y));
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
        for (; j < keyPoints.size(); ++j) 
            if ((facetCenters[i].x - keyPoints[j].x) * (facetCenters[i].x - keyPoints[j].x) + (facetCenters[i].y - keyPoints[j].y) * (facetCenters[i].y - keyPoints[j].y) < 1.f) 
                break;
        float distanceX = keyPoints[j].x - lastFrameKeyPoints[j].x;
        float distanceY = keyPoints[j].y - lastFrameKeyPoints[j].y;
        float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
        float l = distance*255.f*0.025f;
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
    int mipEnd = (int)floorf(MIPEND * (float)(mipmaps1.size() - 1));

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    keyPoints.resize(KEYPOINTCOUNT);
    errors.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2((float)mipmaps1[0].cols);
        keyPoints[i].y = frrand2((float)mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(keyPoints);

    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());
    sampleDescriptors_openCL(mipEnd, searchForDescriptors, 1.f, MIPSCALE);
    uploadDescriptors_openCL(mipEnd, searchForDescriptors);
    refineKeyPoints_openCL(keyPoints, errors, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

    for (int steps = firstFrame+1; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        uploadMipMaps_openCL(mipMaps(mipmaps2));
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;

        refineKeyPoints_openCL(keyPoints, errors, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

        cv::Mat v  = output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27)
            break;
        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint& k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM) || (errors[j] >= MAXVARIANCEINPIXELS)) {
                    k.x = frrand2((float)width);
                    k.y = frrand2((float)height);
                    errors[j] = 0;
                }
            }
        }

        uploadKeyPoints_openCL(keyPoints);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            sampleDescriptors_openCL(mipEnd, resampledDescriptors, 1.f, MIPSCALE);
            for (int i = mipEnd; i >= 0; i--)
                for (int j = (int)keyPoints.size() - 1; j >= 0; j--)
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS))
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
            uploadDescriptors_openCL(mipEnd, searchForDescriptors);
        }
    }

    if (outputVideo) video.release();

    return 0;
}
