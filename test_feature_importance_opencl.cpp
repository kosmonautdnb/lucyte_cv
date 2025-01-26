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
const float DESCRIPTIVITYSTEPS = 10;
const float OUTLIERPERCENTAGE = 20.f;
const float OUTLIEREXPAND = 200.f;

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
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& variancePoints, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<KeyPoint>& lastFrameVariancePoints) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    const float maxLength = nonOutlierLength(keyPoints, lastFrameKeyPoints);
    for (int i = 0; i < keyPoints.size(); i++) {
        float varianceX = variancePoints[i].x - keyPoints[i].x;
        float varianceY = variancePoints[i].y - keyPoints[i].y;
        float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
        float lastFrameVarianceX = lastFrameVariancePoints[i].x - lastFrameKeyPoints[i].x;
        float lastFrameVarianceY = lastFrameVariancePoints[i].y - lastFrameKeyPoints[i].y;
        float lastFrameVariance = sqrtf(lastFrameVarianceX * lastFrameVarianceX + lastFrameVarianceY * lastFrameVarianceY);
        if (variance < MAXVARIANCEINPIXELS && lastFrameVariance < MAXVARIANCEINPIXELS) {
            float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            if (distance < maxLength) {
                validKeyPoints++;
                const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
                const float sx = 2.f;
                const float sy = 4.f;
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
                cv::circle(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), 2.0, cv::Scalar(255, 255, 255));
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

float descriptivity(std::vector<cv::Mat>& mipMaps, const KeyPoint& k, const int mipEnd) {
    Descriptor d;
    float h = 0;
    for (int i = mipEnd; i >= 0; i--) {
        const float descriptorScale = 1 << i;
        const float mipScale = powf(MIPSCALE, float(i));
        const int width = mipMaps[i].cols;
        const int height = mipMaps[i].rows;
        h += sampleDescriptor(k, d, mipMaps[i].data, descriptorScale, width, height, mipScale)*mipScale;
    }
    h /= float(mipEnd + 1);
    return h;
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
        float dBest = -1;
        KeyPoint kHere, kBest;
        for (int t = 0; t < DESCRIPTIVITYSTEPS; t++) {
            kHere.x = frrand2(mipmaps1[0].cols);
            kHere.y = frrand2(mipmaps1[0].rows);
            float d = descriptivity(mipmaps1, kHere, mipEnd);
            if (d > dBest) {
                dBest = d;
                kBest = kHere;
            }
        }
        keyPoints[i] = kBest;
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
        sampleDescriptors_openCL(i,searchForDescriptors,mipmaps1[i].data, descriptorScale, width, height, mipScale);
        uploadDescriptors_openCL(i, searchForDescriptors);
    }

    long long t0 = _Query_perf_counter();;
    long long t00 = _Query_perf_counter();;
    long long fr = _Query_perf_frequency();
    int readded = 0;
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        uploadMipMaps_openCL(mipmaps2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<KeyPoint> lastFrameVariancePoints = variancePoints;

        t00 = _Query_perf_counter();
        refineKeyPoints_openCL(keyPoints, variancePoints, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        long long t1 = _Query_perf_counter();
        printf("Overall seconds: %f; Feature refinement seconds: %f\n", double(t1 - t0) / fr, double(t1 - t00) / fr);
        t0 = _Query_perf_counter();

        video.write(output("keypoints", mat2, keyPoints, variancePoints, lastFrameKeyPoints, lastFrameVariancePoints));
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT) + ", readd " + std::to_string(readded));
        readded = 0;
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
                float varianceX = variancePoints[j].x - keyPoints[j].x;
                float varianceY = variancePoints[j].y - keyPoints[j].y;
                float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM) || variance >= MAXVARIANCEINPIXELS) {
                    float dBest = -1;
                    KeyPoint kHere, kBest;
                    for (int t = 0; t < DESCRIPTIVITYSTEPS; t++) {
                        kHere.x = frrand2(mipmaps1[0].cols);
                        kHere.y = frrand2(mipmaps1[0].rows);
                        float d = descriptivity(mipmaps1, kHere, mipEnd);
                        if (d > dBest) {
                            dBest = d;
                            kBest = kHere;
                        }
                    }
                    k = kBest;
                    readded++;
                    variancePoints[j] = k;
                }
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
