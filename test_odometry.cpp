// Lucyte Created on: 17.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "refinement.hpp"

const int SEED = 0x13337;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;
const float OUTLIERPERCENTAGE = 20.f;
const float OUTLIEREXPAND = 200.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

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
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& variancePoints, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<KeyPoint>& lastFrameVariancePoints, cv::Mat &overlay) {
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
            float distanceX = lastFrameVariancePoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameVariancePoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            if (distance < maxLength) {
                validKeyPoints++;
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
    }
    for (int y = 0; y < overlay.rows; y++) {
        for (int x = 0; x < overlay.cols; x++) {
            mat.at<cv::Vec3b>(x, y) *= 0.75;
            mat.at<cv::Vec3b>(x, y) += cv::Vec3b(10,10,10);
            if (overlay.at<cv::Vec3b>(x, y)[0] > 128)
            {
                mat.at<cv::Vec3b>(x, y) = cv::Vec3b(0,0,255);
            }
        }
    }
    cv::putText(mat, "Camera path:", cv::Point2f(10, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255));
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

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

cv::Mat overlay(std::vector<cv::Point2f> positions) {
    cv::Point2f maxP, minP;
    int start = positions.size() - 1 - 200;
    if (start < 0) start = 0;
    cv::Mat canvas;
    canvas.create(320, 320, CV_8UC3);
    canvas = cv::Mat::zeros(320, 320, CV_8UC3);
    const float m = 10;
    bool first = true;
    for (int i = positions.size() - 1; i >= start; i--) {
        cv::Point2f& p = positions[i];
        if (first || p.x < minP.x) minP.x = p.x;
        if (first || p.x > maxP.x) maxP.x = p.x;
        if (first || p.y < minP.y) minP.y = p.y;
        if (first || p.y > maxP.y) maxP.y = p.y;
        first = false;
    }
    maxP.x += 0.01;
    maxP.y += 0.01;
    minP.x -= 0.01;
    minP.y -= 0.01;
    for (int i = positions.size() - 1; i >= start + 1; i--) {
        cv::Point2f p1 = positions[i - 1];
        cv::Point2f p2 = positions[i];
        p1.x = (p1.x - minP.x) * (canvas.cols - m * 2) / (maxP.x - minP.x) + m;
        p1.y = (p1.y - minP.y) * (canvas.rows - m * 2) / (maxP.y - minP.y) + m;
        p2.x = (p2.x - minP.x) * (canvas.cols - m * 2) / (maxP.x - minP.x) + m;
        p2.y = (p2.y - minP.y) * (canvas.rows - m * 2) / (maxP.y - minP.y) + m;
        cv::line(canvas, p1, p2, cv::Scalar(255, 255, 255), 2);
    }
    return canvas;
}

int main(int argc, char** argv)
{
    srand(SEED);
    defaultDescriptorShape(DESCRIPTORSCALE);

    cv::Mat mat1 = loadImage(firstFrame);
    mipmaps1 = mipMaps(mat1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

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
    for (int i = mipEnd; i >= 0; i--) {
        const float descriptorScale = 1 << i;
        const float mipScale = powf(MIPSCALE, float(i));
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptors[i].resize(keyPoints.size());
        for (int j = 0; j < keyPoints.size(); ++j) {
            sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps1[i].data, descriptorScale, width, height, mipScale);
        }
    }

    cv::Point2f currentPosition = cv::Point2f(0, 0);
    std::vector<cv::Point2f> positions;
    cv::Mat R, T;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat mask;
    T.create(3, 1, CV_64F);
    R.create(3, 3, CV_64F);
    T = cv::Mat::zeros(3, 1, CV_64F);
    R = cv::Mat::zeros(3, 3, CV_64F);
    R.at<double>(0, 0) = 1;
    R.at<double>(1, 1) = 1;
    R.at<double>(2, 2) = 1;
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<KeyPoint> lastFrameVariancePoints = variancePoints;
        for (int v = 0; v < 2; v++) {
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
                        descriptorScale *= (1.0 + frrand(SCALEINVARIANCE));
                        const float angle = frrand(ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                        Descriptor foundDescriptor;
                        kp = refineKeyPoint(BOOLSTEPPING, kp, searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, angle, step, width, height, mipScale);
                    }
                }
                switch (v) {
                case 0: keyPoints[j] = kp;
                case 1: variancePoints[j] = kp; break;
                }
            }
        }
        if (cv::waitKey(1) == 27)
            break;

        const float maxLength = nonOutlierLength(keyPoints, lastFrameKeyPoints);
        std::vector<cv::Point2f> cvPoints1;
        std::vector<cv::Point2f> cvPoints2;
        for (int i = 0; i < KEYPOINTCOUNT; i++) {
            const double distanceX = keyPoints[i].x - lastFrameKeyPoints[i].x;
            const double distanceY = keyPoints[i].y - lastFrameKeyPoints[i].y;
            const double distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const double lastVarianceX = lastFrameVariancePoints[i].x - lastFrameKeyPoints[i].x;
            const double lastVarianceY = lastFrameVariancePoints[i].y - lastFrameKeyPoints[i].y;
            const double lastVariance = sqrtf(lastVarianceX * lastVarianceX + lastVarianceY * lastVarianceY);
            const double varianceX = variancePoints[i].x - keyPoints[i].x;
            const double varianceY = variancePoints[i].y - keyPoints[i].y;
            const double variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
            if (variance < MAXVARIANCEINPIXELS && lastVariance < MAXVARIANCEINPIXELS && distance < maxLength) {
                cvPoints1.push_back(cv::Point2f(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y));
                cvPoints2.push_back(cv::Point2f(keyPoints[i].x, keyPoints[i].y));
            }
        }

        if (cvPoints1.size() > 4) {
            cv::Mat essentialMatrix = cv::findEssentialMat(cvPoints1, cvPoints2, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);
            cv::Mat rotation, translation;
            cv::recoverPose(essentialMatrix, cvPoints1, cvPoints2, cameraMatrix, rotation, translation, mask);
            const float length = 1.f;
            T = T + length * R * translation;
            R *= rotation;
            currentPosition.x = T.at<double>(0);
            currentPosition.y = T.at<double>(2);
            cv::Point2f position;
            position.x = -currentPosition.x;
            position.y = -currentPosition.y;
            positions.push_back(position);
            cv::Mat odometryOutput = overlay(positions);
            output("keypoints", mat2, keyPoints, variancePoints, lastFrameKeyPoints, lastFrameVariancePoints, odometryOutput);
        }

        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint& k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM)) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                    variancePoints[j] = k;
                }
                float varianceX = variancePoints[j].x - keyPoints[j].x;
                float varianceY = variancePoints[j].y - keyPoints[j].y;
                float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                if (variance >= MAXVARIANCEINPIXELS) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                    variancePoints[j] = k;
                }
            }
        }

        const bool resample = true;
        if (resample) {
            for (int i = mipEnd; i >= 0; i--) {
                const float descriptorScale = 1 << i;
                const float mipScale = powf(MIPSCALE, float(i));
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
#pragma omp parallel for num_threads(32)
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    float varianceX = variancePoints[j].x - keyPoints[j].x;
                    float varianceY = variancePoints[j].y - keyPoints[j].y;
                    float variance = sqrtf(varianceX * varianceX + varianceY * varianceY);
                    if ((!RESAMPLEONVARIANCE) || (variance < RESAMPLEONVARIANCERADIUS))
                        sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, width, height, mipScale);
                }
            }
        }
    }

    return 0;
}
