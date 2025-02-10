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
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<float>& errors, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<float>& lastFrameErrors, cv::Mat& overlay) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    const float maxLength = nonOutlierLength(keyPoints, lastFrameKeyPoints);
    for (int i = 0; i < keyPoints.size(); i++) {
        float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
        float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
        float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
        if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS && distance < maxLength) {
            validKeyPoints++;
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
    for (int y = 0; y < overlay.rows; y++) {
        for (int x = 0; x < overlay.cols; x++) {
            mat.at<cv::Vec3b>(x, y) *= 0.75;
            mat.at<cv::Vec3b>(x, y) += cv::Vec3b(10, 10, 10);
            if (overlay.at<cv::Vec3b>(x, y)[0] > 128)
            {
                mat.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 255);
            }
        }
    }
    cv::putText(mat, "Camera path:", cv::Point2f(10, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255));
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

cv::Mat overlay(std::vector<cv::Point2f> positions) {
    cv::Point2f maxP, minP;
    int start = (int)positions.size() - 1 - 200;
    if (start < 0) start = 0;
    cv::Mat canvas;
    canvas.create(320, 320, CV_8UC3);
    canvas = cv::Mat::zeros(320, 320, CV_8UC3);
    const float m = 10;
    bool first = true;
    for (int i = (int)positions.size() - 1; i >= start; i--) {
        cv::Point2f& p = positions[i];
        if (first || p.x < minP.x) minP.x = p.x;
        if (first || p.x > maxP.x) maxP.x = p.x;
        if (first || p.y < minP.y) minP.y = p.y;
        if (first || p.y > maxP.y) maxP.y = p.y;
        first = false;
    }
    maxP.x += 0.01f;
    maxP.y += 0.01f;
    minP.x -= 0.01f;
    minP.y -= 0.01f;
    for (int i = (int)positions.size() - 1; i >= start + 1; i--) {
        cv::Point2f p1 = positions[i - 1];
        cv::Point2f p2 = positions[i];
        p1.x = (p1.x - minP.x) * (canvas.cols - m * 2.f) / (maxP.x - minP.x) + m;
        p1.y = (p1.y - minP.y) * (canvas.rows - m * 2.f) / (maxP.y - minP.y) + m;
        p2.x = (p2.x - minP.x) * (canvas.cols - m * 2.f) / (maxP.x - minP.x) + m;
        p2.y = (p2.y - minP.y) * (canvas.rows - m * 2.f) / (maxP.y - minP.y) + m;
        cv::line(canvas, p1, p2, cv::Scalar(255, 255, 255), 2);
    }
    return canvas;
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
    int mipLevels = (int)floorf(MIPEND * (float)(mipmaps1.size()));

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
    sampleDescriptors_openCL(mipLevels, searchForDescriptors, 1.f, MIPSCALE);
    uploadDescriptors_openCL(mipLevels, searchForDescriptors);
    refineKeyPoints_openCL(keyPoints, errors, mipLevels, STEPCOUNT, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

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
    for (int steps = firstFrame+1; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        uploadMipMaps_openCL(mipMaps(mipmaps2));
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;

        refineKeyPoints_openCL(keyPoints, errors, mipLevels, STEPCOUNT, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);

        if (cv::waitKey(1) == 27)
            break;

        const float maxLength = nonOutlierLength(keyPoints, lastFrameKeyPoints);
        std::vector<cv::Point2f> cvPoints1;
        std::vector<cv::Point2f> cvPoints2;
        for (int i = 0; i < KEYPOINTCOUNT; i++) {
            const float distanceX = keyPoints[i].x - lastFrameKeyPoints[i].x;
            const float distanceY = keyPoints[i].y - lastFrameKeyPoints[i].y;
            const float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS && distance < maxLength) {
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
            currentPosition.x = (float)T.at<double>(0);
            currentPosition.y = (float)T.at<double>(2);
            cv::Point2f position;
            position.x = -currentPosition.x;
            position.y = -currentPosition.y;
            positions.push_back(position);
            cv::Mat odometryOutput = overlay(positions);
            cv::Mat v = output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors, odometryOutput);
            if (outputVideo) video.write(v);
            cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        }


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
