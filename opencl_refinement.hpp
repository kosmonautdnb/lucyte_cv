// Lucyte Created on: 21.01.2025 by Stefan Mader
#pragma once
#include "refinement.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

void initOpenCL();

// buffered / non blocking
void uploadDescriptorShape_openCL(const int queueId);
void uploadMipMaps_openCL(const int queueId, const int mipmapsId, const std::vector<cv::Mat>& mipMaps);
void uploadMipMaps_openCL_waitfor(const int queueId, const int mipmapsId);
void uploadKeyPoints_openCL(const int queueId, const int keyPointsId, const std::vector<KeyPoint> &keyPoints);
void uploadKeyPoints_openCL_waitfor(const int queueId, const int keyPointsId);
void uploadDescriptors_openCL(const int queueId, const int descriptorsId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips);
void uploadDescriptors_openCL_waitfor(const int queueId, const int descriptorsId, const int mipMap);
void sampleDescriptors_openCL(const int queueId, const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipMap, const int keyPointCount, const float descriptorScale, const int width, const int height, const float mipScale);
void sampleDescriptors_openCL_waitfor(const int queueId, const int descriptorsId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips);
void refineKeyPoints_openCL(const int queueId, const int keyPointsId, const int descriptorsId, const int mipmapsId, const int keyPointCount, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);
void refineKeyPoints_openCL_waitfor(const int queueId, const int keyPointsId, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors);

// blocking
void uploadDescriptorShape_openCL();
void uploadMipMaps_openCL(const std::vector<cv::Mat>& mipMaps);
void uploadKeyPoints_openCL(const std::vector<KeyPoint>& keyPoints);
void uploadDescriptors_openCL(const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openCL(const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const float descriptorScale, const int width, const int height, const float mipScale);
void refineKeyPoints_openCL(std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);


