// Lucyte Created on: 21.01.2025 by Stefan Mader
#pragma once
#include "refinement.hpp"
#include <vector>

void initOpenCL();

// buffered / non blocking
void uploadDescriptorShape_openCL(const int queueId);
void uploadMipMaps_openCL(const int queueId, const int mipmapsId, const std::vector<MipMap>& mipMaps);
void uploadMipMaps_openCL_waitfor(const int queueId, const int mipmapsId);
void uploadKeyPoints_openCL(const int queueId, const int keyPointsId, const std::vector<KeyPoint> &keyPoints);
void uploadKeyPoints_openCL_waitfor(const int queueId, const int keyPointsId);
void uploadDescriptors_openCL(const int queueId, const int descriptorsId, const int mipLevels, const std::vector<std::vector<Descriptor>>& sourceMips);
void uploadDescriptors_openCL_waitfor(const int queueId, const int descriptorsId, const int mipLevels);
void sampleDescriptors_openCL(const int queueId, const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipLevels, const float DESCRIPTORSCALE2, const float MIPSCALE);
void sampleDescriptors_openCL_waitfor(const int queueId, const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipLevels, std::vector<std::vector<Descriptor>>& destMips);
void refineKeyPoints_openCL(const int queueId, const int keyPointsId, const int descriptorsId, const int mipmapsId, const int keyPointCount, const int mipLevels, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);
void refineKeyPoints_openCL_waitfor(const int queueId, const int keyPointsId, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors);



// blocking
void uploadDescriptorShape_openCL();
void uploadMipMaps_openCL(const std::vector<MipMap>& mipMaps);
void uploadKeyPoints_openCL(const std::vector<KeyPoint>& keyPoints);
void uploadDescriptors_openCL(const int mipLevels, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openCL(const int mipLevels, std::vector<std::vector<Descriptor>>& destMips, const float DESCRIPTORSCALE2, const float MIPSCALE);
void refineKeyPoints_openCL(std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors, const int mipLevels, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);


