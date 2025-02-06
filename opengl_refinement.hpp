// Lucyte Created on: 03.02.2025 by Stefan Mader
#pragma once
#include "refinement.hpp"
#include <vector>

void initOpenGL();
void uploadDescriptorShape_openGL();
void uploadMipMaps_openGL(const int mipmapsId, const std::vector<MipMap>& mipMaps);
void uploadKeyPoints_openGL(const int keyPointsId, const std::vector<KeyPoint>& keyPoints);
void uploadDescriptors_openGL(const int descriptorsId, const int mipEnd, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipEnd, std::vector<std::vector<Descriptor>>& destMips, const float DESCRIPTORSCALE2, const float MIPSCALE);
void refineKeyPoints_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int keyPointCount, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors);

void glNewFrame();
bool glIsCancelled();