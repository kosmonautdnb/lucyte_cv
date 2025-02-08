// Lucyte Created on: 03.02.2025 by Stefan Mader
#pragma once
#include "refinement.hpp"
#include <vector>

void initOpenGL();
void uploadDescriptorShape_openGL();
void uploadMipMaps_openGL(int mipmapsId, const MipMap& baseLevel);
void uploadKeyPoints_openGL(int keyPointsId, const std::vector<KeyPoint>& keyPoints);
void uploadDescriptors_openGL(int descriptorsId, int mipLevels, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int mipLevels, std::vector<std::vector<Descriptor>>& destMips, float DESCRIPTORSCALE2, float MIPSCALE);
void refineKeyPoints_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int keyPointCount, int mipLevels, int STEPCOUNT, bool stepping, float MIPSCALE, float STEPSIZE, float SCALEINVARIANCE, float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors);

void glNewFrame();
bool glIsCancelled();