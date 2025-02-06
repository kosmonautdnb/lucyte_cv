// Lucyte Created on: 24.01.2025 by Stefan Mader
#pragma once

#define DESCRIPTORSHAPE_64 0
#define DESCRIPTORSHAPE_38 1
#define DESCRIPTORSHAPE_SPIRAL 2
#define DESCRIPTORSHAPE_CROSSES 3

#define DESCRIPTORSHAPE DESCRIPTORSHAPE_SPIRAL
#define DESCRIPTORSCALE 5.f

#if DESCRIPTORSHAPE == 0
#define DESCRIPTORSIZE 64
#define defaultDescriptorShape defaultDescriptorShape64(DESCRIPTORSCALE)
#endif

#if DESCRIPTORSHAPE == 1
#define DESCRIPTORSIZE 38
#define defaultDescriptorShape defaultDescriptorShape38(DESCRIPTORSCALE)
#endif

#if DESCRIPTORSHAPE == 2
#define DESCRIPTORSIZE 128
#define defaultDescriptorShape defaultDescriptorShapeSpiral(DESCRIPTORSCALE, DESCRIPTORSIZE);
#endif

#if DESCRIPTORSHAPE == 3
#define DESCRIPTORSIZE 96
#define defaultDescriptorShape defaultDescriptorShapeCrosses(DESCRIPTORSCALE, DESCRIPTORSIZE);
#endif

extern float descriptorsX1[DESCRIPTORSIZE];
extern float descriptorsY1[DESCRIPTORSIZE];
extern float descriptorsX2[DESCRIPTORSIZE];
extern float descriptorsY2[DESCRIPTORSIZE];

void defaultDescriptorShape38(const float rad);
void defaultDescriptorShape64(const float rad);
void defaultDescriptorShapeSpiral(const float rad, const int descriptorSize);
void defaultDescriptorShapeCrosses(const float rad, const int descriptorSize);
