// Lucyte Created on: 29.01.2025 by Stefan Mader
#pragma once
#define KEYPOINTCOUNT 10000

#define DESCRIPTORSHAPE_64 0
#define DESCRIPTORSHAPE_38 1
#define DESCRIPTORSHAPE_SPIRAL 2
#define DESCRIPTORSHAPE_CROSSES 3
#define DESCRIPTORSHAPE DESCRIPTORSHAPE_SPIRAL


#if DESCRIPTORSHAPE == 0
#define DESCRIPTORSIZE 64
#define defaultDescriptorShape defaultDescriptorShape64
#endif

#if DESCRIPTORSHAPE == 1
#define DESCRIPTORSIZE 38
#define defaultDescriptorShape defaultDescriptorShape38
#endif

#if DESCRIPTORSHAPE == 2
#define DESCRIPTORSIZE 128
#define defaultDescriptorShape(__RAD__) defaultDescriptorShapeSpiral(__RAD__, DESCRIPTORSIZE);
#endif

#if DESCRIPTORSHAPE == 3
#define DESCRIPTORSIZE 96
#define defaultDescriptorShape(__RAD__) defaultDescriptorShapeCrosses(__RAD__, DESCRIPTORSIZE);
#endif
