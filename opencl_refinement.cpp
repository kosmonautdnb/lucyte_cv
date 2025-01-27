// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "refinement.hpp"
#include <opencv2/opencv.hpp>
// not optimized, yet
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif

extern bool ONLYVALID;
extern int DESCRIPTORSIZE;

#define CLBLOCKING CL_TRUE
cl::Platform openCLPlatform;
cl::Device openCLDevice;
cl::Context openCLContext; 
cl::Program openCLProgram;
cl::CommandQueue openCLQueue;
cl::Image2D openCLMipMaps[16];
cl::Buffer openCLMipMapWidths;
cl::Buffer openCLMipMapHeights;
std::vector<unsigned char *> openCLMipMapPointers;
cl::Buffer openCLDescriptors1;
cl::Buffer openCLDescriptors2;
cl::Buffer openCLKeyPointsX;
cl::Buffer openCLKeyPointsY;
cl::Buffer openCLNewKeyPointsX;
cl::Buffer openCLNewKeyPointsY;
cl::Buffer openCLNewVariancePointsX;
cl::Buffer openCLNewVariancePointsY;
int openCLInt[10];
float openCLFloat[10];
std::vector<float> openCLKpx;
std::vector<float> openCLKpy;
cl::Buffer openCLInts;
cl::Buffer openCLFloats;
std::vector<cl::Buffer> openCLBits;
std::vector<cl::Buffer> openCLValid;
cl::Buffer openCLBitsFull;
cl::Buffer openCLValidFull;
cl::Kernel sampleDescriptor_cl;
cl::Kernel refineKeyPoints_cl;
std::vector<unsigned int> openCLDescriptorBits;
std::vector<unsigned int> openCLDescriptorValid;
std::vector<unsigned int> openCLDescriptorBitsFull;
std::vector<unsigned int> openCLDescriptorValidFull;
cl::Buffer openCLDebug;
float clDebug[10];

void initOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        printf(" No OpenCL platforms found. Check OpenCL installation!\n");
        exit(1);
    }
    for (int i = 0; i < platforms.size(); i++)
        printf("Available openCL platform: %s\n", platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
    openCLPlatform = platforms[0];
    printf("- Using openCL platform: %s\n", openCLPlatform.getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    openCLPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        printf(" No OpenCL devices found. Check OpenCL installation!\n");
        exit(1);
    }
    for (int i = 0; i < devices.size(); i++)
        printf("Available openCL device: %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str());
    openCLDevice = devices[0];
    printf("- Using openCL device: %s\n", openCLDevice.getInfo<CL_DEVICE_NAME>().c_str());

    openCLContext = cl::Context({ openCLDevice });
    openCLQueue = cl::CommandQueue(openCLContext, openCLDevice);

    std::string sampleDescriptor_kernel =
        "   constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;"
        "\n"
        "   inline float tex(const image2d_t s, const float2 coord) {\n"
        "         return read_imagef(s,sampler,coord).x;\n"
        "   }\n"
        "\n"
        "   void kernel sampleDescriptor_kernel(global const int* ints, global const float* floats,\n"
        "                                global const float *keyPointsX, global const float *keyPointsY,\n"
        "                                image2d_t s,\n"
        "                                constant const float2* descriptors1, constant const float2* descriptors2,\n"
        "                                global unsigned int* dBits, global unsigned int* dValid, global float *debug) {\n"
        "       const int DESCRIPTORSIZE = ints[1];\n"
        "       const int width = ints[2];\n"
        "       const int height = ints[3];\n"
        "       const float descriptorScale = floats[0];\n"
        "       const float mipScale = floats[1];\n"
        "\n"
        "       const float descriptorSize = descriptorScale * mipScale;\n"
        "       const int keyPointIndex = get_global_id(0);\n"
        "\n"
        "       const float2 kp = (float2)(keyPointsX[keyPointIndex] * mipScale,keyPointsY[keyPointIndex] * mipScale);\n"
        "       unsigned int b = 0, v = 0;"
        "       for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
        "       {\n"
        "           const float l1 = tex(s, kp + descriptors1[i] * descriptorSize);\n"
        "           const float l2 = tex(s, kp + descriptors2[i] * descriptorSize);\n"
        "\n"
        "           if ((i & 31)==0) {\n"
        "               b=0;v=0;\n"
        "           }\n"
        "           b |= (l2 < l1 ? 1 : 0) << (i & 31);\n"
        "           v |= ((l1 < 0 || l2 < 0) ? 0 : 1) << (i & 31);\n"
        "           dBits[(i+keyPointIndex*DESCRIPTORSIZE)/32]=b;"
        "           dValid[(i+keyPointIndex*DESCRIPTORSIZE)/32]=v;"
        "       }\n"
        "   }\n"
        "\n"
        "   const float randomLike(const int index) {"
        "       int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);\n"
        "       b = b ^ (b << 8) ^ (b * 23);\n"
        "       b >>= 3;\n"
        "       return (float)(b & 0xffff) / (float)0x10000;\n"
        "   }\n"
        "\n"
        "   float2 refineKeyPoint( const image2d_t s, global unsigned int* dBits, global unsigned int* dValid, const int keyPointIndex, const float2 kpxy, \n"
        "                           const float mipScale, const float descriptorScale, const float angle, const float step, const int DESCRIPTORSIZE, \n"
        "                           const int ONLYVALID,const int width,const int height,const int stepping,\n"
        "                           constant const float2* descriptors1, constant const float2* descriptors2,\n"
        "                           global float *debug) {\n"
        "       int dIndex = keyPointIndex * DESCRIPTORSIZE;\n"
        "       const float2 kp = kpxy * mipScale;\n"
        "       const float descriptorSize = mipScale * descriptorScale;\n"
        "       const float sina = sin(angle);\n"
        "       const float cosa = cos(angle);\n"
        "       const float2 cosid = (float2)(cosa, sina) * descriptorSize;\n"
        "       const float2 nsicd = (float2)(-sina, cosa) * descriptorSize;\n"
        "       const float2 gdx = (float2)(cosa,sina);\n"
        "       const float2 gdy = (float2)(-sina,cosa);\n"
        "       float2 xya = (float2)(0,0);\n"
        "       for (int b = 0; b < DESCRIPTORSIZE; ++b) {\n"
        "           const int db = dIndex + b;\n"
        "           const float2 dp1 = descriptors1[b];"
        "           const float2 dp2 = descriptors2[b];"
        "           const float2 d1 = kp + (float2)(dot(cosid,dp1), dot(nsicd,dp1));\n"
        "           const float2 d2 = kp + (float2)(dot(cosid,dp2), dot(nsicd,dp2));\n"
        "           const float l1 = tex(s, d1);\n"
        "           const float l2 = tex(s, d2);\n"
        "           if (ONLYVALID!=0) {\n"
        "               unsigned char v1 = (dValid[db>>5]>>(db & 31)) & 1;\n"
        "               unsigned char v2 = ((l1 < 0 || l2 < 0) ? 0 : 1);\n"
        "               if ((v1 == 0 || v2 == 0)) {\n"
        "                   continue;\n"
        "               }\n"
        "           }\n"
        "           const unsigned char b2 = (l2 < l1 ? 1 : 0);\n"
        "           if (b2 != ( (dBits[db>>5]>>(db & 31)) & 1 )) {\n"
        "               float2  gr  = (float2)( tex(s, d2 - gdx) - tex(s, d2 + gdx) , tex(s, d2 - gdy) - tex(s, d2 + gdy) )\n" 
        "                            -(float2)( tex(s, d1 - gdx) - tex(s, d1 + gdx) , tex(s, d1 - gdy) - tex(s, d1 + gdy) ); \n"
        "               if (b2 != 0) gr = -gr;\n"
        "               if (stepping != 0) gr = (float2)((gr.x == 0.f) ? 0.f : (gr.x > 0.f) ? 1.f : -1.f, (gr.y == 0.f) ? 0.f : (gr.y > 0.f) ? 1.f : -1.f);\n"
        "               xya += gr * fast_length(d2 - d1);\n"
        "           }\n"
        "       }\n"
        "       xya = normalize(xya) * (descriptorScale * step);\n"
        "       float2 r = kpxy + (float2)(cosa * xya.x + sina * xya.y, sina * xya.x + cosa * xya.y);\n"
        "       const bool clipping = false; if (clipping) {\n"
        "           const int w = (int)floor((float)width / mipScale);\n"
        "           const int h = (int)floor((float)height / mipScale);\n"
        "           if (r.x < 0) r.x = 0;\n"
        "           if (r.y < 0) r.y = 0;\n"
        "           if (r.x >= w - 1) r.x = w - 1;\n"
        "           if (r.y >= h - 1) r.y = h - 1;\n"
        "       }\n"
        "       return r;\n"
        "   }\n"
        "\n"
        "   void kernel refineKeyPoints_kernel(global const int* ints, global const float* floats,\n"
        "                        global const float *keyPointsX, global const float *keyPointsY,\n"
        "                        const image2d_t s0,\n"
        "                        const image2d_t s1,\n"
        "                        const image2d_t s2,\n"
        "                        const image2d_t s3,\n"
        "                        const image2d_t s4,\n"
        "                        const image2d_t s5,\n"
        "                        const image2d_t s6,\n"
        "                        const image2d_t s7,\n"
        "                        const image2d_t s8,\n"
        "                        const image2d_t s9,\n"
        "                        const image2d_t s10,\n"
        "                        const image2d_t s11,\n"
        "                        const image2d_t s12,\n"
        "                        const image2d_t s13,\n"
        "                        const image2d_t s14,\n"
        "                        const image2d_t s15,\n"
        "                        global const unsigned int* dBits0,\n"
        "                        global const unsigned int* dBits1,\n"
        "                        global const unsigned int* dBits2,\n"
        "                        global const unsigned int* dBits3,\n"
        "                        global const unsigned int* dBits4,\n"
        "                        global const unsigned int* dBits5,\n"
        "                        global const unsigned int* dBits6,\n"
        "                        global const unsigned int* dBits7,\n"
        "                        global const unsigned int* dBits8,\n"
        "                        global const unsigned int* dBits9,\n"
        "                        global const unsigned int* dBits10,\n"
        "                        global const unsigned int* dBits11,\n"
        "                        global const unsigned int* dBits12,\n"
        "                        global const unsigned int* dBits13,\n"
        "                        global const unsigned int* dBits14,\n"
        "                        global const unsigned int* dBits15,\n"
        "                        global const unsigned int* dValid0,\n"
        "                        global const unsigned int* dValid1,\n"
        "                        global const unsigned int* dValid2,\n"
        "                        global const unsigned int* dValid3,\n"
        "                        global const unsigned int* dValid4,\n"
        "                        global const unsigned int* dValid5,\n"
        "                        global const unsigned int* dValid6,\n"
        "                        global const unsigned int* dValid7,\n"
        "                        global const unsigned int* dValid8,\n"
        "                        global const unsigned int* dValid9,\n"
        "                        global const unsigned int* dValid10,\n"
        "                        global const unsigned int* dValid11,\n"
        "                        global const unsigned int* dValid12,\n"
        "                        global const unsigned int* dValid13,\n"
        "                        global const unsigned int* dValid14,\n"
        "                        global const unsigned int* dValid15,\n"
        "                        global const int *widths,\n"
        "                        global const int *heights,\n"
        "                        constant const float2* descriptors1, constant const float2* descriptors2,\n"
        "                        global float *dKeyPointsX, global float *dKeyPointsY,\n"
        "                        global float *dVariancePointsX, global float *dVariancePointsY,\n"
        "                        global float *debug) {\n"
        "       global const unsigned int *dBits[16];\n"
        "       global const unsigned int *dValid[16];\n"
        "       dBits[0] = dBits0;\n"
        "       dBits[1] = dBits1;\n"
        "       dBits[2] = dBits2;\n"
        "       dBits[3] = dBits3;\n"
        "       dBits[4] = dBits4;\n"
        "       dBits[5] = dBits5;\n"
        "       dBits[6] = dBits6;\n"
        "       dBits[7] = dBits7;\n"
        "       dBits[8] = dBits8;\n"
        "       dBits[9] = dBits9;\n"
        "       dBits[10] = dBits10;\n"
        "       dBits[11] = dBits11;\n"
        "       dBits[12] = dBits12;\n"
        "       dBits[13] = dBits13;\n"
        "       dBits[14] = dBits14;\n"
        "       dBits[15] = dBits15;\n"
        "       dValid[0] = dValid0;\n"
        "       dValid[1] = dValid1;\n"
        "       dValid[2] = dValid2;\n"
        "       dValid[3] = dValid3;\n"
        "       dValid[4] = dValid4;\n"
        "       dValid[5] = dValid5;\n"
        "       dValid[6] = dValid6;\n"
        "       dValid[7] = dValid7;\n"
        "       dValid[8] = dValid8;\n"
        "       dValid[9] = dValid9;\n"
        "       dValid[10] = dValid10;\n"
        "       dValid[11] = dValid11;\n"
        "       dValid[12] = dValid12;\n"
        "       dValid[13] = dValid13;\n"
        "       dValid[14] = dValid14;\n"
        "       dValid[15] = dValid15;\n"
        "       const int KEYPOINTCOUNT = ints[0];\n"
        "       const int DESCRIPTORSIZE = ints[1];\n"
        "       const int mipEnd = ints[2];\n"
        "       const int STEPCOUNT = ints[3];\n"
        "       const int ONLYVALID = ints[4];\n"
        "       const int stepping = ints[5];\n"
        "       const float MIPSCALE = floats[0];\n"
        "       const float STEPSIZE = floats[1];\n"
        "       const float SCALEINVARIANCE = floats[2];\n"
        "       const float ROTATIONINVARIANCE = floats[3];\n"
        "       const int v = (get_global_id(0)/KEYPOINTCOUNT) % 2;\n"
        "       const int j = get_global_id(0) % KEYPOINTCOUNT;\n"
        "       float2 kp = (float2)(widths[0] * 0.5, heights[0] * 0.5);\n" //keyPointsX/Y[j];\n"
        "       for (int i = mipEnd; i >= 0; i--) {\n"
        "           const int width = widths[i];\n"
        "           const int height = heights[i];\n"
        "           for (int k = 0; k < STEPCOUNT; k++) {\n"
        "               const float mipScale = pow(MIPSCALE, (float)i);\n"
        "               float descriptorScale = 1.f / mipScale;\n"
        "               const float step = STEPSIZE * descriptorScale;\n"
        "               descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.f - SCALEINVARIANCE;\n"
        "               const float angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.f - ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;\n"
        "               switch(i) {\n"
        "                   case 0:  kp = refineKeyPoint( s0,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 1:  kp = refineKeyPoint( s1,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 2:  kp = refineKeyPoint( s2,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 3:  kp = refineKeyPoint( s3,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 4:  kp = refineKeyPoint( s4,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 5:  kp = refineKeyPoint( s5,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 6:  kp = refineKeyPoint( s6,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 7:  kp = refineKeyPoint( s7,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 8:  kp = refineKeyPoint( s8,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 9:  kp = refineKeyPoint( s9,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 10: kp = refineKeyPoint(s10,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 11: kp = refineKeyPoint(s11,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 12: kp = refineKeyPoint(s12,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 13: kp = refineKeyPoint(s13,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 14: kp = refineKeyPoint(s14,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "                   case 15: kp = refineKeyPoint(s15,dBits[i],dValid[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
        "               };\n"
        "            }\n"
        "       }\n"
        "       switch (v) {\n"
        "           case 0: dKeyPointsX[j] = kp.x; dKeyPointsY[j] = kp.y; break;\n"
        "           case 1: dVariancePointsX[j] = kp.x; dVariancePointsY[j] = kp.y; break;\n"
        "       }\n"
        "   }\n"
        "\n";
    cl::Program::Sources sources;
    sources.push_back({ sampleDescriptor_kernel.c_str(), sampleDescriptor_kernel.length() });
    openCLProgram = cl::Program(openCLContext, sources);
    if (openCLProgram.build({ openCLDevice }) != CL_SUCCESS) {
        printf("Error building: %s\n", openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLDevice).c_str());
        exit(1);
    }
    openCLInts = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLInt));
    openCLFloats = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLFloat));
    openCLDebug = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(clDebug));
    openCLDescriptors1 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*2*MAXDESCRIPTORSIZE);
    openCLDescriptors2 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*2*MAXDESCRIPTORSIZE);
    sampleDescriptor_cl = cl::Kernel(openCLProgram, "sampleDescriptor_kernel");
    refineKeyPoints_cl = cl::Kernel(openCLProgram, "refineKeyPoints_kernel");
}

void uploadMipMaps_openCL(const std::vector<cv::Mat> &mipMaps) {
    if (mipMaps.size() != openCLMipMapPointers.size()) {
        openCLMipMapPointers.resize(mipMaps.size());
        for (int i = 0; i < mipMaps.size(); i++) {
            int success = 0;
            openCLMipMaps[i] = cl::Image2D(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cl::ImageFormat(CL_R, CL_UNORM_INT8), mipMaps[i].cols, mipMaps[i].rows, 0, nullptr, &success);
            if (success != CL_SUCCESS) 
                exit(1);
        }
        openCLMipMapWidths = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * mipMaps.size());
        openCLMipMapHeights = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * mipMaps.size());
    }
    for (int i = 0; i < mipMaps.size(); i++) {
        cl::array<cl::size_type, 2> origin = { 0,0 };
        cl::array<cl::size_type, 2> region = { mipMaps[i].cols, mipMaps[i].rows };
        if (openCLQueue.enqueueWriteImage(openCLMipMaps[i], CLBLOCKING, origin, region, 0, 0, (uchar* const)mipMaps[i].data) != CL_SUCCESS) 
            exit(1);
    }
    for (int i = 0; i < mipMaps.size(); i++) openCLMipMapPointers[i] = mipMaps[i].data;
    std::vector<unsigned int> mipMapWidths;
    std::vector<unsigned int> mipMapHeights;
    mipMapWidths.resize(mipMaps.size());
    mipMapHeights.resize(mipMaps.size());
    for (int i = 0; i < mipMaps.size(); i++) mipMapWidths[i] = mipMaps[i].cols;
    for (int i = 0; i < mipMaps.size(); i++) mipMapHeights[i] = mipMaps[i].rows;
    openCLQueue.enqueueWriteBuffer(openCLMipMapWidths, CLBLOCKING, 0, sizeof(unsigned int) * mipMapWidths.size(), &(mipMapWidths[0]));
    openCLQueue.enqueueWriteBuffer(openCLMipMapHeights, CLBLOCKING, 0, sizeof(unsigned int) * mipMapHeights.size(), &(mipMapHeights[0]));
}

void uploadDescriptorShape_openCL() {
    std::vector<float> descriptors1;
    std::vector<float> descriptors2;
    descriptors1.resize(DESCRIPTORSIZE * 2);
    descriptors2.resize(DESCRIPTORSIZE * 2);
    for (int i = 0; i < DESCRIPTORSIZE; i++) {
        descriptors1[i * 2 + 0] = descriptorsX1[i];
        descriptors1[i * 2 + 1] = descriptorsY1[i];
        descriptors2[i * 2 + 0] = descriptorsX2[i];
        descriptors2[i * 2 + 1] = descriptorsY2[i];
    }
    openCLQueue.enqueueWriteBuffer(openCLDescriptors1, CLBLOCKING, 0, sizeof(float)*2*DESCRIPTORSIZE, &(descriptors1[0]));
    openCLQueue.enqueueWriteBuffer(openCLDescriptors2, CLBLOCKING, 0, sizeof(float)*2*DESCRIPTORSIZE, &(descriptors2[0]));
}

void uploadKeyPoints_openCL(const std::vector<KeyPoint>& keyPoints) {
    if (keyPoints.size() != openCLKpx.size()) {
        openCLKpx.resize(keyPoints.size());
        openCLKpy.resize(keyPoints.size());
        openCLKeyPointsX = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLKeyPointsY = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsX = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsY = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsX = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsY = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
    }
    for (int i = keyPoints.size() - 1; i >= 0; i--) {
        openCLKpx[i] = keyPoints[i].x;
        openCLKpy[i] = keyPoints[i].y;
    }
    openCLQueue.enqueueWriteBuffer(openCLKeyPointsX, CLBLOCKING, 0, sizeof(float)*openCLKpx.size(), &(openCLKpx[0]));
    openCLQueue.enqueueWriteBuffer(openCLKeyPointsY, CLBLOCKING, 0, sizeof(float)*openCLKpy.size(), &(openCLKpy[0]));
}

void uploadDescriptors_openCL(const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    const std::vector<Descriptor>& descriptors = sourceMips[mipMap];

    if (mipMap >= openCLBits.size() || (descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits.size()) {
        if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits.size()) {
            openCLDescriptorBits.resize(descriptors.size() * Descriptor::uint32count);
            openCLDescriptorValid.resize(descriptors.size() * Descriptor::uint32count);
        }
        openCLBits.resize(mipMap + 1);
        openCLValid.resize(mipMap + 1);
        for (int i = 0; i < mipMap + 1; i++) {
            openCLBits[i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBits.size());
            openCLValid[i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorValid.size());
        }
    }

    for (int i = descriptors.size() - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            openCLDescriptorBits[j + i * Descriptor::uint32count] = descriptors[i].bits[j];
            openCLDescriptorValid[j + i * Descriptor::uint32count] = descriptors[i].valid[j];
        }
    }
    openCLQueue.enqueueWriteBuffer(openCLBits[mipMap], CLBLOCKING, 0, sizeof(unsigned int) * openCLDescriptorBits.size(), &(openCLDescriptorBits[0]));
    openCLQueue.enqueueWriteBuffer(openCLValid[mipMap], CLBLOCKING, 0, sizeof(unsigned int) * openCLDescriptorValid.size(), &(openCLDescriptorValid[0]));
}

void sampleDescriptors_openCL(const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale) {
    std::vector<Descriptor>& dest = destMips[mipMap];

    if ((dest.size() * Descriptor::uint32count) != openCLDescriptorBitsFull.size()) {
        openCLDescriptorBitsFull.resize(dest.size() * Descriptor::uint32count);
        openCLDescriptorValidFull.resize(dest.size() * Descriptor::uint32count);
        openCLBitsFull = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBitsFull.size());
        openCLValidFull = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorValidFull.size());
    }

    openCLInt[0] = dest.size();
    openCLInt[1] = DESCRIPTORSIZE;
    openCLInt[2] = width;
    openCLInt[3] = height;
    openCLFloat[0] = descriptorScale;
    openCLFloat[1] = mipScale;
    openCLQueue.enqueueWriteBuffer(openCLInts, CLBLOCKING, 0, sizeof(openCLInt), openCLInt);
    openCLQueue.enqueueWriteBuffer(openCLFloats, CLBLOCKING, 0, sizeof(openCLFloat), openCLFloat);
    sampleDescriptor_cl.setArg(0, openCLInts);
    sampleDescriptor_cl.setArg(1, openCLFloats);
    sampleDescriptor_cl.setArg(2, openCLKeyPointsX);
    sampleDescriptor_cl.setArg(3, openCLKeyPointsY);
    sampleDescriptor_cl.setArg(4, openCLMipMaps[mipMap]);
    sampleDescriptor_cl.setArg(5, openCLDescriptors1);
    sampleDescriptor_cl.setArg(6, openCLDescriptors2);
    sampleDescriptor_cl.setArg(7, openCLBitsFull);
    sampleDescriptor_cl.setArg(8, openCLValidFull);
    sampleDescriptor_cl.setArg(9, openCLDebug);
    openCLQueue.enqueueNDRangeKernel(sampleDescriptor_cl, cl::NullRange, cl::NDRange(openCLKpx.size()), cl::NullRange);
    openCLQueue.enqueueReadBuffer(openCLBitsFull, CLBLOCKING, 0, openCLDescriptorBitsFull.size() * sizeof(unsigned int), &(openCLDescriptorBitsFull[0]));
    openCLQueue.enqueueReadBuffer(openCLValidFull, CLBLOCKING, 0, openCLDescriptorValidFull.size() * sizeof(unsigned int), &(openCLDescriptorValidFull[0]));
    openCLQueue.enqueueReadBuffer(openCLDebug, CLBLOCKING, 0, sizeof(clDebug), clDebug);
    openCLQueue.finish();
    for (int i = 0; i < dest.size(); ++i) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
             dest[i].bits[j] = openCLDescriptorBitsFull[i * Descriptor::uint32count + j];
             dest[i].valid[j] = openCLDescriptorValidFull[i * Descriptor::uint32count + j];
        }
    }
}

void refineKeyPoints_openCL(std::vector<KeyPoint> &destKeyPoints, std::vector<KeyPoint>& destVariancePoints, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE) {
    destVariancePoints.resize(destKeyPoints.size());
    openCLInt[0] = openCLKpx.size();
    openCLInt[1] = DESCRIPTORSIZE;
    openCLInt[2] = mipEnd;
    openCLInt[3] = STEPCOUNT;
    openCLInt[4] = ONLYVALID ? 1 : 0;
    openCLInt[5] = stepping ? 1 : 0;
    openCLFloat[0] = MIPSCALE;
    openCLFloat[1] = STEPSIZE;
    openCLFloat[2] = SCALEINVARIANCE;
    openCLFloat[3] = ROTATIONINVARIANCE;
    openCLQueue.enqueueWriteBuffer(openCLInts, CLBLOCKING, 0, sizeof(openCLInt), openCLInt);
    openCLQueue.enqueueWriteBuffer(openCLFloats, CLBLOCKING, 0, sizeof(openCLFloat), openCLFloat);
    int a = 0;
    refineKeyPoints_cl.setArg(a, openCLInts); a++;
    refineKeyPoints_cl.setArg(a, openCLFloats); a++;
    refineKeyPoints_cl.setArg(a, openCLKeyPointsX); a++;
    refineKeyPoints_cl.setArg(a, openCLKeyPointsY); a++;
    for (int i = 0; i < 16; i++) {
        refineKeyPoints_cl.setArg(a, i < openCLMipMapPointers.size() ? openCLMipMaps[i] : openCLMipMaps[0]); a++;
    }
    for (int i = 0; i < 16; i++) {
        refineKeyPoints_cl.setArg(a, i < openCLBits.size() ? openCLBits[i] : openCLBits[0]); a++;
    }
    for (int i = 0; i < 16; i++) {
        refineKeyPoints_cl.setArg(a, i < openCLValid.size() ? openCLValid[i] : openCLValid[0]); a++;
    }
    refineKeyPoints_cl.setArg(a, openCLMipMapWidths); a++;
    refineKeyPoints_cl.setArg(a, openCLMipMapHeights); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptors1); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptors2); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsX); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsY); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsX); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsY); a++;
    refineKeyPoints_cl.setArg(a, openCLDebug); a++;
    openCLQueue.enqueueNDRangeKernel(refineKeyPoints_cl, cl::NullRange, cl::NDRange(openCLKpx.size() * 2), cl::NullRange);
    std::vector<float> destKeyPointsX; destKeyPointsX.resize(destKeyPoints.size());
    std::vector<float> destKeyPointsY; destKeyPointsY.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsX; destVariancePointsX.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsY; destVariancePointsY.resize(destKeyPoints.size());
    openCLQueue.enqueueReadBuffer(openCLNewKeyPointsX, CLBLOCKING, 0, destKeyPointsX.size() * sizeof(float), &(destKeyPointsX[0]));
    openCLQueue.enqueueReadBuffer(openCLNewKeyPointsY, CLBLOCKING, 0, destKeyPointsY.size() * sizeof(float), &(destKeyPointsY[0]));
    openCLQueue.enqueueReadBuffer(openCLNewVariancePointsX, CLBLOCKING, 0, destVariancePointsX.size() * sizeof(float), &(destVariancePointsX[0]));
    openCLQueue.enqueueReadBuffer(openCLNewVariancePointsY, CLBLOCKING, 0, destVariancePointsY.size() * sizeof(float), &(destVariancePointsY[0]));
    openCLQueue.enqueueReadBuffer(openCLDebug, CLBLOCKING, 0, sizeof(clDebug), clDebug);
    openCLQueue.finish();
    for (int i = 0; i < destKeyPoints.size(); i++) {
        destKeyPoints[i].x = destKeyPointsX[i];
        destKeyPoints[i].y = destKeyPointsY[i];
        destVariancePoints[i].x = destVariancePointsX[i];
        destVariancePoints[i].y = destVariancePointsY[i];
    }
}
