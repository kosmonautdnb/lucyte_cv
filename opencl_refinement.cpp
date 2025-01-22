// Created on: 21.01.2025 by Stefan Mader
#include "refinement.hpp"
#include <opencv2/opencv.hpp>
// not optimized, yet
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif

const bool ONLYVALID = false;
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
cl::Buffer openCLDescriptorsX1;
cl::Buffer openCLDescriptorsY1;
cl::Buffer openCLDescriptorsX2;
cl::Buffer openCLDescriptorsY2;
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
cl::Kernel sampleDescriptor_cl;
cl::Kernel refineKeyPoints_cl;
std::vector<unsigned char> openCLDescriptorBits;
std::vector<unsigned char> openCLDescriptorValid;
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
        printf("- Available openCL platform: %s\n", platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
    openCLPlatform = platforms[0];
    printf("Using openCL platform: %s\n", openCLPlatform.getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    openCLPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        printf(" No OpenCL devices found. Check OpenCL installation!\n");
        exit(1);
    }
    for (int i = 0; i < devices.size(); i++)
        printf("- Available openCL device: %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str());
    openCLDevice = devices[0];
    printf("Using openCL device: %s\n", openCLDevice.getInfo<CL_DEVICE_NAME>().c_str());

    openCLContext = cl::Context({ openCLDevice });
    openCLQueue = cl::CommandQueue(openCLContext, openCLDevice);

    std::string sampleDescriptor_kernel =
        "   constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;"
        "   int tex(const image2d_t s, const float x, const float y, const int width, const int height) {\n"
        "         float2 coord; coord.x = x; coord.y = y;\n"
        "         return (int)(read_imagef(s,sampler,coord).r*1024.f);\n"
        "     }\n"
        "\n"
        "   void kernel sampleDescriptor_kernel(global const int* ints, global const float* floats,\n"
        "                                global const float *keyPointsX, global const float *keyPointsY,\n"
        "                                image2d_t s,\n"
        "                                global const float* descriptorsX1, global const float* descriptorsY1, global const float* descriptorsX2, global const float* descriptorsY2,\n"
        "                                global unsigned char* dBits, global unsigned char* dValid, global float *debug) {\n"
        "       const int KEYPOINTCOUNT = ints[0];\n"
        "       const int DESCRIPTORSIZE = ints[1];\n"
        "       const int width = ints[2];\n"
        "       const int height = ints[3];\n"
        "       const float descriptorScale = floats[0];\n"
        "       const float mipScale = floats[1];\n"
        "\n"
        "       const float descriptorSize = descriptorScale * mipScale;\n"
        "       const int keyPointIndex = get_global_id(0)/DESCRIPTORSIZE;\n"
        "       const int descriptorIndex =  get_global_id(0)%DESCRIPTORSIZE;\n"
        "\n"
        "       const float kp2x = keyPointsX[keyPointIndex] * mipScale;\n"
        "       const float kp2y = keyPointsY[keyPointIndex] * mipScale;\n"
        "\n"
        "       const int l1 = tex(s, kp2x + (descriptorSize * descriptorsX1[descriptorIndex]), kp2y + (descriptorSize * descriptorsY1[descriptorIndex]), width, height);\n"
        "       const int l2 = tex(s, kp2x + (descriptorSize * descriptorsX2[descriptorIndex]), kp2y + (descriptorSize * descriptorsY2[descriptorIndex]), width, height);\n"
        "\n"
        "       dBits[descriptorIndex+keyPointIndex*DESCRIPTORSIZE] = (l2 < l1 ? 1 : 0);\n"
        "       dValid[descriptorIndex+keyPointIndex*DESCRIPTORSIZE] = ((l1 < 0 || l2 < 0) ? 0 : 1);\n"
        "   }\n"
        "\n"
        "   const float randomLike(const int index) {"
        "       int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);\n"
        "       b = b ^ (b << 8) ^ (b * 23);\n"
        "       b >>= 3;\n"
        "       return (float)(b & 0xffff) / (float)0x10000;\n"
        "   }\n"
        "\n"
        "   void refineKeyPoint( const image2d_t s, global unsigned char* dBits, global unsigned char* dValid, const int keyPointIndex, const float kpx, const float kpy, \n"
        "                           const float mipScale, const float descriptorScale, const float angle, const float step, const int DESCRIPTORSIZE, \n"
        "                           const int ONLYVALID,const int width,const int height,const int stepping,\n"
        "                           global const float* descriptorsX1, global const float* descriptorsY1, global const float* descriptorsX2, global const float* descriptorsY2,\n"
        "                           float* ret,global float *debug) {\n"
        "       int dIndex = keyPointIndex * DESCRIPTORSIZE;\n"
        "       const float kp2x = kpx * mipScale;\n"
        "       const float kp2y = kpy * mipScale;\n"
        "       const float descriptorSize = mipScale * descriptorScale;\n"
        "       const float sina = sin(angle);\n"
        "       const float cosa = cos(angle);\n"
        "       const float sinad = sina * descriptorSize;\n"
        "       const float cosad = cosa * descriptorSize;\n"
        "       const float gdxx = cosa; const float gdxy = sina;\n"
        "       const float gdyx = -sina; const float gdyy = cosa;\n"
        "       float xa = 0;\n"
        "       float ya = 0;\n"
        "       for (int b = 0; b < DESCRIPTORSIZE; ++b) {\n"
        "           const int l1 = tex(s, kp2x + (descriptorSize * descriptorsX1[b]), kp2y + (descriptorSize * descriptorsY1[b]), width, height);\n"
        "           const int l2 = tex(s, kp2x + (descriptorSize * descriptorsX2[b]), kp2y + (descriptorSize * descriptorsY2[b]), width, height);\n"
        "           int b2 = (l2 < l1 ? 1 : 0);\n"
        "           int v2 = ((l1 < 0 || l2 < 0) ? 0 : 1);\n"
        "           if (ONLYVALID!=0) {\n"
        "               int v1 = dValid[dIndex+b];\n"
        "               if ((v1 == 0 || v2 == 0)) {\n"
        "                   continue;\n"
        "               }\n"
        "           }\n"
        "           int b1 = dBits[dIndex+b];\n"
        "           if (b2 != b1) {\n"
        "               const float d1x = (cosad * descriptorsX1[b] + sinad * descriptorsY1[b]);\n"
        "               const float d1y = (-sinad * descriptorsX1[b] + cosad * descriptorsY1[b]);\n"
        "               const float d2x = (cosad * descriptorsX2[b] + sinad * descriptorsY2[b]);\n"
        "               const float d2y = (-sinad * descriptorsX2[b] + cosad * descriptorsY2[b]);\n"
        "               const int g1x = tex(s, kp2x + d1x - gdxx, kp2y + d1y - gdxy, width, height) - tex(s, kp2x + d1x + gdxx, kp2y + d1y + gdxy, width, height);\n"
        "               const int g1y = tex(s, kp2x + d1x - gdyx, kp2y + d1y - gdyy, width, height) - tex(s, kp2x + d1x + gdyx, kp2y + d1y + gdyy, width, height);\n"
        "               const int g2x = tex(s, kp2x + d2x - gdxx, kp2y + d2y - gdxy, width, height) - tex(s, kp2x + d2x + gdxx, kp2y + d2y + gdxy, width, height);\n"
        "               const int g2y = tex(s, kp2x + d2x - gdyx, kp2y + d2y - gdyy, width, height) - tex(s, kp2x + d2x + gdyx, kp2y + d2y + gdyy, width, height);\n"
        "               int gx = g2x - g1x;\n"
        "               int gy = g2y - g1y;\n"
        "               if (b2 != 0) {\n"
        "                   gx = -gx;\n"
        "                   gy = -gy;\n"
        "               }\n"
        "               if (stepping != 0) {\n"
        "                   gx = (gx == 0) ? 0 : (gx > 0) ? 1 : -1;\n"
        "                   gy = (gy == 0) ? 0 : (gy > 0) ? 1 : -1;\n"
        "               }\n"
        "               const float dx = d2x - d1x;\n"
        "               const float dy = d2y - d1y;\n"
        "               const float l = sqrt(dx * dx + dy * dy);\n"
        "               xa += (float)gx * l;\n"
        "               ya += (float)gy * l;\n"
        "           }\n"
        "       }\n"
        "       float l = xa * xa + ya * ya;\n"
        "       if (l > 0.f) {\n"
        "           l = sqrt(l);\n"
        "           xa /= l;\n"
        "           ya /= l;\n"
        "       }\n"
        "       const float stepSize = descriptorScale * step;\n"
        "       float rx = kpx + (cosa * xa + sina * ya) * stepSize;\n"
        "       float ry = kpy + (-sina * xa + cosa * ya) * stepSize;\n"
        "       const bool clipping = false; if (clipping) {\n"
        "           const int w = (int)floor((float)width / mipScale);\n"
        "           const int h = (int)floor((float)height / mipScale);\n"
        "           if (rx < 0) rx = 0;\n"
        "           if (ry < 0) ry = 0;\n"
        "           if (rx >= w - 1) rx = w - 1;\n"
        "           if (ry >= h - 1) ry = h - 1;\n"
        "       }\n"
        "       ret[0] = rx;\n"
        "       ret[1] = ry;\n"
        "}\n"
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
        "                        global const unsigned char* dBits0,\n"
        "                        global const unsigned char* dBits1,\n"
        "                        global const unsigned char* dBits2,\n"
        "                        global const unsigned char* dBits3,\n"
        "                        global const unsigned char* dBits4,\n"
        "                        global const unsigned char* dBits5,\n"
        "                        global const unsigned char* dBits6,\n"
        "                        global const unsigned char* dBits7,\n"
        "                        global const unsigned char* dBits8,\n"
        "                        global const unsigned char* dBits9,\n"
        "                        global const unsigned char* dBits10,\n"
        "                        global const unsigned char* dBits11,\n"
        "                        global const unsigned char* dBits12,\n"
        "                        global const unsigned char* dBits13,\n"
        "                        global const unsigned char* dBits14,\n"
        "                        global const unsigned char* dBits15,\n"
        "                        global const unsigned char* dValid0,\n"
        "                        global const unsigned char* dValid1,\n"
        "                        global const unsigned char* dValid2,\n"
        "                        global const unsigned char* dValid3,\n"
        "                        global const unsigned char* dValid4,\n"
        "                        global const unsigned char* dValid5,\n"
        "                        global const unsigned char* dValid6,\n"
        "                        global const unsigned char* dValid7,\n"
        "                        global const unsigned char* dValid8,\n"
        "                        global const unsigned char* dValid9,\n"
        "                        global const unsigned char* dValid10,\n"
        "                        global const unsigned char* dValid11,\n"
        "                        global const unsigned char* dValid12,\n"
        "                        global const unsigned char* dValid13,\n"
        "                        global const unsigned char* dValid14,\n"
        "                        global const unsigned char* dValid15,\n"
        "                        global const int *widths,\n"
        "                        global const int *heights,\n"
        "                        global const float* descriptorsX1, global const float* descriptorsY1, global const float* descriptorsX2, global const float* descriptorsY2,\n"
        "                        global float *dKeyPointsX, global float *dKeyPointsY,\n"
        "                        global float *dVariancePointsX, global float *dVariancePointsY,\n"
        "                        global float *debug) {\n"
        "       global const unsigned char *dBits[16];\n"
        "       global const unsigned char *dValid[16];\n"
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
        "       float kpx = widths[0] * 0.5;\n"//keyPointsX[j];\n"
        "       float kpy = heights[0] * 0.5;\n"//keyPointsY[j];\n"
        "       for (int i = mipEnd; i >= 0; i--) {\n"
        "           const int width = widths[i];\n"
        "           const int height = heights[i];\n"
        "           for (int k = 0; k < STEPCOUNT; k++) {\n"
        "               float descriptorScale = (float)(1 << i);\n"
        "               const float mipScale = pow(MIPSCALE, (float)i);\n"
        "               const float step = STEPSIZE * descriptorScale;\n"
        "               descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.f - SCALEINVARIANCE;\n"
        "               const float angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.f - ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;\n"
        "               float ret[2];\n"
        "               switch(i) {\n"
        "                   case 0: refineKeyPoint(s0,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 1: refineKeyPoint(s1,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 2: refineKeyPoint(s2,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 3: refineKeyPoint(s3,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 4: refineKeyPoint(s4,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 5: refineKeyPoint(s5,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 6: refineKeyPoint(s6,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 7: refineKeyPoint(s7,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); debug[0]=tex(s7, 0, 0, width, height); debug[1]=tex(s7, 10.5, 10, width, height); debug[2]=tex(s7, 11, 10, width, height); break;\n"
        "                   case 8: refineKeyPoint(s8,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 9: refineKeyPoint(s9,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 10: refineKeyPoint(s10,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 11: refineKeyPoint(s11,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 12: refineKeyPoint(s12,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 13: refineKeyPoint(s13,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 14: refineKeyPoint(s14,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "                   case 15: refineKeyPoint(s15,dBits[i],dValid[i],j,kpx,kpy,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE,ONLYVALID, width, height, stepping,descriptorsX1, descriptorsY1, descriptorsX2, descriptorsY2, ret, debug); break;\n"
        "               };\n"
        "               kpx = ret[0];\n"
        "               kpy = ret[1];\n"
        "            }\n"
            "   }\n"
        "       switch (v) {\n"
        "           case 0: dKeyPointsX[j] = kpx; dKeyPointsY[j] = kpy; break;\n"
        "           case 1: dVariancePointsX[j] = kpx; dVariancePointsY[j] = kpy; break;\n"
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
    openCLDebug = cl::Buffer(openCLContext, CL_MEM_READ_WRITE, sizeof(clDebug));
    openCLDescriptorsX1 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(descriptorsX1));
    openCLDescriptorsY1 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(descriptorsY1));
    openCLDescriptorsX2 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(descriptorsX2));
    openCLDescriptorsY2 = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(descriptorsY2));
    sampleDescriptor_cl = cl::Kernel(openCLProgram, "sampleDescriptor_kernel");
    refineKeyPoints_cl = cl::Kernel(openCLProgram, "refineKeyPoints_kernel");
}

void uploadMipMaps_openCL(const std::vector<cv::Mat> &mipMaps) {
    if (mipMaps.size() != openCLMipMapPointers.size()) {
        openCLMipMapPointers.resize(mipMaps.size());
        for (int i = 0; i < mipMaps.size(); i++) {
            int success = 0;
            openCLMipMaps[i] = cl::Image2D(openCLContext, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_UNORM_INT8), mipMaps[i].cols, mipMaps[i].rows, 0, nullptr, &success);
            if (success != CL_SUCCESS) 
                exit(1);
        }
        openCLMipMapWidths = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * mipMaps.size());
        openCLMipMapHeights = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * mipMaps.size());
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
    openCLQueue.enqueueWriteBuffer(openCLDescriptorsX1, CLBLOCKING, 0, sizeof(descriptorsX1), descriptorsX1);
    openCLQueue.enqueueWriteBuffer(openCLDescriptorsY1, CLBLOCKING, 0, sizeof(descriptorsY1), descriptorsY1);
    openCLQueue.enqueueWriteBuffer(openCLDescriptorsX2, CLBLOCKING, 0, sizeof(descriptorsX2), descriptorsX2);
    openCLQueue.enqueueWriteBuffer(openCLDescriptorsY2, CLBLOCKING, 0, sizeof(descriptorsY2), descriptorsY2);
}

void uploadKeyPoints_openCL(const std::vector<KeyPoint>& keyPoints) {
    if (keyPoints.size() != openCLKpx.size()) {
        openCLKpx.resize(keyPoints.size());
        openCLKpy.resize(keyPoints.size());
        openCLKeyPointsX = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(float) * keyPoints.size());
        openCLKeyPointsY = cl::Buffer(openCLContext, CL_MEM_READ_ONLY, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsX = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsY = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsX = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsY = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY, sizeof(float) * keyPoints.size());
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
    if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits.size()) {
        openCLDescriptorBits.resize(descriptors.size() * Descriptor::uint32count * 32);
        openCLDescriptorValid.resize(descriptors.size() * Descriptor::uint32count * 32);
        if (mipMap >= openCLBits.size()) {
            openCLBits.resize(mipMap + 1);
            openCLValid.resize(mipMap + 1);
        }
        openCLBits[mipMap] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * openCLDescriptorBits.size());
        openCLValid[mipMap] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * openCLDescriptorValid.size());
    }
    for (int i = descriptors.size() - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            for (int k = 0; k < 32; k++) {
                openCLDescriptorBits[(j + i * Descriptor::uint32count) * 32 + k] = (descriptors[i].bits[j]>>k) & 1;
                openCLDescriptorValid[(j + i * Descriptor::uint32count) * 32 + k] = (descriptors[i].valid[j]>>k) & 1;
            }
        }
    }
    openCLQueue.enqueueWriteBuffer(openCLBits[mipMap], CLBLOCKING, 0, sizeof(unsigned char) * openCLDescriptorBits.size(), &(openCLDescriptorBits[0]));
    openCLQueue.enqueueWriteBuffer(openCLValid[mipMap], CLBLOCKING, 0, sizeof(unsigned char) * openCLDescriptorValid.size(), &(openCLDescriptorValid[0]));
}

void sampleDescriptors_openCL(const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale) {
    std::vector<Descriptor>& dest = destMips[mipMap];
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
    sampleDescriptor_cl.setArg(5, openCLDescriptorsX1);
    sampleDescriptor_cl.setArg(6, openCLDescriptorsY1);
    sampleDescriptor_cl.setArg(7, openCLDescriptorsX2);
    sampleDescriptor_cl.setArg(8, openCLDescriptorsY2);
    sampleDescriptor_cl.setArg(9, openCLBits[mipMap]);
    sampleDescriptor_cl.setArg(10, openCLValid[mipMap]);
    sampleDescriptor_cl.setArg(11, openCLDebug);
    openCLQueue.enqueueNDRangeKernel(sampleDescriptor_cl, cl::NullRange, cl::NDRange(openCLKpx.size() * DESCRIPTORSIZE), cl::NullRange);
    openCLQueue.flush();
    sampleDescriptor_cl();
    openCLQueue.enqueueReadBuffer(openCLBits[mipMap], CLBLOCKING, 0, openCLDescriptorBits.size() * sizeof(unsigned char), &(openCLDescriptorBits[0]));
    openCLQueue.enqueueReadBuffer(openCLValid[mipMap], CLBLOCKING, 0, openCLDescriptorValid.size() * sizeof(unsigned char), &(openCLDescriptorValid[0]));
    openCLQueue.enqueueReadBuffer(openCLDebug, CLBLOCKING, 0, sizeof(clDebug), clDebug);
    openCLQueue.flush();
    for (int i = 0; i < dest.size(); ++i) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            dest[i].bits[j] = 0;
            dest[i].valid[j] = 0;
            for (int k = 0; k < 32; k++) {
                dest[i].bits[j] |= (openCLDescriptorBits[i * Descriptor::uint32count * 32 + j * 32 + k] != 0 ? 1 : 0)<<k;
                dest[i].valid[j] |= (openCLDescriptorValid[i * Descriptor::uint32count * 32 + j * 32 + k] != 0 ? 1 : 0)<<k;
            }
        }
    }
}

void refineKeyPoints_openCL(std::vector<KeyPoint> &destKeyPoints, std::vector<KeyPoint>& destVariancePoints, const int mipEnd, const int STEPCOUNT, const int stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE) {
    destVariancePoints.resize(destKeyPoints.size());
    openCLInt[0] = openCLKpx.size();
    openCLInt[1] = DESCRIPTORSIZE;
    openCLInt[2] = mipEnd;
    openCLInt[3] = STEPCOUNT;
    openCLInt[4] = ONLYVALID;
    openCLInt[5] = stepping;
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
    refineKeyPoints_cl.setArg(a, openCLDescriptorsX1); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptorsY1); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptorsX2); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptorsY2); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsX); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsY); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsX); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsY); a++;
    refineKeyPoints_cl.setArg(a, openCLDebug); a++;
    openCLQueue.enqueueNDRangeKernel(refineKeyPoints_cl, cl::NullRange, cl::NDRange(openCLKpx.size() * 2), cl::NullRange);
    openCLQueue.flush();
    std::vector<float> destKeyPointsX; destKeyPointsX.resize(destKeyPoints.size());
    std::vector<float> destKeyPointsY; destKeyPointsY.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsX; destVariancePointsX.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsY; destVariancePointsY.resize(destKeyPoints.size());
    openCLQueue.enqueueReadBuffer(openCLNewKeyPointsX, CLBLOCKING, 0, destKeyPointsX.size() * sizeof(float), &(destKeyPointsX[0]));
    openCLQueue.enqueueReadBuffer(openCLNewKeyPointsY, CLBLOCKING, 0, destKeyPointsY.size() * sizeof(float), &(destKeyPointsY[0]));
    openCLQueue.enqueueReadBuffer(openCLNewVariancePointsX, CLBLOCKING, 0, destVariancePointsX.size() * sizeof(float), &(destVariancePointsX[0]));
    openCLQueue.enqueueReadBuffer(openCLNewVariancePointsY, CLBLOCKING, 0, destVariancePointsY.size() * sizeof(float), &(destVariancePointsY[0]));
    openCLQueue.enqueueReadBuffer(openCLDebug, CLBLOCKING, 0, sizeof(clDebug), clDebug);
    openCLQueue.flush();
    for (int i = 0; i < destKeyPoints.size(); i++) {
        destKeyPoints[i].x = destKeyPointsX[i];
        destKeyPoints[i].y = destKeyPointsY[i];
        destVariancePoints[i].x = destVariancePointsX[i];
        destVariancePoints[i].y = destVariancePointsY[i];
    }
}
