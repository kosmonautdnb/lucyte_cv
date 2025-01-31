// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "opencl_refinement.hpp"
#include <opencv2/opencv.hpp>
// not optimized, yet
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif

#include "constants.hpp"

const int BUFFERCOUNT = 2;

#define CLBLOCKING CL_TRUE
#define CLNONBLOCKING CL_FALSE
static const int MAXMIPMAPS = 32;
cl::Platform openCLPlatform;
cl::Device openCLDevice;
cl::Context openCLContext;
static cl::Program openCLProgram;
static cl::CommandQueue openCLQueue[BUFFERCOUNT];
static cl::Image2D openCLMipMaps[BUFFERCOUNT][MAXMIPMAPS];
static cl::Buffer openCLMipMapWidths[BUFFERCOUNT];
static cl::Buffer openCLMipMapHeights[BUFFERCOUNT];
static cl::Buffer openCLDescriptors1[BUFFERCOUNT];
static cl::Buffer openCLDescriptors2[BUFFERCOUNT];
static cl::Buffer openCLKeyPointsX[BUFFERCOUNT];
static cl::Buffer openCLKeyPointsY[BUFFERCOUNT];
static cl::Buffer openCLNewKeyPointsX[BUFFERCOUNT];
static cl::Buffer openCLNewKeyPointsY[BUFFERCOUNT];
static cl::Buffer openCLNewVariancePointsX[BUFFERCOUNT];
static cl::Buffer openCLNewVariancePointsY[BUFFERCOUNT];
static cl::Buffer openCLBits[BUFFERCOUNT][MAXMIPMAPS];
static cl::Buffer openCLBitsFull[BUFFERCOUNT][MAXMIPMAPS];
static cl::Buffer openCLInts1[BUFFERCOUNT];
static cl::Buffer openCLInts2[BUFFERCOUNT];
static cl::Buffer openCLFloats1[BUFFERCOUNT];
static cl::Buffer openCLFloats2[BUFFERCOUNT];
static cl::Buffer openCLDebug1[BUFFERCOUNT];
static cl::Buffer openCLDebug2[BUFFERCOUNT];
static cl::Kernel sampleDescriptor_cl[BUFFERCOUNT];
static cl::Kernel refineKeyPoints_cl[BUFFERCOUNT];
static int openCLInt1[BUFFERCOUNT][10];
static int openCLInt2[BUFFERCOUNT][10];
static float openCLFloat1[BUFFERCOUNT][10];
static float openCLFloat2[BUFFERCOUNT][10];
static float clDebug1[BUFFERCOUNT][10];
static float clDebug2[BUFFERCOUNT][10];
static std::vector<float> openCLKpx[BUFFERCOUNT];
static std::vector<float> openCLKpy[BUFFERCOUNT];
static std::vector<unsigned int> openCLDescriptorBits[BUFFERCOUNT][MAXMIPMAPS];
static std::vector<unsigned int> openCLDescriptorBitsFull[BUFFERCOUNT][MAXMIPMAPS];
static std::vector<unsigned int> mipMapWidths[BUFFERCOUNT];
static std::vector<unsigned int> mipMapHeights[BUFFERCOUNT];
static cl::Event mipMapUploadEvent[BUFFERCOUNT][MAXMIPMAPS];
static cl::Event keyPointsXEvent[BUFFERCOUNT];
static cl::Event keyPointsYEvent[BUFFERCOUNT];
static cl::Event descriptorEvent[BUFFERCOUNT][MAXMIPMAPS];

static std::string openCV_program =
"   constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;\n"
"\n"
"   inline float tex(const image2d_t s, const float2 coord) {\n"
"         return read_imagef(s,sampler,coord).x;\n"
"   }\n"
"\n"
"   void kernel sampleDescriptor_kernel(global const int* ints, global const float* floats,\n"
"                                global const float *keyPointsX, global const float *keyPointsY,\n"
"                                image2d_t s,\n"
"                                constant const float2* descriptors1, constant const float2* descriptors2,\n"
"                                global unsigned int* dBits, global float *debug) {\n"
"       const int DESCRIPTORSIZE = ints[1];\n"
"       const int width = ints[2];\n"
"       const int height = ints[3];\n"
"       const int DESCRIPTORSIZE2 = ints[4];\n"
"       const float descriptorScale = floats[0];\n"
"       const float mipScale = floats[1];\n"
"\n"
"       const float descriptorSize = descriptorScale * mipScale;\n"
"       const int keyPointIndex = get_global_id(0);\n"
"\n"
"       const float2 kp = (float2)(keyPointsX[keyPointIndex] * mipScale,keyPointsY[keyPointIndex] * mipScale);\n"
"       unsigned int b = 0;\n"
"#pragma unroll 4\n"
"       for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
"       {\n"
"           const float l1 = tex(s, kp + descriptors1[i] * descriptorSize);\n"
"           const float l2 = tex(s, kp + descriptors2[i] * descriptorSize);\n"
"\n"
"           if ((i & 31)==0) b=0;\n"
"           b |= (l2 < l1 ? 1 : 0) << (i & 31);\n"
"           dBits[(i+keyPointIndex*DESCRIPTORSIZE2)>>5]=b;\n"
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
"   float2 refineKeyPoint( const image2d_t s, global unsigned int* dBits, const int keyPointIndex, const float2 kpxy, \n"
"                           const float mipScale, const float descriptorScale, const float angle, const float step, const int DESCRIPTORSIZE, const int DESCRIPTORSIZE2,\n"
"                           const int width,const int height,const int stepping,\n"
"                           constant const float2* descriptors1, constant const float2* descriptors2,\n"
"                           global float *debug) {\n"
"       int dIndex = keyPointIndex * DESCRIPTORSIZE2;\n"
"       const float2 kp = kpxy * mipScale;\n"
"       const float descriptorSize = mipScale * descriptorScale;\n"
"       const float sina = sin(angle);\n"
"       const float cosa = cos(angle);\n"
"       const float2 cosid = (float2)(cosa, sina) * descriptorSize;\n"
"       const float2 nsicd = (float2)(-sina, cosa) * descriptorSize;\n"
"       const float2 gdx = (float2)(cosa,sina);\n"
"       const float2 gdy = (float2)(-sina,cosa);\n"
"       float2 xya = (float2)(0,0);\n"
"#pragma unroll 4\n"
"       for (int b = 0; b < DESCRIPTORSIZE; ++b) {\n"
"           const int db = dIndex + b;\n"
"           const float2 dp1 = descriptors1[b];"
"           const float2 dp2 = descriptors2[b];"
"           const float2 d1 = kp + (float2)(dot(cosid,dp1), dot(nsicd,dp1));\n"
"           const float2 d2 = kp + (float2)(dot(cosid,dp2), dot(nsicd,dp2));\n"
"           const float l1 = tex(s, d1);\n"
"           const float l2 = tex(s, d2);\n"
"           const unsigned char b2 = (l2 < l1 ? 1 : 0);\n"
"           if (b2 != ( (dBits[db>>5]>>(db & 31)) & 1 )) {\n"
"               float2  gr  = (float2)( tex(s, d2 - gdx) - tex(s, d2 + gdx) , tex(s, d2 - gdy) - tex(s, d2 + gdy) )\n"
"                            -(float2)( tex(s, d1 - gdx) - tex(s, d1 + gdx) , tex(s, d1 - gdy) - tex(s, d1 + gdy) ); \n"
"               if (b2 != 0) gr = -gr;\n"
"               if (stepping != 0) gr = (float2)((gr.x == 0.f) ? 0.f : (gr.x > 0.f) ? 1.f : -1.f, (gr.y == 0.f) ? 0.f : (gr.y > 0.f) ? 1.f : -1.f);\n"
"               xya += gr * fast_length(d2 - d1);\n"
"           }\n"
"       }\n"
"       xya = normalize(xya) * (step / mipScale);\n"
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
"                        const image2d_t sb0,\n"
"                        const image2d_t sb1,\n"
"                        const image2d_t sb2,\n"
"                        const image2d_t sb3,\n"
"                        const image2d_t sb4,\n"
"                        const image2d_t sb5,\n"
"                        const image2d_t sb6,\n"
"                        const image2d_t sb7,\n"
"                        const image2d_t sb8,\n"
"                        const image2d_t sb9,\n"
"                        const image2d_t sb10,\n"
"                        const image2d_t sb11,\n"
"                        const image2d_t sb12,\n"
"                        const image2d_t sb13,\n"
"                        const image2d_t sb14,\n"
"                        const image2d_t sb15,\n"
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
"                        global const unsigned int* dbBits0,\n"
"                        global const unsigned int* dbBits1,\n"
"                        global const unsigned int* dbBits2,\n"
"                        global const unsigned int* dbBits3,\n"
"                        global const unsigned int* dbBits4,\n"
"                        global const unsigned int* dbBits5,\n"
"                        global const unsigned int* dbBits6,\n"
"                        global const unsigned int* dbBits7,\n"
"                        global const unsigned int* dbBits8,\n"
"                        global const unsigned int* dbBits9,\n"
"                        global const unsigned int* dbBits10,\n"
"                        global const unsigned int* dbBits11,\n"
"                        global const unsigned int* dbBits12,\n"
"                        global const unsigned int* dbBits13,\n"
"                        global const unsigned int* dbBits14,\n"
"                        global const unsigned int* dbBits15,\n"
"                        global const int *widths,\n"
"                        global const int *heights,\n"
"                        constant const float2* descriptors1, constant const float2* descriptors2,\n"
"                        global float *dKeyPointsX, global float *dKeyPointsY,\n"
"                        global float *dVariancePointsX, global float *dVariancePointsY,\n"
"                        global float *debug) {\n"
"       global const unsigned int *dBits[32];\n"
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
"       dBits[16] = dbBits0;\n"
"       dBits[17] = dbBits1;\n"
"       dBits[18] = dbBits2;\n"
"       dBits[19] = dbBits3;\n"
"       dBits[20] = dbBits4;\n"
"       dBits[21] = dbBits5;\n"
"       dBits[22] = dbBits6;\n"
"       dBits[23] = dbBits7;\n"
"       dBits[24] = dbBits8;\n"
"       dBits[25] = dbBits9;\n"
"       dBits[26] = dbBits10;\n"
"       dBits[27] = dbBits11;\n"
"       dBits[28] = dbBits12;\n"
"       dBits[29] = dbBits13;\n"
"       dBits[30] = dbBits14;\n"
"       dBits[31] = dbBits15;\n"
"       const int KEYPOINTCOUNT = ints[0];\n"
"       const int DESCRIPTORSIZE = ints[1];\n"
"       const int mipEnd = ints[2];\n"
"       const int STEPCOUNT = ints[3];\n"
"       const int stepping = ints[4];\n"
"       const int DESCRIPTORSIZE2 = ints[5];\n"
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
"                   case 0:  kp = refineKeyPoint( s0,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 1:  kp = refineKeyPoint( s1,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 2:  kp = refineKeyPoint( s2,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 3:  kp = refineKeyPoint( s3,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 4:  kp = refineKeyPoint( s4,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 5:  kp = refineKeyPoint( s5,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 6:  kp = refineKeyPoint( s6,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 7:  kp = refineKeyPoint( s7,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 8:  kp = refineKeyPoint( s8,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 9:  kp = refineKeyPoint( s9,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 10: kp = refineKeyPoint(s10,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 11: kp = refineKeyPoint(s11,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 12: kp = refineKeyPoint(s12,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 13: kp = refineKeyPoint(s13,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 14: kp = refineKeyPoint(s14,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 15: kp = refineKeyPoint(s15,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 16: kp = refineKeyPoint( sb0,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 17: kp = refineKeyPoint( sb1,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 18: kp = refineKeyPoint( sb2,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 19: kp = refineKeyPoint( sb3,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 20: kp = refineKeyPoint( sb4,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 21: kp = refineKeyPoint( sb5,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 22: kp = refineKeyPoint( sb6,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 23: kp = refineKeyPoint( sb7,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 24: kp = refineKeyPoint( sb8,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 25: kp = refineKeyPoint( sb9,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 26: kp = refineKeyPoint(sb10,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 27: kp = refineKeyPoint(sb11,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 28: kp = refineKeyPoint(sb12,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 29: kp = refineKeyPoint(sb13,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 30: kp = refineKeyPoint(sb14,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"                   case 31: kp = refineKeyPoint(sb15,dBits[i],j,kp,mipScale, descriptorScale, angle, step, DESCRIPTORSIZE, DESCRIPTORSIZE2, width, height, stepping,descriptors1, descriptors2, debug); break;\n"
"               };\n"
"            }\n"
"       }\n"
"       switch (v) {\n"
"           case 0: dKeyPointsX[j] = kp.x; dKeyPointsY[j] = kp.y; break;\n"
"           case 1: dVariancePointsX[j] = kp.x; dVariancePointsY[j] = kp.y; break;\n"
"       }\n"
"   }\n"
"\n";

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

    cl::Program::Sources sources;
    sources.push_back({ openCV_program.c_str(), openCV_program.length() });
    openCLProgram = cl::Program(openCLContext, sources);
    if (openCLProgram.build({ openCLDevice }, "-cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math") != CL_SUCCESS) {
        printf("Error building: %s\n", openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLDevice).c_str());
        exit(1);
    }
    for (int i = 0; i < BUFFERCOUNT; i++) {
        openCLQueue[i] = cl::CommandQueue(openCLContext, openCLDevice);
        openCLInts1[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLInt1[i]));
        openCLInts2[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLInt2[i]));
        openCLFloats1[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLFloat1[i]));
        openCLFloats2[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLFloat2[i]));
        openCLDebug1[i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(clDebug1[i]));
        openCLDebug2[i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(clDebug2[i]));
        openCLDescriptors1[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 2 * DESCRIPTORSIZE);
        openCLDescriptors2[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 2 * DESCRIPTORSIZE);
        sampleDescriptor_cl[i] = cl::Kernel(openCLProgram, "sampleDescriptor_kernel");
        refineKeyPoints_cl[i] = cl::Kernel(openCLProgram, "refineKeyPoints_kernel");
    }
}

void uploadDescriptorShape_openCL(const int bufferId) {
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
    // can be blocking
    openCLQueue[bufferId].enqueueWriteBuffer(openCLDescriptors1[bufferId], CL_TRUE, 0, sizeof(float) * 2 * DESCRIPTORSIZE, &(descriptors1[0]));
    openCLQueue[bufferId].enqueueWriteBuffer(openCLDescriptors2[bufferId], CL_TRUE, 0, sizeof(float) * 2 * DESCRIPTORSIZE, &(descriptors2[0]));
}

void uploadMipMaps_openCL(const int bufferId, const std::vector<cv::Mat> &mipMaps) {
    uploadMipMaps_openCL_waitfor(bufferId);
    if (mipMaps.empty()) return;
    if (mipMaps.size() >= MAXMIPMAPS) {
        printf("Error: too many mipmaps %d of %d\n", int(mipMaps.size()), MAXMIPMAPS);
    }
    if (mipMaps.size() != mipMapWidths[bufferId].size() || mipMaps[0].cols != mipMapWidths[bufferId][0] || mipMaps[0].rows != mipMapHeights[bufferId][0]) { // can be blocking
        for (int i = 0; i < mipMaps.size(); i++) {
            int success = 0;
            openCLMipMaps[bufferId][i] = cl::Image2D(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cl::ImageFormat(CL_R, CL_UNORM_INT8), mipMaps[i].cols, mipMaps[i].rows, 0, nullptr, &success);
            if (success != CL_SUCCESS) 
                exit(1);
        }
        openCLMipMapWidths[bufferId] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * mipMaps.size());
        openCLMipMapHeights[bufferId] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * mipMaps.size());
        mipMapWidths[bufferId].resize(mipMaps.size());
        mipMapHeights[bufferId].resize(mipMaps.size());
        for (int i = 0; i < mipMapWidths[bufferId].size(); i++) mipMapWidths[bufferId][i] = mipMaps[i].cols;
        for (int i = 0; i < mipMapHeights[bufferId].size(); i++) mipMapHeights[bufferId][i] = mipMaps[i].rows;
        openCLQueue[bufferId].enqueueWriteBuffer(openCLMipMapWidths[bufferId], CLBLOCKING, 0, sizeof(unsigned int) * mipMapWidths[bufferId].size(), &(mipMapWidths[bufferId][0]));
        openCLQueue[bufferId].enqueueWriteBuffer(openCLMipMapHeights[bufferId], CLBLOCKING, 0, sizeof(unsigned int) * mipMapHeights[bufferId].size(), &(mipMapHeights[bufferId][0]));
    }
    for (int i = 0; i < mipMaps.size(); i++) {
        cl::array<cl::size_type, 2> origin = { 0,0 };
        cl::array<cl::size_type, 2> region = { cl::size_type(mipMaps[i].cols), cl::size_type(mipMaps[i].rows) };
        if (openCLQueue[bufferId].enqueueWriteImage(openCLMipMaps[bufferId][i], CLNONBLOCKING, origin, region, 0, 0, (uchar* const)mipMaps[i].data, nullptr, &(mipMapUploadEvent[bufferId][i])) != CL_SUCCESS)
            exit(1);
    }
}

void uploadMipMaps_openCL_waitfor(const int bufferId) {
    for (int i = 0; i < mipMapWidths[bufferId].size(); i++)
        mipMapUploadEvent[bufferId][i].wait();
}

void uploadKeyPoints_openCL(const int bufferId, const std::vector<KeyPoint>& keyPoints) {
    uploadKeyPoints_openCL_waitfor(bufferId);
    if (keyPoints.size() != openCLKpx[bufferId].size()) {
        openCLKpx[bufferId].resize(keyPoints.size());
        openCLKpy[bufferId].resize(keyPoints.size());
        openCLKeyPointsX[bufferId] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLKeyPointsY[bufferId] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsX[bufferId] = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewKeyPointsY[bufferId] = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsX[bufferId] = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
        openCLNewVariancePointsY[bufferId] = cl::Buffer(openCLContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * keyPoints.size());
    }
    for (int i = int(keyPoints.size()) - 1; i >= 0; i--) {
        openCLKpx[bufferId][i] = keyPoints[i].x;
        openCLKpy[bufferId][i] = keyPoints[i].y;
    }
    openCLQueue[bufferId].enqueueWriteBuffer(openCLKeyPointsX[bufferId], CLNONBLOCKING, 0, sizeof(float)*openCLKpx[bufferId].size(), &(openCLKpx[bufferId][0]), nullptr, &(keyPointsXEvent[bufferId]));
    openCLQueue[bufferId].enqueueWriteBuffer(openCLKeyPointsY[bufferId], CLNONBLOCKING, 0, sizeof(float)*openCLKpy[bufferId].size(), &(openCLKpy[bufferId][0]), nullptr, &(keyPointsYEvent[bufferId]));
}

void uploadKeyPoints_openCL_waitfor(const int bufferId) {
    keyPointsXEvent[bufferId].wait();
    keyPointsYEvent[bufferId].wait();
}

void uploadDescriptors_openCL(const int bufferId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    uploadDescriptors_openCL_waitfor(bufferId, mipMap);
    const std::vector<Descriptor>& descriptors = sourceMips[mipMap];
    if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId][mipMap].size()) {
        if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId][mipMap].size()) {
            openCLDescriptorBits[bufferId][mipMap].resize(descriptors.size() * Descriptor::uint32count);
        }
        openCLBits[bufferId][mipMap] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBits[bufferId][mipMap].size());
    }
    for (int i = int(descriptors.size()) - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            openCLDescriptorBits[bufferId][mipMap][j + i * Descriptor::uint32count] = descriptors[i].bits[j];
        }
    }
    openCLQueue[bufferId].enqueueWriteBuffer(openCLBits[bufferId][mipMap], CLNONBLOCKING, 0, sizeof(unsigned int) * openCLDescriptorBits[bufferId][mipMap].size(), &(openCLDescriptorBits[bufferId][mipMap][0]), nullptr, &(descriptorEvent[bufferId][mipMap]));
}

void uploadDescriptors_openCL_waitfor(const int bufferId, const int mipMap) {
    descriptorEvent[bufferId][mipMap].wait();
}

void sampleDescriptors_openCL(const int bufferId, const int mipMap, const int keyPointCount, const float descriptorScale, const int width, const int height, const float mipScale) {
    if ((keyPointCount * Descriptor::uint32count) != openCLDescriptorBitsFull[bufferId][mipMap].size()) {
        openCLDescriptorBitsFull[bufferId][mipMap].resize(keyPointCount * Descriptor::uint32count);
        openCLBitsFull[bufferId][mipMap] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBitsFull[bufferId][mipMap].size());
    }
    openCLInt2[bufferId][0] = keyPointCount;
    openCLInt2[bufferId][1] = DESCRIPTORSIZE;
    openCLInt2[bufferId][2] = width;
    openCLInt2[bufferId][3] = height;
    openCLInt2[bufferId][4] = Descriptor::uint32count*32;
    openCLFloat2[bufferId][0] = descriptorScale;
    openCLFloat2[bufferId][1] = mipScale;
    openCLQueue[bufferId].enqueueWriteBuffer(openCLInts2[bufferId], CLNONBLOCKING, 0, sizeof(openCLInt2[bufferId]), openCLInt2[bufferId]);
    openCLQueue[bufferId].enqueueWriteBuffer(openCLFloats2[bufferId], CLNONBLOCKING, 0, sizeof(openCLFloat2[bufferId]), openCLFloat2[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(0, openCLInts2[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(1, openCLFloats2[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(2, openCLKeyPointsX[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(3, openCLKeyPointsY[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(4, openCLMipMaps[bufferId][mipMap]);
    sampleDescriptor_cl[bufferId].setArg(5, openCLDescriptors1[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(6, openCLDescriptors2[bufferId]);
    sampleDescriptor_cl[bufferId].setArg(7, openCLBitsFull[bufferId][mipMap]);
    sampleDescriptor_cl[bufferId].setArg(8, openCLDebug2[bufferId]);
    openCLQueue[bufferId].enqueueNDRangeKernel(sampleDescriptor_cl[bufferId], cl::NullRange, cl::NDRange(openCLKpx[bufferId].size()), cl::NullRange);
    openCLQueue[bufferId].flush();
}

void sampleDescriptors_openCL_waitfor(const int bufferId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips) {
    const int keyPointCount = openCLDescriptorBitsFull[bufferId][mipMap].size() / Descriptor::uint32count;
    std::vector<Descriptor>& dest = destMips[mipMap];
    dest.resize(keyPointCount);
    openCLQueue[bufferId].enqueueReadBuffer(openCLBitsFull[bufferId][mipMap], CL_TRUE, 0, openCLDescriptorBitsFull[bufferId][mipMap].size() * sizeof(unsigned int), &(openCLDescriptorBitsFull[bufferId][mipMap][0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLDebug2[bufferId], CL_TRUE, 0, sizeof(clDebug2[bufferId]), clDebug2[bufferId]);
    openCLQueue[bufferId].finish();
    for (int i = 0; i < keyPointCount; ++i) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            dest[i].bits[j] = openCLDescriptorBitsFull[bufferId][mipMap][i * Descriptor::uint32count + j];
        }
    }
}

void refineKeyPoints_openCL(const int bufferId, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE) {
    openCLInt1[bufferId][0] = int(openCLKpx[bufferId].size());
    openCLInt1[bufferId][1] = DESCRIPTORSIZE;
    openCLInt1[bufferId][2] = mipEnd;
    openCLInt1[bufferId][3] = STEPCOUNT;
    openCLInt1[bufferId][4] = stepping ? 1 : 0;
    openCLInt1[bufferId][5] = Descriptor::uint32count * 32;
    openCLFloat1[bufferId][0] = MIPSCALE;
    openCLFloat1[bufferId][1] = STEPSIZE;
    openCLFloat1[bufferId][2] = SCALEINVARIANCE;
    openCLFloat1[bufferId][3] = ROTATIONINVARIANCE;
    openCLQueue[bufferId].enqueueWriteBuffer(openCLInts1[bufferId], CLNONBLOCKING, 0, sizeof(openCLInt1[bufferId]), openCLInt1[bufferId]);
    openCLQueue[bufferId].enqueueWriteBuffer(openCLFloats1[bufferId], CLNONBLOCKING, 0, sizeof(openCLFloat1[bufferId]), openCLFloat1[bufferId]);
    int a = 0;
    refineKeyPoints_cl[bufferId].setArg(a, openCLInts1[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLFloats1[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLKeyPointsX[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLKeyPointsY[bufferId]); a++;
    for (int i = 0; i < MAXMIPMAPS; i++) {
        refineKeyPoints_cl[bufferId].setArg(a, i < mipMapWidths[bufferId].size() ? openCLMipMaps[bufferId][i] : openCLMipMaps[bufferId][0]); a++;
    }
    for (int i = 0; i < MAXMIPMAPS; i++) {
        refineKeyPoints_cl[bufferId].setArg(a, i < mipMapWidths[bufferId].size() ? openCLBits[bufferId][i] : openCLBits[bufferId][0]); a++;
    }
    refineKeyPoints_cl[bufferId].setArg(a, openCLMipMapWidths[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLMipMapHeights[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLDescriptors1[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLDescriptors2[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLNewKeyPointsX[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLNewKeyPointsY[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLNewVariancePointsX[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLNewVariancePointsY[bufferId]); a++;
    refineKeyPoints_cl[bufferId].setArg(a, openCLDebug1[bufferId]); a++;
    openCLQueue[bufferId].enqueueNDRangeKernel(refineKeyPoints_cl[bufferId], cl::NullRange, cl::NDRange(openCLKpx[bufferId].size() * 2), cl::NullRange);
    openCLQueue[bufferId].flush();
}

void refineKeyPoints_openCL_waitfor(const int bufferId, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors) {
    destErrors.resize(destKeyPoints.size());
    std::vector<float> destKeyPointsX; destKeyPointsX.resize(destKeyPoints.size());
    std::vector<float> destKeyPointsY; destKeyPointsY.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsX; destVariancePointsX.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsY; destVariancePointsY.resize(destKeyPoints.size());
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewKeyPointsX[bufferId], CL_TRUE, 0, destKeyPointsX.size() * sizeof(float), &(destKeyPointsX[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewKeyPointsY[bufferId], CL_TRUE, 0, destKeyPointsY.size() * sizeof(float), &(destKeyPointsY[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewVariancePointsX[bufferId], CL_TRUE, 0, destVariancePointsX.size() * sizeof(float), &(destVariancePointsX[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewVariancePointsY[bufferId], CL_TRUE, 0, destVariancePointsY.size() * sizeof(float), &(destVariancePointsY[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLDebug1[bufferId], CL_TRUE, 0, sizeof(clDebug1[bufferId]), clDebug1[bufferId]);
    openCLQueue[bufferId].finish();
    for (int i = 0; i < destKeyPoints.size(); i++) {
        destKeyPoints[i].x = (destKeyPointsX[i] + destVariancePointsX[i]) * 0.5f;
        destKeyPoints[i].y = (destKeyPointsY[i] + destVariancePointsY[i]) * 0.5f;
        const float errX = (destKeyPointsX[i] - destVariancePointsX[i]);
        const float errY = (destKeyPointsY[i] - destVariancePointsY[i]);
        destErrors[i] = sqrtf(errX * errX + errY * errY);
    }
}

void uploadDescriptorShape_openCL() {
    uploadDescriptorShape_openCL(0);
}
void uploadMipMaps_openCL(const std::vector<cv::Mat>& mipMaps) {
    uploadMipMaps_openCL(0, mipMaps);
    uploadMipMaps_openCL_waitfor(0);
}
void uploadKeyPoints_openCL(const std::vector<KeyPoint>& keyPoints) {
    uploadKeyPoints_openCL(0, keyPoints);
    uploadKeyPoints_openCL_waitfor(0);
}
void uploadDescriptors_openCL(const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    uploadDescriptors_openCL(0, mipMap, sourceMips);
    uploadDescriptors_openCL_waitfor(0, mipMap);
}
void sampleDescriptors_openCL(const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const float descriptorScale, const int width, const int height, const float mipScale) {
    sampleDescriptors_openCL(0, mipMap, destMips[mipMap].size(), descriptorScale, width, height, mipScale);
    sampleDescriptors_openCL_waitfor(0, mipMap, destMips);
}
void refineKeyPoints_openCL(std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE) {
    refineKeyPoints_openCL(0, mipEnd, STEPCOUNT, stepping, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
    refineKeyPoints_openCL_waitfor(0, destKeyPoints, destErrors);
}
