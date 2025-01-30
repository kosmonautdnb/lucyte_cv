// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "opencl_refinement_nonblocking.hpp"
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
static cl::Platform openCLPlatform;
static cl::Device openCLDevice;
static cl::Context openCLContext;
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
static std::vector<cl::Buffer> openCLBits[BUFFERCOUNT];
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

extern std::string openCV_program;

void initOpenCL_nonblocking() {
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
    openCLQueue[0] = cl::CommandQueue(openCLContext, openCLDevice);
    openCLQueue[1] = cl::CommandQueue(openCLContext, openCLDevice);

    cl::Program::Sources sources;
    sources.push_back({ openCV_program.c_str(), openCV_program.length() });
    openCLProgram = cl::Program(openCLContext, sources);
    if (openCLProgram.build({ openCLDevice }, "-cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math") != CL_SUCCESS) {
        printf("Error building: %s\n", openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLDevice).c_str());
        exit(1);
    }
    for (int i = 0; i < BUFFERCOUNT; i++) {
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
    if (mipMap >= openCLBits[bufferId].size() || (descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId][mipMap].size()) {
        if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId][mipMap].size()) {
            openCLDescriptorBits[bufferId][mipMap].resize(descriptors.size() * Descriptor::uint32count);
        }
        openCLBits[bufferId].resize(mipMap + 1);
        for (int i = 0; i < mipMap + 1; i++) {
            openCLBits[bufferId][i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBits[bufferId][i].size());
        }
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
    if (openCLBits[bufferId].empty()) return;
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
    for (int i = 0; i < 32; i++) {
        refineKeyPoints_cl[bufferId].setArg(a, i < mipMapWidths[bufferId].size() ? openCLMipMaps[bufferId][i] : openCLMipMaps[bufferId][0]); a++;
    }
    for (int i = 0; i < 32; i++) {
        refineKeyPoints_cl[bufferId].setArg(a, i < openCLBits[bufferId].size() ? openCLBits[bufferId][i] : openCLBits[bufferId][0]); a++;
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
