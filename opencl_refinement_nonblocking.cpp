// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "refinement.hpp"
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
static cl::Buffer openCLBitsFull[BUFFERCOUNT];
static cl::Buffer openCLInts[BUFFERCOUNT];
static cl::Buffer openCLFloats[BUFFERCOUNT];
static cl::Buffer openCLDebug[BUFFERCOUNT];
static cl::Kernel sampleDescriptor_cl;
static cl::Kernel refineKeyPoints_cl;
static int openCLInt[BUFFERCOUNT][10];
static float openCLFloat[BUFFERCOUNT][10];
static float clDebug[BUFFERCOUNT][10];
static std::vector<float> openCLKpx[BUFFERCOUNT];
static std::vector<float> openCLKpy[BUFFERCOUNT];
static std::vector<unsigned int> openCLDescriptorBits[BUFFERCOUNT];
static std::vector<unsigned int> openCLDescriptorBitsFull[BUFFERCOUNT];
static std::vector<unsigned int> mipMapWidths[BUFFERCOUNT];
static std::vector<unsigned int> mipMapHeights[BUFFERCOUNT];
static cl::Event mipMapUploadEvent[BUFFERCOUNT][MAXMIPMAPS];

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
        openCLInts[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLInt[i]));
        openCLFloats[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(openCLFloat[i]));
        openCLDebug[i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(clDebug[i]));
        openCLDescriptors1[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 2 * DESCRIPTORSIZE);
        openCLDescriptors2[i] = cl::Buffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 2 * DESCRIPTORSIZE);
    }
    sampleDescriptor_cl = cl::Kernel(openCLProgram, "sampleDescriptor_kernel");
    refineKeyPoints_cl = cl::Kernel(openCLProgram, "refineKeyPoints_kernel");
}

void uploadMipMaps_openCL(const int bufferId, const std::vector<cv::Mat> &mipMaps) {
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
        if (openCLQueue[bufferId].enqueueWriteImage(openCLMipMaps[bufferId][i], CL_FALSE, origin, region, 0, 0, (uchar* const)mipMaps[i].data, nullptr, &(mipMapUploadEvent[bufferId][i])) != CL_SUCCESS)
            exit(1);
    }
}

void uploadMipMaps_openCL_waitfor(const int bufferId) {
    for (int i = 0; i < mipMapWidths[bufferId].size(); i++)
        mipMapUploadEvent[bufferId][i].wait();
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
    openCLQueue[bufferId].enqueueWriteBuffer(openCLDescriptors1[bufferId], CLBLOCKING, 0, sizeof(float) * 2 * DESCRIPTORSIZE, &(descriptors1[0]));
    openCLQueue[bufferId].enqueueWriteBuffer(openCLDescriptors2[bufferId], CLBLOCKING, 0, sizeof(float) * 2 * DESCRIPTORSIZE, &(descriptors2[0]));
}

void uploadKeyPoints_openCL(const int bufferId, const std::vector<KeyPoint>& keyPoints) {
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
    openCLQueue[bufferId].enqueueWriteBuffer(openCLKeyPointsX[bufferId], CLBLOCKING, 0, sizeof(float)*openCLKpx[bufferId].size(), &(openCLKpx[bufferId][0]));
    openCLQueue[bufferId].enqueueWriteBuffer(openCLKeyPointsY[bufferId], CLBLOCKING, 0, sizeof(float)*openCLKpy[bufferId].size(), &(openCLKpy[bufferId][0]));
}

void uploadDescriptors_openCL(const int bufferId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    const std::vector<Descriptor>& descriptors = sourceMips[mipMap];

    if (mipMap >= openCLBits[bufferId].size() || (descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId].size()) {
        if ((descriptors.size() * Descriptor::uint32count) != openCLDescriptorBits[bufferId].size()) {
            openCLDescriptorBits[bufferId].resize(descriptors.size() * Descriptor::uint32count);
        }
        openCLBits[bufferId].resize(mipMap + 1);
        for (int i = 0; i < mipMap + 1; i++) {
            openCLBits[bufferId][i] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBits[bufferId].size());
        }
    }

    for (int i = int(descriptors.size()) - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            openCLDescriptorBits[bufferId][j + i * Descriptor::uint32count] = descriptors[i].bits[j];
        }
    }
    openCLQueue[bufferId].enqueueWriteBuffer(openCLBits[bufferId][mipMap], CLBLOCKING, 0, sizeof(unsigned int) * openCLDescriptorBits[bufferId].size(), &(openCLDescriptorBits[bufferId][0]));
}

void sampleDescriptors_openCL(const int bufferId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale) {
    std::vector<Descriptor>& dest = destMips[mipMap];

    if ((dest.size() * Descriptor::uint32count) != openCLDescriptorBitsFull[bufferId].size()) {
        openCLDescriptorBitsFull[bufferId].resize(dest.size() * Descriptor::uint32count);
        openCLBitsFull[bufferId] = cl::Buffer(openCLContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned int) * openCLDescriptorBitsFull[bufferId].size());
    }

    openCLInt[bufferId][0] = int(dest.size());
    openCLInt[bufferId][1] = DESCRIPTORSIZE;
    openCLInt[bufferId][2] = width;
    openCLInt[bufferId][3] = height;
    openCLInt[bufferId][4] = Descriptor::uint32count*32;
    openCLFloat[bufferId][0] = descriptorScale;
    openCLFloat[bufferId][1] = mipScale;
    openCLQueue[bufferId].enqueueWriteBuffer(openCLInts[bufferId], CLBLOCKING, 0, sizeof(openCLInt[bufferId]), openCLInt[bufferId]);
    openCLQueue[bufferId].enqueueWriteBuffer(openCLFloats[bufferId], CLBLOCKING, 0, sizeof(openCLFloat[bufferId]), openCLFloat[bufferId]);
    sampleDescriptor_cl.setArg(0, openCLInts[bufferId]);
    sampleDescriptor_cl.setArg(1, openCLFloats[bufferId]);
    sampleDescriptor_cl.setArg(2, openCLKeyPointsX[bufferId]);
    sampleDescriptor_cl.setArg(3, openCLKeyPointsY[bufferId]);
    sampleDescriptor_cl.setArg(4, openCLMipMaps[bufferId][mipMap]);
    sampleDescriptor_cl.setArg(5, openCLDescriptors1[bufferId]);
    sampleDescriptor_cl.setArg(6, openCLDescriptors2[bufferId]);
    sampleDescriptor_cl.setArg(7, openCLBitsFull[bufferId]);
    sampleDescriptor_cl.setArg(8, openCLDebug[bufferId]);
    openCLQueue[bufferId].enqueueNDRangeKernel(sampleDescriptor_cl, cl::NullRange, cl::NDRange(openCLKpx[bufferId].size()), cl::NullRange);
    openCLQueue[bufferId].flush();
    openCLQueue[bufferId].enqueueReadBuffer(openCLBitsFull[bufferId], CLBLOCKING, 0, openCLDescriptorBitsFull[bufferId].size() * sizeof(unsigned int), &(openCLDescriptorBitsFull[bufferId][0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLDebug[bufferId], CLBLOCKING, 0, sizeof(clDebug[bufferId]), clDebug[bufferId]);
    for (int i = 0; i < dest.size(); ++i) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
             dest[i].bits[j] = openCLDescriptorBitsFull[bufferId][i * Descriptor::uint32count + j];
        }
    }
}

void refineKeyPoints_openCL(const int bufferId, std::vector<KeyPoint> &destKeyPoints, std::vector<float>& destErrors, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE) {
    destErrors.resize(destKeyPoints.size());
    std::vector<KeyPoint> destVariancePoints;
    destVariancePoints.resize(destKeyPoints.size());
    openCLInt[bufferId][0] = int(openCLKpx[bufferId].size());
    openCLInt[bufferId][1] = DESCRIPTORSIZE;
    openCLInt[bufferId][2] = mipEnd;
    openCLInt[bufferId][3] = STEPCOUNT;
    openCLInt[bufferId][4] = stepping ? 1 : 0;
    openCLInt[bufferId][5] = Descriptor::uint32count * 32;
    openCLFloat[bufferId][0] = MIPSCALE;
    openCLFloat[bufferId][1] = STEPSIZE;
    openCLFloat[bufferId][2] = SCALEINVARIANCE;
    openCLFloat[bufferId][3] = ROTATIONINVARIANCE;
    openCLQueue[bufferId].enqueueWriteBuffer(openCLInts[bufferId], CLBLOCKING, 0, sizeof(openCLInt[bufferId]), openCLInt[bufferId]);
    openCLQueue[bufferId].enqueueWriteBuffer(openCLFloats[bufferId], CLBLOCKING, 0, sizeof(openCLFloat[bufferId]), openCLFloat[bufferId]);
    int a = 0;
    refineKeyPoints_cl.setArg(a, openCLInts[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLFloats[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLKeyPointsX[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLKeyPointsY[bufferId]); a++;
    for (int i = 0; i < 32; i++) {
        refineKeyPoints_cl.setArg(a, i < mipMapWidths[bufferId].size() ? openCLMipMaps[bufferId][i] : openCLMipMaps[bufferId][0]); a++;
    }
    for (int i = 0; i < 32; i++) {
        refineKeyPoints_cl.setArg(a, i < openCLBits[bufferId].size() ? openCLBits[bufferId][i] : openCLBits[bufferId][0]); a++;
    }
    refineKeyPoints_cl.setArg(a, openCLMipMapWidths[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLMipMapHeights[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptors1[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLDescriptors2[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsX[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLNewKeyPointsY[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsX[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLNewVariancePointsY[bufferId]); a++;
    refineKeyPoints_cl.setArg(a, openCLDebug[bufferId]); a++;
    openCLQueue[bufferId].enqueueNDRangeKernel(refineKeyPoints_cl, cl::NullRange, cl::NDRange(openCLKpx[bufferId].size() * 2), cl::NullRange);
    openCLQueue[bufferId].flush();
    std::vector<float> destKeyPointsX; destKeyPointsX.resize(destKeyPoints.size());
    std::vector<float> destKeyPointsY; destKeyPointsY.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsX; destVariancePointsX.resize(destKeyPoints.size());
    std::vector<float> destVariancePointsY; destVariancePointsY.resize(destKeyPoints.size());
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewKeyPointsX[bufferId], CLBLOCKING, 0, destKeyPointsX.size() * sizeof(float), &(destKeyPointsX[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewKeyPointsY[bufferId], CLBLOCKING, 0, destKeyPointsY.size() * sizeof(float), &(destKeyPointsY[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewVariancePointsX[bufferId], CLBLOCKING, 0, destVariancePointsX.size() * sizeof(float), &(destVariancePointsX[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLNewVariancePointsY[bufferId], CLBLOCKING, 0, destVariancePointsY.size() * sizeof(float), &(destVariancePointsY[0]));
    openCLQueue[bufferId].enqueueReadBuffer(openCLDebug[bufferId], CLBLOCKING, 0, sizeof(clDebug[bufferId]), clDebug[bufferId]);
    for (int i = 0; i < destKeyPoints.size(); i++) {
        destKeyPoints[i].x = (destKeyPointsX[i] + destVariancePointsX[i]) * 0.5f;
        destKeyPoints[i].y = (destKeyPointsY[i] + destVariancePointsY[i]) * 0.5f;
        const float errX = (destKeyPointsX[i] - destVariancePointsX[i]);
        const float errY = (destKeyPointsY[i] - destVariancePointsY[i]);
        destErrors[i] = sqrtf(errX*errX+errY*errY);
    }
}
