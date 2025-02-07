#include <jni.h>
#include <string>
#include "opengl_refinement.hpp"
#include <vector>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lucyteandroid_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_initOpenGL(JNIEnv *env, jobject thiz) {
    initOpenGL();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_uploadMipMap(JNIEnv *env, jobject thiz,
                                                         jbyteArray base_mip_map, jint width,
                                                         jint height) {
    jboolean isCopy;
    jbyte* data = env->GetByteArrayElements(base_mip_map, &isCopy);
    MipMap k;
    k.width = width;
    k.height = height;
    k.data = (unsigned char*)data;
    uploadMipMaps_openGL(0,k);
    env->ReleaseByteArrayElements(base_mip_map, data, JNI_ABORT);
}

void displayMipMap(int mipmapsId, int mipLevel);

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_displayMipMap(JNIEnv *env, jobject thiz,
                                                          jint mipmaps_id, jint mip_map) {
    displayMipMap(mipmaps_id,mip_map);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_uploadKeyPoints(JNIEnv *env, jobject thiz,
                                                            jint key_points_id,
                                                            jfloatArray key_points_x,
                                                            jfloatArray key_points_y,
                                                            jint key_point_count) {
    jboolean isCopy;
    jfloat* kpx = env->GetFloatArrayElements(key_points_x, &isCopy);
    jfloat* kpy = env->GetFloatArrayElements(key_points_y, &isCopy);
    std::vector<KeyPoint> keyPoints;
    keyPoints.resize(key_point_count);
    for (int i = 0; i < key_point_count; i++) {
        keyPoints[i].x = kpx[i];
        keyPoints[i].y = kpy[i];
    }
    uploadKeyPoints_openGL(key_points_id, keyPoints);
    env->ReleaseFloatArrayElements(key_points_x, kpx, JNI_ABORT);
    env->ReleaseFloatArrayElements(key_points_y, kpy, JNI_ABORT);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_uploadDescriptors(JNIEnv *env, jobject thiz,
                                                              jint descriptors_id,
                                                              jintArray descriptors,
                                                              jint uint32_count,
                                                              jint key_point_count,
                                                              jint mip_map_count) {
    jboolean isCopy;
    jint* dsc = env->GetIntArrayElements(descriptors, &isCopy);
    std::vector<std::vector<Descriptor>> mips;
    mips.resize(mip_map_count);
    unsigned int *d = (unsigned int*)dsc;
    for (int i = 0; i < mip_map_count; i++) {
        mips[i].resize(key_point_count);
        for (int j = 0; j < key_point_count; j++) {
            for (int k = 0; k < uint32_count; k++) {
                mips[i][j].bits[k] = *d++;
            }
        }
    }
    env->ReleaseIntArrayElements(descriptors, dsc, JNI_ABORT);
    uploadDescriptors_openGL(descriptors_id, mip_map_count-1, mips);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_sampleDescriptors(JNIEnv *env, jobject thiz,
                                                              jint key_points_id,
                                                              jint descriptors_id, jint mipmaps_id,
                                                              jint mip_end, jint key_point_count, jint uint32_count,
                                                              jintArray descriptors,
                                                              jfloat descriptor_scale,
                                                              jfloat mip_scale) {
    std::vector<std::vector<Descriptor>> destMips;
    sampleDescriptors_openGL(key_points_id, descriptors_id, mipmaps_id, mip_end, destMips, descriptor_scale, mip_scale);
    jboolean isCopy;
    jint* dsc = env->GetIntArrayElements(descriptors, &isCopy);
    unsigned int *d = (unsigned int*)dsc;
    for (int i = 0; i < mip_end; i++)
        for (int j = 0; j < key_point_count; j++)
            for (int k = 0; k < uint32_count; k++)
                *d++ = destMips[i][j].bits[k];
    env->ReleaseIntArrayElements(descriptors, dsc, JNI_COMMIT);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_refineKeyPoints(JNIEnv *env, jobject thiz,
                                                            jint key_points_id, jint descriptors_id,
                                                            jint mipmaps_id, jint mip_end,
                                                            jint key_point_count, jint uint32_count,
                                                            jint stepcount, jboolean stepping,
                                                            jfloat mipscale, jfloat stepsize,
                                                            jfloat scaleinvariance,
                                                            jfloat rotationinvariance,
                                                            jfloatArray key_points_x,
                                                            jfloatArray key_points_y,
                                                            jfloatArray errors) {
    std::vector<KeyPoint> destKeyPoints;
    std::vector<float> destErrors;
    refineKeyPoints_openGL(key_points_id, descriptors_id, mipmaps_id, key_point_count, mip_end, stepcount, stepping, mipscale, stepsize, scaleinvariance, rotationinvariance, destKeyPoints, destErrors);
    jboolean isCopy;
    jfloat* kpx = env->GetFloatArrayElements(key_points_x, &isCopy);
    jfloat* kpy = env->GetFloatArrayElements(key_points_y, &isCopy);
    jfloat* err = env->GetFloatArrayElements(errors, &isCopy);
    for (int i = 0; i < key_point_count; i++) {
        kpx[i] = destKeyPoints[i].x;
        kpy[i] = destKeyPoints[i].y;
        err[i] = destErrors[i];
    }
    env->ReleaseFloatArrayElements(key_points_x, kpx, JNI_COMMIT);
    env->ReleaseFloatArrayElements(key_points_y, kpy, JNI_COMMIT);
    env->ReleaseFloatArrayElements(errors, err, JNI_COMMIT);
}