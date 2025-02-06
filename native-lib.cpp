#include <jni.h>
#include <string>
#include "opengl_refinement.hpp"

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