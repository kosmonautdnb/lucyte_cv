#include <jni.h>
#include <string>
#include "opengl_refinement.hpp"
#include <vector>
#include <GLES3/gl3.h>
#include <EGL/egl.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_lucyteandroid_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

const std::string paintColor = "#version 300 es\nprecision highp float;precision highp int;\n"
     "in vec2 p;\n"
     "out vec4 frag_color;\n"
     "uniform vec4 color;\n"
     "void main()\n"
     "{\n"
     "   frag_color = color;"
     "}\n"
     "\n";


const std::string basicProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
     "layout(location = 0) in vec2 pos;\n"
     "void main()\n"
     "{\n"
     "    gl_Position = vec4(pos.xy, 0.0, 1.0);\n"
     "}\n"
     "\n";

GLuint openGLVertexShader_basic;
GLuint openGLFragmentShader_paintColor;
GLuint openGLProgram_paintColor;

GLuint pixelShader(const char* shaderProgram, unsigned int size);
GLuint vertexShader(const char* shaderProgram, unsigned int size);
GLuint program(GLuint vertexShader, GLuint pixelShader);

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_initOpenGL(JNIEnv *env, jobject thiz) {
    initOpenGL();
    openGLVertexShader_basic = vertexShader(basicProgram.c_str(), (unsigned int)basicProgram.length());
    openGLFragmentShader_paintColor = pixelShader(paintColor.c_str(), (unsigned int)paintColor.length());
    openGLProgram_paintColor = program(openGLVertexShader_basic, openGLFragmentShader_paintColor);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_uploadMipMap(JNIEnv *env, jobject thiz,
                                                         jint mipmapsId, jbyteArray base_mip_map, jint width,
                                                         jint height) {
    jboolean isCopy;
    jbyte* data = env->GetByteArrayElements(base_mip_map, &isCopy);
    MipMap k;
    k.width = width;
    k.height = height;
    k.data = (unsigned char*)data;
    uploadMipMaps_openGL(mipmapsId,k);
    env->ReleaseByteArrayElements(base_mip_map, data, JNI_ABORT);
}

void displayMipMap(int mipmapsId, int mipLevel, int screenWidth, int screenHeight, float scaleX, float scaleY);

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_displayMipMap(JNIEnv *env, jobject thiz,
                                                          jint mipmaps_id, jint mip_map, jint screen_width, jint screen_height, jfloat scaleX, jfloat scaleY) {
    displayMipMap(mipmaps_id,mip_map,screen_width,screen_height, scaleX, scaleY);
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
                                                              jint mip_levels, jint key_point_count, jint uint32_count,
                                                              jintArray descriptors,
                                                              jfloat descriptor_scale,
                                                              jfloat mip_scale) {
    std::vector<std::vector<Descriptor>> destMips;
    sampleDescriptors_openGL(key_points_id, descriptors_id, mipmaps_id, mip_levels, destMips, descriptor_scale, mip_scale);
    jboolean isCopy;
    jint* dsc = env->GetIntArrayElements(descriptors, &isCopy);
    unsigned int *d = (unsigned int*)dsc;
    for (int i = 0; i < mip_levels; i++)
        for (int j = 0; j < key_point_count; j++)
            for (int k = 0; k < uint32_count; k++)
                *d++ = destMips[i][j].bits[k];
    env->ReleaseIntArrayElements(descriptors, dsc, 0);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_refineKeyPoints(JNIEnv *env, jobject thiz,
                                                            jint key_points_id, jint descriptors_id,
                                                            jint mipmaps_id, jint mip_levels,
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
    refineKeyPoints_openGL(key_points_id, descriptors_id, mipmaps_id, key_point_count, mip_levels, stepcount, stepping, mipscale, stepsize, scaleinvariance, rotationinvariance, destKeyPoints, destErrors);
    jboolean isCopy;
    jfloat* kpx = env->GetFloatArrayElements(key_points_x, &isCopy);
    jfloat* kpy = env->GetFloatArrayElements(key_points_y, &isCopy);
    jfloat* err = env->GetFloatArrayElements(errors, &isCopy);
    for (int i = 0; i < key_point_count; i++) {
        kpx[i] = destKeyPoints[i].x;
        kpy[i] = destKeyPoints[i].y;
        err[i] = destErrors[i];
    }
    env->ReleaseFloatArrayElements(key_points_x, kpx, 0);
    env->ReleaseFloatArrayElements(key_points_y, kpy, 0);
    env->ReleaseFloatArrayElements(errors, err, 0);
}

GLuint vbo = 0;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_drawRectangle(JNIEnv *env, jobject thiz, jfloat x0,
                                                          jfloat y0, jfloat x1, jfloat y1,
                                                          jint color, jint screen_width,
                                                          jint screen_height) {
    x0 /= (float)screen_width;
    y0 /= (float)screen_height;
    x1 /= (float)screen_width;
    y1 /= (float)screen_height;
    x0 = x0 * 2.f - 1.f;
    y0 = (y0 * 2.f - 1.f) * -1.f;
    x1 = x1 * 2.f - 1.f;
    y1 = (y1 * 2.f - 1.f) * -1.f;
    glViewport(0,0,screen_width, screen_height);

    glUseProgram(openGLProgram_paintColor);
    float points[] = {
            x1,  y1,  0.0f,
            x1, y0,  0.0f,
            x0, y0,  0.0f,
            x1,  y1,  0.0f,
            x0,  y1,  0.0f,
            x0, y0,  0.0f
    };
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * 6 * sizeof(float), points, GL_STATIC_DRAW);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    GLint color_location = glGetUniformLocation(openGLProgram_paintColor, "color");
    float r = (float)(color & 255)/255.f;
    float g = (float)((color>>8) & 255)/255.f;
    float b = (float)((color>>16) & 255)/255.f;
    float a = (float)((color>>24) & 255)/255.f;
    glUniform4f(color_location,r,g,b,a);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glUseProgram(0);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_lucyteandroid_MyGLRenderer_drawLine(JNIEnv *env, jobject thiz, jfloat x0,
                                                          jfloat y0, jfloat x1, jfloat y1,
                                                          jint color, jint screen_width,
                                                          jint screen_height) {
    x0 /= (float)screen_width;
    y0 /= (float)screen_height;
    x1 /= (float)screen_width;
    y1 /= (float)screen_height;
    x0 = x0 * 2.f - 1.f;
    y0 = (y0 * 2.f - 1.f) * -1.f;
    x1 = x1 * 2.f - 1.f;
    y1 = (y1 * 2.f - 1.f) * -1.f;
    glViewport(0,0,screen_width, screen_height);

    glUseProgram(openGLProgram_paintColor);
    float points[] = {
            x0,  y0,  0.0f,
            x1, y1,  0.0f,
    };
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * 2 * sizeof(float), points, GL_STATIC_DRAW);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    GLint color_location = glGetUniformLocation(openGLProgram_paintColor, "color");
    float r = (float)(color & 255)/255.f;
    float g = (float)((color>>8) & 255)/255.f;
    float b = (float)((color>>16) & 255)/255.f;
    float a = (float)((color>>24) & 255)/255.f;
    glUniform4f(color_location,r,g,b,a);
    glDrawArrays(GL_LINES, 0, 2);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glUseProgram(0);
}