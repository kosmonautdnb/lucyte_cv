// Lucyte Created on: 03.02.2025 by Stefan Mader
#include <GL/glew.h>
#include "opengl_refinement.hpp"

GLFWwindow* window = NULL;
int window_width;
int window_height;

static const int MAXMIPMAPS = 32;
static const int MIPMAPIDCOUNT = 8;
static const int KEYPOINTIDCOUNT = 2;
static const int DESCRIPTORIDCOUNT = 2;
static const int maxTextureSize = 1024;

static GLuint openGLDescriptorShape[4] = { 0 };
static int openGLDescriptorShapeTextureWidth = 0;
static int openGLDescriptorShapeTextureHeight = 0;
static GLuint openGLMipMaps[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLMipMapsTextureWidth[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLMipMapsTextureHeight[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static GLuint openGLKeyPointsX[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLKeyPointsY[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureWidth[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureHeight[KEYPOINTIDCOUNT] = { 0 };
static std::pair<GLuint, GLuint> openGLDescriptorRenderTarget[KEYPOINTIDCOUNT] = { {0,0} };
static GLuint openGLDescriptorRenderTargetWidth[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLDescriptorRenderTargetHeight[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLDescriptors[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLDescriptorsTextureWidth[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLDescriptorsTextureHeight[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static GLuint openGLFragmentShader_sampleDescriptors = 0;
static GLuint openGLFragmentShader_refineKeyPoints = 0;
static GLuint openGLVertexShader_basic = 0;
static GLuint openGLProgram_sampleDescriptors = 0;
static GLuint openGLProgram_refineKeyPoints = 0;
const std::string sampleDescriptorProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
"in vec2 p;\n"
"out uvec4 frag_color;\n"
"uniform int maxTextureSize;\n"
"uniform int texWidthDescriptor;\n"
"uniform int texHeightDescriptor;\n"
"uniform int texWidthMipMap;\n"
"uniform int texHeightMipMap;\n"
"uniform int texWidthKeyPoints;\n"
"uniform int texHeightKeyPoints;\n"
"uniform int DESCRIPTORSIZE;\n"
"uniform float descriptorScale;\n"
"uniform float mipScale;\n"
"uniform sampler2D descriptor1x;\n"
"uniform sampler2D descriptor1y;\n"
"uniform sampler2D descriptor2x;\n"
"uniform sampler2D descriptor2y;\n"
"uniform sampler2D mipMap;\n"
"uniform sampler2D keyPointsx;\n"
"uniform sampler2D keyPointsy;\n"
"float t2d_nearest(sampler2D t, int x, int y, int width, int height)\n"
"{\n"
"   return textureLod(t,vec2((float(x)+0.25)/float(width),(float(y)+0.25)/float(height)),0.0).x;\n"
"}\n"
"vec2 descriptors1(int i) {\n"
"   float x = t2d_nearest(descriptor1x, i % maxTextureSize, i / maxTextureSize, texWidthDescriptor, texHeightDescriptor);\n"
"   float y = t2d_nearest(descriptor1y, i % maxTextureSize, i / maxTextureSize, texWidthDescriptor, texHeightDescriptor);\n"
"   return vec2(x,y);\n"
"}\n"
"vec2 descriptors2(int i) {\n"
"   float x = t2d_nearest(descriptor2x, i % maxTextureSize, i / maxTextureSize, texWidthDescriptor, texHeightDescriptor);\n"
"   float y = t2d_nearest(descriptor2y, i % maxTextureSize, i / maxTextureSize, texWidthDescriptor, texHeightDescriptor);\n"
"   return vec2(x,y);\n"
"}\n"
"void main()\n"
"{\n"
"       int txx = int(floor(p.x+0.25));\n"
"       int txy = int(floor(p.y+0.25));\n"
"       float descriptorSize = descriptorScale * mipScale;\n"
"       vec2 kp = vec2(t2d_nearest(keyPointsx,txx,txy,texWidthKeyPoints,texHeightKeyPoints),t2d_nearest(keyPointsy,txx,txy,texWidthKeyPoints,texHeightKeyPoints));\n"
"       vec2 sc = vec2(1.0/float(texWidthMipMap),1.0/float(texHeightMipMap));\n"
"       unsigned int b[4]; b[0]=unsigned int(0); b[1]=unsigned int(0); b[2]=unsigned int(0); b[3]=unsigned int(0);\n"
"       for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
"       {\n"
"           float l1 = textureLod(mipMap, (kp + descriptors1(i) * descriptorSize) * sc, 0.0).x;\n"
"           float l2 = textureLod(mipMap, (kp + descriptors2(i) * descriptorSize) * sc, 0.0).x;\n"
"           b[i>>5] |= l2 < l1 ? unsigned int(1<<(i & 31)) : unsigned int(0);\n"
"       }\n"
"       frag_color = uvec4(b[0],b[1],b[2],b[3]);\n"
"}\n"
"\n";
const std::string refineKeyPointsProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
"out vec4 frag_color;\n"
"void main()\n"
"{\n"
"    frag_color = vec4(1.0,0.5,0.2,1.0);\n"
"}\n"
"\n";
const std::string basicProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
"layout(location = 0) in vec2 pos;\n"
"uniform int texWidthKeyPoints;\n"
"uniform int texHeightKeyPoints;\n"
"out vec2 p;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(pos.xy, 0.0, 1.0);\n"
"    p = ((pos.xy + 1.0) * 0.5) * vec2(float(texWidthKeyPoints),float(texHeightKeyPoints));\n"
"}\n"
"\n";

void checkGLError() {
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        printf("OpenGL error(%d): %s\n", error, glewGetErrorString(error));
        int debug = 1;
    }
}

void checkShader(GLuint shader) {
    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled); checkGLError();
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength); checkGLError();
        std::vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]); checkGLError();
        printf("%s", &(errorLog[0]));
        glDeleteShader(shader); checkGLError();
        exit(0);
    }
}

void checkProgram(GLuint program) {
    GLint isCompiled = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isCompiled); checkGLError();
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength); checkGLError();
        std::vector<char> errorLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]); checkGLError();
        printf("%s", &(errorLog[0]));
        glDeleteProgram(program); checkGLError();
        exit(0);
    }
}

GLuint pixelShader(const char *shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint vertexShader(const char* shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_VERTEX_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint program(const GLuint vertexShader, const GLuint pixelShader) {
    GLuint program = glCreateProgram(); checkGLError();
    glAttachShader(program, vertexShader); checkGLError();
    glAttachShader(program, pixelShader); checkGLError();
    glLinkProgram(program); checkGLError();
    checkProgram(program);
    return program;
}

void createUint32RenderTarget(std::pair<GLuint, GLuint> &dest, int width, int height, bool nearest) {
    glDeleteFramebuffers(1, &dest.first); checkGLError();
    glDeleteTextures(1, &dest.second);
    GLuint framebufferId = 0;
    glGenFramebuffers(1, &framebufferId); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferId); checkGLError();
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0); checkGLError();
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
    dest = { framebufferId, renderedTexture };
}


void glfwErrorCallback(int error_code, const char* description) {

}

void initOpenGL() {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        exit(0);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(1920, 1080, "Lucyte", NULL, NULL);
    if (window == NULL)
        exit(0);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewInit();
    openGLFragmentShader_sampleDescriptors = pixelShader(sampleDescriptorProgram.c_str(), sampleDescriptorProgram.length());
    openGLFragmentShader_refineKeyPoints = pixelShader(refineKeyPointsProgram.c_str(), refineKeyPointsProgram.length());
    openGLVertexShader_basic = vertexShader(basicProgram.c_str(), basicProgram.length());
    openGLProgram_sampleDescriptors = program(openGLVertexShader_basic, openGLFragmentShader_sampleDescriptors);
    openGLProgram_refineKeyPoints = program(openGLVertexShader_basic, openGLFragmentShader_refineKeyPoints);
}

void upload2DFloatTexture(const GLuint& tex, const float* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload2DUnsignedCharTexture(const GLuint& tex, const float* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload1DFloatTexture(const GLuint& tex, const float* data, const unsigned int count, int& width, int& height, const bool nearest) {
    width = count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
        float* data2 = new float[width * height];
        memcpy(data2, data, count * sizeof(float));
        upload2DFloatTexture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2DFloatTexture(tex, data, width, height, nearest);
}

void upload2Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_INT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload1Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int count, int &width, int &height, const bool nearest) {
    width = count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
        unsigned int* data2 = new unsigned int[width * height];
        memcpy(data2, data, count * sizeof(unsigned int));
        upload2Duint32Texture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2Duint32Texture(tex, data, width, height, nearest);
}

void uploadDescriptorShape_openGL() {
    glDeleteTextures(4, openGLDescriptorShape);
    glGenTextures(4, openGLDescriptorShape);
    upload1DFloatTexture(openGLDescriptorShape[0], descriptorsX1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[1], descriptorsY1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[2], descriptorsX2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[3], descriptorsY2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
}

void uploadMipMaps_openGL(const int mipmapsId, const std::vector<cv::Mat>& mipMaps) {
    glDeleteTextures(MAXMIPMAPS, openGLMipMaps[mipmapsId]);
    glGenTextures(mipMaps.size(), openGLMipMaps[mipmapsId]);
    for (int i = 0; i < mipMaps.size(); i++) {
        openGLMipMapsTextureWidth[mipmapsId][i] = mipMaps[i].cols;
        openGLMipMapsTextureHeight[mipmapsId][i] = mipMaps[i].rows;
        upload2DUnsignedCharTexture(openGLMipMaps[mipmapsId][i], (float*)(mipMaps[i].data), mipMaps[i].cols, mipMaps[i].rows, false);
    }
}

void uploadKeyPoints_openGL(const int keyPointsId, const std::vector<KeyPoint>& keyPoints) {
    std::vector<float> kpx; kpx.resize(keyPoints.size());
    std::vector<float> kpy; kpy.resize(keyPoints.size());
    for (int i = int(keyPoints.size()) - 1; i >= 0; i--) {
        kpx[i] = keyPoints[i].x;
        kpy[i] = keyPoints[i].y;
    }
    glDeleteTextures(1, &(openGLKeyPointsX[keyPointsId]));
    glDeleteTextures(1, &(openGLKeyPointsY[keyPointsId]));
    glGenTextures(1, &(openGLKeyPointsX[keyPointsId]));
    glGenTextures(1, &(openGLKeyPointsY[keyPointsId]));
    
    const int widthBefore = openGLKeyPointsTextureWidth[keyPointsId];
    const int heightBefore = openGLKeyPointsTextureHeight[keyPointsId];
    
    upload1DFloatTexture(openGLKeyPointsX[keyPointsId], &(kpx[0]), keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    upload1DFloatTexture(openGLKeyPointsY[keyPointsId], &(kpy[0]), keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    
    if (openGLDescriptorRenderTarget[keyPointsId].first == 0 || widthBefore != openGLKeyPointsTextureWidth[keyPointsId] || heightBefore != openGLKeyPointsTextureHeight[keyPointsId]) {
        createUint32RenderTarget(openGLDescriptorRenderTarget[keyPointsId], openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    }
}

void uploadDescriptors_openGL(const int descriptorsId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    const std::vector<Descriptor>& descriptors = sourceMips[mipMap];
    std::vector<unsigned int> bits; bits.resize(Descriptor::uint32count * descriptors.size());
    for (int i = int(descriptors.size()) - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            bits[j + i * Descriptor::uint32count] = descriptors[i].bits[j];
        }
    }
    glDeleteTextures(1, &(openGLDescriptors[descriptorsId][mipMap]));
    glGenTextures(1, &(openGLDescriptors[descriptorsId][mipMap]));
    upload1Duint32Texture(openGLDescriptors[descriptorsId][mipMap], &(bits[0]), bits.size(), openGLDescriptorsTextureWidth[descriptorsId][mipMap], openGLDescriptorsTextureHeight[descriptorsId][mipMap],true);
}

void sampleDescriptors_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const float descriptorScale, const int width, const int height, const float mipScale) {

    GLuint sampleDescriptors_location_maxTextureSize = glGetUniformLocation(openGLProgram_sampleDescriptors, "maxTextureSize"); checkGLError();
    GLuint sampleDescriptors_location_texWidthDescriptor = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthDescriptor"); checkGLError();
    GLuint sampleDescriptors_location_texHeightDescriptor = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightDescriptor"); checkGLError();
    GLuint sampleDescriptors_location_texWidthMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthMipMap"); checkGLError();
    GLuint sampleDescriptors_location_texHeightMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightMipMap"); checkGLError();
    GLuint sampleDescriptors_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthKeyPoints"); checkGLError();
    GLuint sampleDescriptors_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightKeyPoints"); checkGLError();
    GLuint sampleDescriptors_location_DESCRIPTORSIZE = glGetUniformLocation(openGLProgram_sampleDescriptors, "DESCRIPTORSIZE"); checkGLError();
    GLuint sampleDescriptors_location_descriptorScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptorScale"); checkGLError();
    GLuint sampleDescriptors_location_mipScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipScale"); checkGLError();

    glUseProgram(openGLProgram_sampleDescriptors); checkGLError();
    glUniform1i(sampleDescriptors_location_maxTextureSize, maxTextureSize); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthDescriptor, openGLDescriptorsTextureWidth[descriptorsId][mipMap]); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightDescriptor, openGLDescriptorsTextureHeight[descriptorsId][mipMap]); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthMipMap, openGLMipMapsTextureWidth[mipmapsId][mipMap]); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightMipMap, openGLMipMapsTextureHeight[mipmapsId][mipMap]); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthKeyPoints, openGLKeyPointsTextureWidth[keyPointsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightKeyPoints, openGLKeyPointsTextureHeight[keyPointsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_DESCRIPTORSIZE, DESCRIPTORSIZE); checkGLError();
    glUniform1f(sampleDescriptors_location_descriptorScale, descriptorScale); checkGLError();
    glUniform1f(sampleDescriptors_location_mipScale, mipScale); checkGLError();
    
    glBindFramebuffer(GL_FRAMEBUFFER, openGLDescriptorRenderTarget[keyPointsId].first); checkGLError();
    glRects(-1, -1, 1, 1);
    unsigned int* data = new unsigned int[4 * openGLKeyPointsTextureWidth[keyPointsId] * openGLKeyPointsTextureHeight[keyPointsId]];
    glReadPixels(0, 0, openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], GL_RGBA_INTEGER, GL_UNSIGNED_INT, data); checkGLError();
    delete[] data;
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
}
