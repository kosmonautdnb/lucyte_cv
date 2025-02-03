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

GLuint openGLDescriptorShape[4] = {0};
GLuint openGLMipMaps[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
GLuint openGLKeyPointsX[KEYPOINTIDCOUNT] = { 0 };
GLuint openGLKeyPointsY[KEYPOINTIDCOUNT] = { 0 };
GLuint openGLDescriptors[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };

GLuint openGLFragmentShader_sampleDescriptors;
GLuint openGLFragmentShader_refineKeyPoints;
GLuint openGLVertexShader_basic;
GLuint openGLProgram_sampleDescriptors;
GLuint openGLProgram_refineKeyPoints;
const std::string sampleDescriptorProgram = "#version 300 es\nprecision highp float;\n"
"out vec4 frag_color;\n"
"void main()\n"
"{\n"
"    frag_color = vec4(1.0,0.5,0.2,1.0);\n"
"}\n"
"\n";
const std::string refineKeyPointsProgram = "#version 300 es\nprecision highp float;\n"
"out vec4 frag_color;\n"
"void main()\n"
"{\n"
"    frag_color = vec4(1.0,0.5,0.2,1.0);\n"
"}\n"
"\n";
const std::string basicProgram = "#version 300 es\n"
"layout(location = 0) in vec2 pos;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(pos.xy, 0.0, 1.0);\n"
"}\n"
"\n";

void checkShader(GLuint shader) {
    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
        std::vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);
        printf("%s", &(errorLog[0]));
        glDeleteShader(shader);
        exit(0);
    }
}

void checkProgram(GLuint program) {
    GLint isCompiled = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
        std::vector<char> errorLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]);
        printf("%s", &(errorLog[0]));
        glDeleteProgram(program);
        exit(0);
    }
}

GLuint pixelShader(const char *shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader, 1, &shaderProgram, length);
    glCompileShader(shader);
    checkShader(shader);
    return shader;
}

GLuint vertexShader(const char* shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader, 1, &shaderProgram, length);
    glCompileShader(shader);
    checkShader(shader);
    return shader;
}

GLuint program(const GLuint vertexShader, const GLuint pixelShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, pixelShader);
    glLinkProgram(program);
    checkProgram(program);
    return program;
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
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void upload1DFloatTexture(const GLuint& tex, const float* data, const unsigned int count, const bool nearest) {
    int width = count;
    int height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
    }
    upload2DFloatTexture(tex, data, width, height, nearest);
}

void upload2Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_INT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void upload1Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int count, const bool nearest) {
    int width = count;
    int height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
    }
    upload2Duint32Texture(tex, data, width, height, nearest);
}

void uploadDescriptorShape_openGL() {
    glDeleteTextures(4, openGLDescriptorShape);
    glGenTextures(4, openGLDescriptorShape);
    upload1DFloatTexture(openGLDescriptorShape[0], descriptorsX1, DESCRIPTORSIZE, true);
    upload1DFloatTexture(openGLDescriptorShape[1], descriptorsY1, DESCRIPTORSIZE, true);
    upload1DFloatTexture(openGLDescriptorShape[2], descriptorsX2, DESCRIPTORSIZE, true);
    upload1DFloatTexture(openGLDescriptorShape[3], descriptorsY2, DESCRIPTORSIZE, true);
}

void uploadMipMaps_openGL(const int mipmapsId, const std::vector<cv::Mat>& mipMaps) {
    glDeleteTextures(MAXMIPMAPS, openGLMipMaps[mipmapsId]);
    glGenTextures(mipMaps.size(), openGLMipMaps[mipmapsId]);
    for (int i = 0; i < mipMaps.size(); i++) {
        upload2DFloatTexture(openGLMipMaps[mipmapsId][i], (float*)(mipMaps[i].data), mipMaps[i].cols, mipMaps[i].rows, false);
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
    upload1DFloatTexture(openGLKeyPointsX[keyPointsId], &(kpx[0]), keyPoints.size(), true);
    upload1DFloatTexture(openGLKeyPointsY[keyPointsId], &(kpy[0]), keyPoints.size(), true);
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
    upload1Duint32Texture(openGLDescriptors[descriptorsId][mipMap], &(bits[0]), bits.size(), true);
}