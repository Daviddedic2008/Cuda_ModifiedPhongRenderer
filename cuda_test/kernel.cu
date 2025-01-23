
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <time.h>
#include <glad/glad.h>
#include <glfw3.h>
#include <string.h>
#include <fstream>
#include <sstream>

#define fov 1.0f
#define scr_w 512
#define scr_h 512

#define num_triangles 100000
#define max_streams 512

cudaError_t ercall;
cudaError_t err;
#define CCALL(call) ercall = call; if(cudaSuccess != ercall){fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(ercall)); exit(EXIT_FAILURE);}
#define printLastErrorCUDA() err = cudaGetLastError(); if(err != cudaSuccess){printf("%s\n",cudaGetErrorString(err));}

#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define dot2D(vec2_v1, vec2_v2) (vec2_v1.x * vec2_v2.x + vec2_v1.y * vec2_v2.y)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define matgnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))
#define magnitude2D(vec2_a) (sqrtf(dot2D(vec2_a, vec2_a)))


// init of structs and methods as well as global vars and respective functions and macros
//*****************************************************************************************************************************************************************************************

// too lazy to set up cudas rng so i use this bad one
inline __host__ __device__ long int xorRand(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

struct vec2 {
    float x, y;

    __host__ __device__ vec2() : x(0.0f), y(0.0f) {}
    __host__ __device__ vec2(float X, float Y) : x(X), y(Y) {}

    inline __host__ __device__ vec2 operator+(const vec2& f) const {
        return vec2(x + f.x, y + f.y);
    }

    inline __host__ __device__ vec2 operator-(const vec2& f) const {
        return vec2(x - f.x, y - f.y);
    }

    inline __host__ __device__ vec2 operator*(const float scalar) const {
        return vec2(x * scalar, y * scalar);
    }

    inline __host__ __device__ vec2 normalize() {
        const float scl = magnitude2D((*this));
        return vec2(x / scl, y / scl);
    }

    inline __host__ __device__ float dist_from_vec(vec2& v) {
        const float addx = (x + v.x);
        const float addy = (y + v.y);
        return sqrtf(addx*addx + addy*addy);
    }
};

// Define the vec3 struct
struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    inline __host__ __device__ vec3 operator+(const vec3& f) const {
        return vec3(x + f.x, y + f.y, z + f.z);
    }

    inline __host__ __device__ vec3 operator-(const vec3& f) const {
        return vec3(x - f.x, y - f.y, z - f.z);
    }

    inline __host__ __device__ vec3 operator*(const float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    inline __host__ __device__ vec3 normalize() {
        const float scl = matgnitude((*this));
        return vec3(x / scl, y / scl, z / scl);
    }

    inline __host__ __device__ vec2 convert_vec2() {
        return vec2(x / (z * fov), y / (z * fov));
    }
};

// cross is more logical as its own function

inline __host__ __device__ vec3 cross(const vec3 v1, const vec3 v2) {
    vec3 ret;
    ret.x = matrix2D_eval(v1.y, v1.z, v2.y, v2.z);
    ret.y = matrix2D_eval(v1.x, v1.z, v2.x, v2.z);
    ret.z = matrix2D_eval(v1.x, v1.y, v2.x, v2.y);
    return ret;
}

inline __host__ __device__ float get_max(float f1, float f2) {
    return (f1 > f2) * f1 + (f1 <= f2) * f2;
}

inline __host__ __device__ float get_min(float f1, float f2) {
    return (f1 < f2) * f1 + (f1 >= f2) * f2;
}

struct bounding_box{
    vec2 min, max;

    __host__ __device__ bounding_box(){}

    __host__ __device__ bounding_box(const float minx, const float maxx, const float miny, const float maxy) : min(vec2(minx, miny)), max(vec2(maxx, maxy)){}
};

typedef struct {
    float a, b, c;
}barycentric_return;

struct triangle2D {
    vec2 p1, p2, p3;
    float denom, y2_y3, x1_x3, x3_x2, y3_y1;
    bounding_box bound_box;

    __host__ __device__ triangle2D() {}

    inline __host__ __device__ void calc_denom_and_vals() {
        denom = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
        y2_y3 = p2.y - p3.y;
        x1_x3 = p1.x - p3.x;
        x3_x2 = p3.x - p2.x;
        y3_y1 = p3.y - p1.y;
        float minx = get_min(p1.x, p2.x);
        minx = get_min(minx, p3.x);

        float maxx = get_max(p1.x, p2.x);
        maxx = get_max(maxx, p3.x);

        float miny = get_min(p1.y, p2.y);
        miny = get_min(miny, p3.y);

        float maxy = get_max(p1.y, p2.y);
        maxy = get_max(maxy, p3.y);

        bound_box = bounding_box(minx, maxx, miny, maxy);
    }

    __host__ __device__ triangle2D(const vec2 P1, const vec2 P2, const vec2 P3) {
        p1 = P1; p2 = P2; p3 = P3;
    }

    inline __host__ __device__ barycentric_return point_in_triangle(const vec2 p, int seed) const {
        const float x3m = p.x - p3.x;
        const float y3m = p.y - p3.y;

        const float a = (y2_y3 * x3m + x3_x2 * y3m) / denom;
        const float b = (y3_y1 * x3m + x1_x3 * y3m) / denom;
        const float c = 1.0f - a - b;

        barycentric_return r;
        r.a = a; r.b = b; r.c = c;
        return r;
    }
};


struct triangle {
    vec3 p1, p2, p3;
    vec3 nv;
    vec3 sb21, sb31;
    float dot2121, dot2131, dot3131;

    __host__ __device__ triangle() : p1(vec3(0.0f, 0.0f, 0.0f)), p2(vec3(0.0f, 0.0f, 0.0f)), p3(vec3(0.0f, 0.0f, 0.0f)) {}

    __host__ __device__ triangle(const vec3 P1, const vec3 P2, const vec3 P3) {
        p1 = P1;
        p2 = P2;
        p3 = P3;
        sb21 = p2 - p1;
        sb31 = p3 - p1;
        dot2121 = dot(sb21, sb21);
        dot2131 = dot(sb21, sb31);
        dot3131 = dot(sb31, sb31);
        nv = cross(sb21, sb31).normalize();
    }

    inline __host__ __device__ triangle2D convert_to_2D() {
        return triangle2D(p1.convert_vec2(), p2.convert_vec2(), p3.convert_vec2());
    }
};

struct color {
    float r, g, b;

    __host__ __device__ color(){}
    __host__ __device__ color(const float R, const float G, const float B) : r(R), g(G), b(B){}

    inline __host__ __device__ color operator+(const color& c) {
        return color(r + c.r, g + c.g, b + c.b);
    }

    inline __host__ __device__ color operator*(const float f) {
        return color(r * f, g * f, b * f);
    }

    inline __host__ __device__ color operator*(const bool f) {
        return color(r * f, g * f, b * f);
    }
};

// global device screen buffer
__device__ char screen_buffer[sizeof(color) * scr_w * scr_h];

__device__ float depth_buffer[scr_w * scr_h];

// global device array of all triangles
__device__ char triangles[sizeof(triangle) * num_triangles];

__global__ void fillPixels(const int minx, const int maxx, const int miny, const int maxy, color c, triangle2D t2D, const vec3 z_coords) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    const int x = id % (maxx - minx) + minx;
    const int y = id / (maxx - minx) + miny;

    vec2 v = vec2(x, y);

    const barycentric_return r = t2D.point_in_triangle(vec2(x, y), 10 * id);

    const float z = z_coords.x * r.a + z_coords.y * r.b + z_coords.z * r.c;

    const float d = depth_buffer[x + y * scr_w];

    depth_buffer[x + y * scr_w] = (d == 0) * z + (d > z) * z + (d != 0 && d > z) * d;

    if (r.a >= -0.001f && r.b >= -0.001f && r.c >= -0.001f) {
        ((color*)screen_buffer)[x + y * scr_w] = color(z / 10.0f, z / 10.0f, z / 10.0f);
    }
}

// rasterization function tests
__global__ void rasterize_triangles_single_thread(color c) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_triangles) { return; }
    triangle t = ((triangle*)triangles)[index];
    triangle2D t2D = t.convert_to_2D();
    const vec2 tmp = vec2(scr_w / 2, scr_h / 2);
    t2D.p1 = t2D.p1 + tmp;
    t2D.p2 = t2D.p2 + tmp;
    t2D.p3 = t2D.p3 + tmp;
    
    t2D.calc_denom_and_vals();

    const vec3 z_coords = vec3(t.p1.z, t.p2.z, t.p3.z);

    const int p_minx = (int)t2D.bound_box.min.x;
    const int p_miny = (int)t2D.bound_box.min.y;
    const int p_maxx = (int)t2D.bound_box.max.x;
    const int p_maxy = (int)t2D.bound_box.max.y;

    fillPixels << <512, (p_maxx - p_minx) * (p_maxy - p_miny) / 512+1 >> > (p_minx, p_maxx, p_miny, p_maxy, c, t2D, z_coords);
}

int clamp(int i) {
    return (i < max_streams) ? i : max_streams;
}

// being worked on
/*
void rasterize_all_triangles_multi_thread() {
    cudaStream_t streams[max_streams];

    const int num_iterations = num_triangles / max_streams + 1;

    int total_tris = 0;

    for (int i = 0; i < num_iterations; i++) {
        const int num_streams = clamp((num_triangles - total_tris));
        total_tris += num_streams;
        for (int s = 0; s < num_streams; s++) {
            cudaStreamCreate(&streams[s]);
            rasterize_triangle_multi_thread<<<512, >>>(s, color(1.0f, 0.0f, 0.0f));
        }
    }
}
*/

void rasterize_all_triangles(color c) {
    rasterize_triangles_single_thread << <1024, num_triangles / 1024 + 1 >> > (c);
    cudaDeviceSynchronize();
}

void add_triangle(vec3 p1, vec3 p2, vec3 p3, int idx) {
    triangle t = triangle(p1, p2, p3);
    cudaMemcpyToSymbol(triangles, &t, sizeof(triangle), sizeof(triangle) * idx);
}

//*****************************************************************************************************************************************************************************************
// opengl stuff
// draws 2 triangles at z=0 and textures them with the pixel colors outputted by the cuda program
// no interop, data transfers from GPU to CPU each frame

char cpu_colors[sizeof(color) * scr_w * scr_h];

void copyBufferToCPU() {
    cudaMemcpyFromSymbol(cpu_colors, screen_buffer, sizeof(color) * scr_w * scr_h);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
unsigned int SCR_WIDTH = scr_w;
unsigned int SCR_HEIGHT = scr_h;


char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aColor;\n"
"layout(location = 2) in vec2 aTexCoord;\n"
"out vec3 ourColor;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   ourColor = aColor;\n"
"   TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
"}\0";

char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D texture1;\n"
"void main()\n"
"{\n"
"   FragColor = texture(texture1, TexCoord);\n"
"}\n\0";

float truncate(float f) {
    return fabs(1.0 / (1.0 + exp(-1.0 * f))-0.5) * 2.0f;
}

int main()
{
    // add triangles
    for (int t = 0; t < num_triangles; t++) {
        add_triangle(vec3(-50.0f, -43.3f, 1.0f), vec3(50.0f, -43.3f, 1.0f), vec3(0.0f, 43.3f, 10.0f), t);
    }
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(scr_w, scr_h, "Cuda-openGL Interop", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        return -1;
    }
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::FRAGMENT::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

    unsigned int texture1;
    //uint8_t pixels[grid_h * grid_l * 3];
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scr_w, scr_h, 0, GL_RGB, GL_FLOAT, cpu_colors);
	glGenerateMipmap(GL_TEXTURE_2D);
    glUseProgram(shaderProgram); 
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int frametime = 0;
    unsigned int frame = 0;
    
    while (!glfwWindowShouldClose(window))
    {
        cudaEventRecord(start);
        processInput(window);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        //glClear(GL_COLOR_BUFFER_BIT);
        int ind = 0;
        // **
        // dodaj boje tu u pixels
        rasterize_all_triangles(color(1.0f, 0.0f, 0.0f));
        copyBufferToCPU();
        // **
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scr_w, scr_h, 0, GL_RGB, GL_FLOAT, cpu_colors);
		glGenerateMipmap(GL_TEXTURE_2D);
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);

        // render container
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
        cudaDeviceSynchronize();
        if (frame % 10 == 1) {
            int fps = 10000 / (frametime);
            printf("%d\n", fps);
            frametime = 0;
        }
        cudaDeviceSynchronize();
        cudaEventRecord(end);
        frame++;
        float milis;
        cudaEventElapsedTime(&milis, start, end);
        frametime += milis;
        printLastErrorCUDA()
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}