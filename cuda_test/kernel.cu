
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
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
#include <iostream>
#include <unordered_map>
#include <vector>
#include <list>
#include <algorithm>
#include <cuda_fp16.h>
#include <thread>

#define fov 0.0035f
#define scr_w 512
#define scr_h 512

#define num_triangles 207
#define max_streams 512

cudaError_t ercall;
cudaError_t err;
#define CCALL(call) ercall = call; if(cudaSuccess != ercall){fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(ercall)); exit(EXIT_FAILURE);}
#define printLastErrorCUDA() err = cudaGetLastError(); if(err != cudaSuccess){printf("%s\n",cudaGetErrorString(err));}

#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define dot_fastmath(vec3_v1, vec3_v2) __fmaf_rn(vec3_v1.x, vec3_v2.x,__fmaf_rn(vec3_v1.y, vec3_v2.y,vec3_v1.z * vec3_v2.z))
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

    inline __device__ float dist_from_vec(vec2& v) {
        const float addx = (x + v.x);
        const float addy = (y + v.y);
        return __fsqrt_rn(__fmaf_rn(addx,addx,addy*addy));
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

    inline __host__ __device__ bool operator==(const vec3& f) const {
        return fabs(x - f.x) < 0.01f && fabs(y - f.y) < 0.01f && fabs(z - f.z) < 0.01f;
    }

    inline __host__ __device__ vec2 convert_vec2() const {
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

struct bounding_box{
    vec2 min, max;

    __device__ bounding_box(){}

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
        denom = 1.0f/((p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y));
        y2_y3 = (p2.y - p3.y) * denom;
        x1_x3 = (p1.x - p3.x) * denom;
        x3_x2 = (p3.x - p2.x) * denom;
        y3_y1 = (p3.y - p1.y) * denom;
        float minx = fminf(p1.x, p2.x);
        minx = fminf(minx, p3.x);

        float maxx = fmaxf(p1.x, p2.x);
        maxx = fmaxf(maxx, p3.x);

        float miny = fminf(p1.y, p2.y);
        miny = fminf(miny, p3.y);

        float maxy = fmaxf(p1.y, p2.y);
        maxy = fmaxf(maxy, p3.y);

        bound_box = bounding_box(minx-1, maxx+1, miny-1, maxy+1);
    }

    __host__ __device__ triangle2D(const vec2 P1, const vec2 P2, const vec2 P3) {
        p1 = P1; p2 = P2; p3 = P3;
    }

    inline __device__ barycentric_return point_in_triangle(const vec2 p, int seed) const {
        const float x3m = p.x - p3.x;
        const float y3m = p.y - p3.y;

        barycentric_return r;
        r.a = (y2_y3 * x3m + x3_x2 * y3m); r.b = (y3_y1 * x3m + x1_x3 * y3m); r.c = 1.0f - r.a - r.b;
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

    inline __host__ __device__ triangle2D convert_to_2D() const {
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

// vertex norms
__constant__ char vertex_norms[sizeof(vec3) * num_triangles * 3];

typedef struct {
    vec3 v1, v2, v3;
}triplevec3;

inline __device__ float barycentric_float_interpolation(const vec3 floats, const barycentric_return r) {
    return floats.x * r.a + floats.y * r.b + floats.z * r.c;
}

__global__ void fillPixels(triangle2D t2D, const vec3 z_coords, const triplevec3 n) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    const int tmp = (t2D.bound_box.max.x - t2D.bound_box.min.x);

    const int tmp2 = (t2D.bound_box.max.y - t2D.bound_box.min.y);

    const int x = id % tmp + t2D.bound_box.min.x;
    const int y = id / tmp + t2D.bound_box.min.y;

    const barycentric_return r = t2D.point_in_triangle(vec2(x, y), 10 * id);

    const float scl = -1.0f / ((tmp < tmp2) * tmp + (tmp > tmp2) * tmp2);

    const float z = __fmaf_rn(z_coords.x, r.a, __fmaf_rn(z_coords.y, r.b, z_coords.z * r.c));

    float d;
    atomicExch(&d, depth_buffer[x + y * scr_w]);

    if (!(r.a >= scl && r.b >= scl && r.c >= scl && (d == 0 || d > z))) {
        return;
    }

    const vec3 interpolated_norm = vec3(__fmaf_rn(n.v1.x, r.a, __fmaf_rn(n.v2.x, r.b, n.v3.x * r.c)), __fmaf_rn(n.v1.y, r.a, __fmaf_rn(n.v2.y, r.b, n.v3.y * r.c)), __fmaf_rn(n.v1.z, r.a, __fmaf_rn(n.v2.z, r.b, n.v3.z * r.c)));
    //((color*)screen_buffer)[x + y * scr_w] = color(r.a, r.b, r.c);
    atomicExch(&(depth_buffer[x + y * scr_w]), z);
    //depth_buffer[x + y * scr_w] = z;
    //((color*)screen_buffer)[x + y * scr_w] = color(1.0f, (z-22) / 20.0f, (z-22) / 20.0f);
    ((color*)screen_buffer)[x + y * scr_w] = color(interpolated_norm.x, interpolated_norm.y, interpolated_norm.z);
}

// rasterization function tests
__global__ void rasterize_triangles_single_thread() {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_triangles) { return; }
    const triangle t = ((triangle*)triangles)[index];
    triangle2D t2D = t.convert_to_2D();

    const vec2 tmp = vec2(scr_w / 2, scr_h / 2);

    t2D.p1 = t2D.p1 + tmp;
    t2D.p2 = t2D.p2 + tmp;
    t2D.p3 = t2D.p3 + tmp;
    
    t2D.calc_denom_and_vals();

    __syncthreads();
    fillPixels << <32, ((int)t2D.bound_box.max.x - (int)t2D.bound_box.min.x) * ((int)t2D.bound_box.max.y - (int)t2D.bound_box.min.y) / 32 + 1 >> > (t2D, vec3(t.p1.z, t.p2.z, t.p3.z), ((triplevec3*)vertex_norms)[index]);
    

}

// rasterization with cpu-gpu load distribution. useful if system doesnt support nested kernels

triangle trianglesCPU[num_triangles];

void rasterizeTrianglesThread(const int threadId, const int trianglesToInit, const int start_idx) {
    for (int tind = start_idx; tind < start_idx + trianglesToInit; tind++) {
        triangle2D t2D = trianglesCPU[tind].convert_to_2D();
        triangle t = trianglesCPU[tind];
        const vec2 tmp = vec2(scr_w / 2, scr_h / 2);

        t2D.p1 = t2D.p1 + tmp;
        t2D.p2 = t2D.p2 + tmp;
        t2D.p3 = t2D.p3 + tmp;
        t2D.calc_denom_and_vals();

        triplevec3 tv3;
        cudaMemcpyFromSymbol(&tv3, vertex_norms, sizeof(triplevec3), tind * sizeof(triplevec3));

        fillPixels << <32, ((int)t2D.bound_box.max.x - (int)t2D.bound_box.min.x)* ((int)t2D.bound_box.max.y - (int)t2D.bound_box.min.y) / 32 + 1 >> > (t2D, vec3(t.p1.z, t.p2.z, t.p3.z), tv3);
    }
}

void rasterizeAllTrianglesCPUandGPU() {
    const int threadLimit = std::thread::hardware_concurrency() * 20;
    int triangles_called = 0;

    const int trianglesPerThread = num_triangles / threadLimit + 1;

    const int num_calls = num_triangles / threadLimit + 1;
    std::vector<std::thread> threads(num_calls);
    for (int c = 0; c < num_calls; c++) {
        int trianglesToRender = num_triangles < (triangles_called + trianglesPerThread) ? trianglesPerThread : num_triangles - triangles_called;
        threads[c] = std::thread(rasterizeTrianglesThread, c, trianglesToRender, triangles_called);
        triangles_called += trianglesToRender;
    }
    for (int c = 0; c < num_calls; c++) {
        threads[c].join();
    }
}

#define threads_rasterization 256

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

void rasterize_all_triangles(bool cpu_used) {
    float* d_depth_buffer;
    cudaGetSymbolAddress((void**)&d_depth_buffer, depth_buffer);

    if (cpu_used) {
        rasterizeAllTrianglesCPUandGPU();
    }
    else {
        rasterize_triangles_single_thread << <threads_rasterization, num_triangles / threads_rasterization + 1 >> > ();
    }
    cudaDeviceSynchronize();
    
    cudaMemset(d_depth_buffer, 0, sizeof(float) * scr_w * scr_h);
}

void add_triangle(vec3 p1, vec3 p2, vec3 p3, int idx) {
    triangle t = triangle(p1, p2, p3);
    cudaMemcpyToSymbol(triangles, &t, sizeof(triangle), sizeof(triangle) * idx);
}

vec3 computeNorm(vec3 p1, vec3 p2, vec3 p3) {
    return cross(p1 - p2, p1 - p3).normalize();
}

void loadRawModel(const char* filename, const char* filenameNorm, int start_idx) {
    vec3* vertexNorms = (vec3*)calloc(num_triangles * 3, sizeof(vec3));

    vec3 verts[num_triangles * 3];
    vec3 facetNorms[num_triangles];

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    int i = 0;
    while (file.good()) {
        vec3 vertices[3];

        file.read(reinterpret_cast<char*>(&vertices[0]), sizeof(vec3));
        file.read(reinterpret_cast<char*>(&vertices[1]), sizeof(vec3));
        file.read(reinterpret_cast<char*>(&vertices[2]), sizeof(vec3));

        for (int i2 = 0; i2 < 3; i2++) { float tmp = vertices[i2].z; vertices[i2].z = vertices[i2].y + 60.0f; vertices[i2].y = tmp; }

        verts[i * 3] = vertices[0];
        verts[i * 3 + 1] = vertices[1];
        verts[i * 3 + 2] = vertices[2];
        if (!file.eof()) {
            add_triangle(vertices[0], vertices[1], vertices[2], start_idx + i);
            i++;
        }
    }

    file.close();

    std::ifstream fileNorm(filenameNorm);

    if (!fileNorm) {
        std::cerr << "Unable to open normal vector file: " << filenameNorm << std::endl;
        return;
    }

    i = 0;
    std::string line;
    while (std::getline(fileNorm, line)) {
        std::stringstream ss(line);
        vec3 n;
        ss >> n.x >> n.y >> n.z;

        if (i < num_triangles) {
            facetNorms[i] = n.normalize();
            i++;
        }
        else {
            std::cerr << "More normals than expected. Stopping read." << std::endl;
            break;
        }
    }

    fileNorm.close();

    for (int n = 0; n < num_triangles*3; n++) {
        for (int v2 = 0; v2 < num_triangles * 3; v2++) {
            if (verts[v2] == verts[n]) {
                vertexNorms[v2] = vertexNorms[v2] + facetNorms[n/3];
            }
        }
    }

    for (int i = 0; i < num_triangles * 3; i++) {
        vertexNorms[i].normalize();
        //printf("%f %f %f\n", vertexNorms[i].x, vertexNorms[i].y, vertexNorms[i].z);
    }

    cudaMemcpyToSymbol(vertex_norms, vertexNorms, sizeof(vec3) * num_triangles * 3);
    free(vertexNorms);
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
    
    for (int i = 0; i < 1; i++) {
        loadRawModel("C:\\Users\\david\\Downloads\\pythonAndModels\\raw_model.raw", "C:\\Users\\david\\Downloads\\pythonAndModels\\norm_vecs.txt", i * 207);
    }

    cudaMemcpyFromSymbol(trianglesCPU, triangles, sizeof(triangle) * num_triangles);

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
        // **
        // dodaj boje tu u pixels
        for (int i = 0; i < 1; i++) {
            rasterize_all_triangles(false);
        }
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
        goto skp;
        if (frame % 10 == 1) {
            int fps = 10000 / (frametime);
            //printf("%d\n", fps);
            if (fps < 10) {
                printf("\rFPS: 000%d", fps);
            }
            else if (fps < 100) {
                printf("\rFPS: 00%d", fps);
            }
            else if (fps < 1000) {
                printf("\rFPS: 0%d", fps);
            }

            else {
                printf("\rFPS: %d", fps);
            }
            frametime = 0;
        }
        skp:
        
        cudaEventRecord(end);
        frame++;
        float milis;
        cudaEventElapsedTime(&milis, start, end);
        frametime += milis;
        printf("%f\n", milis);
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