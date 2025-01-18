
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

cudaError_t ercall;
#define CCALL(call) ercall = call; if(cudaSuccess != ercall){fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(ercall)); exit(EXIT_FAILURE);}

//#define DEBUG //if defined, extra stuff is printed(shouldn't slow down program noticeably)

// init of structs and methods as well as global vars and respective functions and macros
//*****************************************************************************************************************************************************************************************

// sizes for renderer
#define num_triangles 36
#define scr_w 512
#define scr_h 512
#define max_triangles_per_node 10 // max triangles in each BVH node before divisions stop
#define node_density 1.0f // nodes per x, y, z of overall bounding box

// macros to replace functions
#define dot(vec3_v1, vec3_v2) (vec3_v1.x * vec3_v2.x + vec3_v1.y * vec3_v2.y + vec3_v1.z * vec3_v2.z)
#define matrix2D_eval(float_a , float_b, float_c, float_d) (float_a*float_d - float_b*float_c)
#define magnitude(vec3_a) (sqrtf(dot(vec3_a, vec3_a)))

// too lazy to set up cudas rng so i use this bad one
inline __host__ __device__ long int xorRand(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

// Define the vec3 struct
typedef struct {
    float x, y, z;
    bool null;
}vec3_tmp; // to avoid dynamic init(placeholder)

struct vec3 {
    float x, y, z;
    bool null;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z), null(false) {}

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
        const float scl = magnitude((*this));
        return vec3(x / scl, y / scl, z / scl);
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

// for BVH traversal/init
__device__ vec3_tmp min_point_BVH;
__device__ vec3_tmp max_point_BVH;
vec3_tmp BVHMin, BVHMax; // global CPU min/max

// structs
struct color {
    float r, g, b;
    __host__ __device__ color(float R, float G, float B) : r(R), g(G), b(B) {}

    inline __host__ __device__ color operator+(const color& f) const {
        return color(r + f.r, g + f.g, b + f.b);
    }

    inline __host__ __device__ color operator*(const float f) const {
        return color(r * f, g * f, b * f);
    }
};

struct material {
    color c;
    float brightness, roughness;

    __host__ __device__ material() : c(color(0.0f, 0.0f, 0.0f)) {}

    __host__ __device__ material(color C, float B, float rough) : c(C), brightness(B), roughness(rough) {}
};

struct ray {
    vec3 origin, direction;
    __host__ __device__ ray() : origin(vec3(0.0f, 0.0f, 0.0f)), direction(vec3(0.0f, 0.0f, 0.0f)) {}
    __host__ __device__ ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}
};

struct triangle {
    vec3 p1, p2, p3;
    vec3 nv;
    vec3 sb21, sb31;
    float dot2121, dot2131, dot3131;

    __device__ triangle() : p1(vec3(0.0f, 0.0f, 0.0f)), p2(vec3(0.0f, 0.0f, 0.0f)), p3(vec3(0.0f, 0.0f, 0.0f)) {}

    __host__ __device__ triangle(vec3 P1, vec3 P2, vec3 P3) {
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
};

inline __host__ __device__ float get_min(const float v1, const float v2, const float v3) {
    const float minab = (v1 + v2 - fabs(v1 - v2)) / 2;
    const float minfinal = (minab + v3 - fabs(minab - v3)) / 2;
    return minfinal;
}

inline __host__ __device__ float get_max(const float v1, const float v2, const float v3) {
    const float maxab = (v1 + v2 + fabs(v1 - v2)) / 2;
    const float maxfinal = (maxab + v3 + fabs(maxab - v3)) / 2;
    return maxfinal;
}

inline __host__ __device__ float get_min2(const float v1, const float v2) {
    return (v1 + v2 - fabs(v1 - v2)) / 2;
}

inline __host__ __device__ float get_max2(const float v1, const float v2) {
    return (v1 + v2 + fabs(v1 - v2)) / 2;
}

struct axis_aligning_bounding_box {
    vec3 min, max;

    __device__ axis_aligning_bounding_box(triangle t) {
        min = vec3(get_min(t.p1.x, t.p2.x, t.p3.x), get_min(t.p1.y, t.p2.y, t.p3.y), get_min(t.p1.z, t.p2.z, t.p3.z));
        max = vec3(get_max(t.p1.x, t.p2.x, t.p3.x), get_max(t.p1.y, t.p2.y, t.p3.y), get_max(t.p1.z, t.p2.z, t.p3.z));

        // get bounding box of all bounding boxes :)
        // fast bc its in cache
        min_point_BVH.x = get_min2(min_point_BVH.x, min.x);
        min_point_BVH.y = get_min2(min_point_BVH.y, min.y);
        min_point_BVH.z = get_min2(min_point_BVH.z, min.z);

        max_point_BVH.x = get_max2(max_point_BVH.x, max.x);
        max_point_BVH.y = get_max2(max_point_BVH.y, max.y);
        max_point_BVH.z = get_max2(max_point_BVH.z, max.z);
    }

    __host__ axis_aligning_bounding_box(triangle t, int placeholder) {
        min = vec3(get_min(t.p1.x, t.p2.x, t.p3.x), get_min(t.p1.y, t.p2.y, t.p3.y), get_min(t.p1.z, t.p2.z, t.p3.z));
        max = vec3(get_max(t.p1.x, t.p2.x, t.p3.x), get_max(t.p1.y, t.p2.y, t.p3.y), get_max(t.p1.z, t.p2.z, t.p3.z));

        // get bounding box of all bounding boxes :)
        BVHMin.x = get_min2(BVHMin.x, min.x);
        BVHMin.y = get_min2(BVHMin.y, min.y);
        BVHMin.z = get_min2(BVHMin.z, min.z);

        BVHMax.x = get_max2(BVHMax.x, max.x);
        BVHMax.y = get_max2(BVHMax.y, max.y);
        BVHMax.z = get_max2(BVHMax.z, max.z);
    }
};

// using grid method for BVH to make dynamic scenes fast
struct grid_node {
    vec3 bottomLeftForwardsPoint;
    float length;
    int cur_index = 0;
    int triangle_indices[max_triangles_per_node];

    __device__ grid_node(vec3 c, float l) : bottomLeftForwardsPoint(c), length(l) {}

    __device__ void add_triangle(const int t_id) {
        triangle_indices[cur_index] = t_id;
    }
};

__device__ char* grid_nodes;

inline __device__ grid_node init_grid_node(int x, int y, int z) {
    return grid_node(vec3(x * node_density, y * node_density, z * node_density), node_density);
}
__global__ void nodeInitKernel(const int numNodesX, const int numNodesY, const int numNodesZ) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int zLayer = id / (numNodesX * numNodesY);
    const int tmp = id % (numNodesX * numNodesY);
    const int xLayer = tmp % numNodesX;
    const int yLayer = tmp / numNodesX;

    ((grid_node*)grid_nodes)[id] = init_grid_node(xLayer, yLayer, zLayer);
}

#define threads_grid_alloc 256

void allocGridNodes() {
    // assumes min and max points of BVH are already defined. Can only be run after triangle init
    // 
    // cuda malloc funcs onyl accessible in host funcs.. 
    cudaMemcpyFromSymbol(&BVHMin, &min_point_BVH, sizeof(vec3_tmp));
    cudaMemcpyFromSymbol(&BVHMax, &max_point_BVH, sizeof(vec3_tmp));

    const int numNodesX = (BVHMax.x - BVHMin.x) / node_density;
    const int numNodesY = (BVHMax.y - BVHMin.y) / node_density;
    const int numNodesZ = (BVHMax.z - BVHMin.z) / node_density;

    char* tmpPtr;
    cudaMalloc(&tmpPtr, sizeof(grid_node) * numNodesX * numNodesY * numNodesZ);
    cudaMemcpyToSymbol(grid_nodes, &tmpPtr, sizeof(char*));
    
    const int numBlocks = numNodesX * numNodesY * numNodesZ / threads_grid_alloc;

    nodeInitKernel << <threads_grid_alloc, numBlocks >> > (numNodesX, numNodesY, numNodesZ);

    grid_node* gridNodesCPU = (grid_node*)malloc(numNodesX * numNodesY * numNodesZ * sizeof(grid_node));

    // because dynamic init isn't supported on GPU, its done CPU
}

// global device arrs
// init as chars to bypass restrictions on dynamic initialization
__device__ char triangles[num_triangles * sizeof(triangle)]; // all triangles(on global mem, so slow access)
char trianglesCPU[num_triangles * sizeof(triangle)];
__device__ char triangle_materials[num_triangles * sizeof(material)]; // materials corresponding to triangles
__device__ char aabb_arr[num_triangles * sizeof(axis_aligning_bounding_box)];
char aabb_arrCPU[num_triangles * sizeof(axis_aligning_bounding_box)];
__device__ char screen_buffer[scr_w * scr_h * sizeof(color)];

// cached triangles arr to prevent more lookups to global
__constant__ char triangles_cached[10];

typedef struct {
    vec3 intersect;
    float dist;
}intersect_pkg;

inline __device__ intersect_pkg get_triangle_intersect(const ray r, const triangle t) {
    const float disc = dot(r.direction, t.nv);
    const float dt = fabs(disc);
    intersect_pkg ret;

    if (dt <= 1e-10) { // check if the plane and ray are paralell enough to be ignored
        ret.intersect.null = true;
        return ret;
    }

    vec3 temp_sub =t.p1 - r.origin;
    temp_sub = r.direction * __fdividef(dot(t.nv, temp_sub), disc);// fast division since fastmath doesnt work on my system for some reason
    ret.intersect = r.origin + temp_sub;
    const vec3 v2 = ret.intersect - t.p1;
    const float dot02 = dot(t.sb21, v2);
    const float dot12 = dot(t.sb31, v2);
    const float disc2 = (t.dot2121 * t.dot3131 - t.dot2131 * t.dot2131);
    if (disc2 == 0.0f) { ret.intersect.null = true; return ret; }
    const float u = (t.dot3131 * dot02 - t.dot2131 * dot12) / disc2;
    const float v = (t.dot2121 * dot12 - t.dot2131 * dot02) / disc2;
    if ((((u < 0) || (v < 0) || (u + v > 1) || dot(temp_sub, r.direction) < 0.0f))) { ret.intersect.null = true; return ret; }
    ret.dist = magnitude(temp_sub);
    return ret;
}

inline __device__ vec3 get_grid_node_intersect(const ray r, const grid_node n) {
    float closest_dist = 1e20f;
    vec3 intersect; intersect.null = true;
    for (int i = 0; i < n.cur_index; i++) {
        intersect_pkg it = get_triangle_intersect(r, ((triangle*)triangles)[n.triangle_indices[i]]);
        if (it.dist < closest_dist) {
            closest_dist = it.dist;
            intersect = it.intersect;
        }
    }
    return intersect;
}

inline __device__ float get_specular_intensity(const ray r, const vec3 light_point, const float light_strength, const int spec_pow) {
    const vec3 light_dir = light_point - r.origin;
    return -1 * pow(dot(light_dir, r.direction), spec_pow) * light_strength;
}

inline __device__ float get_diffuse_intensity(const ray intersect, vec3 normal_vec, const vec3 light_point, const float light_intensity) {
    // after intersect, calc diffuse intensity

    if (dot(intersect.direction, normal_vec) >= 0.0f) {
        normal_vec = normal_vec * -1;
    }

    const vec3 vec_to_light = light_point - intersect.origin;
    const float dt = dot(vec_to_light, normal_vec);
    return (dt <= 0.0f) * 0.0f + (dt > 0.0f) * dt * light_intensity; // avoid branching
}

inline __device__ float get_total_light_intensity(const ray intersect, const vec3 nv, const vec3 light_point, const float light_intensity, const float ambient_intensity) {
    return get_diffuse_intensity(intersect, nv, light_point, light_intensity) + get_specular_intensity(intersect, light_point, light_intensity, 10) + ambient_intensity;
}

__global__ void init_triangle(vec3 p1, vec3 p2, vec3 p3, material m, int index) {
    const triangle t = triangle(p1, p2, p3);
    ((triangle*)triangles)[index] = t;
    ((material*)triangle_materials)[index] = m;
    ((axis_aligning_bounding_box*)aabb_arr)[index] = axis_aligning_bounding_box(t);
}

void initTriangle(vec3 p1, vec3 p2, vec3 p3, material m, int index) {
    init_triangle << <1, 1 >> > (p1, p2, p3, m, index);
    const triangle t = triangle(p1, p2, p3);
    ((triangle*)trianglesCPU)[index] = t;
    ((axis_aligning_bounding_box*)aabb_arrCPU)[index] = axis_aligning_bounding_box(t,0); // int to differentiate host from device call(compiler is stupid)
}

char cpuColors[scr_w * scr_h * sizeof(color)];

//*****************************************************************************************************************************************************************************************
// opengl stuff
// draws 2 triangles at z=0 and textures them with the pixel colors outputted by the cuda program
// no interop, data transfers from GPU to CPU and back to GPU each frame
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

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Cuda-openGL Interop", NULL, NULL);

    glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);  // Remove window decorations
    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);  // Make the window non-resizable

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scr_w, scr_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cpuColors);
	glGenerateMipmap(GL_TEXTURE_2D);
    glUseProgram(shaderProgram); 
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

    clock_t start, end;
    int frametime = 0;
    unsigned int frame = 0;
    while (!glfwWindowShouldClose(window))
    {
        start = clock();
        processInput(window);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        int ind = 0;
        // **
        // dodaj boje tu u pixels
        // **
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scr_w, scr_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cpuColors);
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
        end = clock();
        if (frame % 10 == 1) {
            int fps = 10000 / (frametime);
            fps = fps < 100 ? fps : 99;
            printf("\rFPS: %d", fps);
            frametime = 0;
        }
        frame++;
        frametime += end - start;
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