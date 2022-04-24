
#include <vec3.hpp>
#include <mat4x4.hpp>
#include <gtx/string_cast.hpp>

#include "ray.hpp"
#include "light.hpp"
#include "hit.cuh"
#include "material.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define TILE_WIDTH 32

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    int* elements;
} Matrix;

__constant__ float c_lights[12]; // lights source in constant memory

//index 0 ~ 2 are ambient light. 3 ~ 7 are clipping frustum
//8~10 eye position
__constant__ float c_ambient[11];

__global__ void ray_color(Matrix dcs_x, Matrix dcs_y, int width, Ray* rays,
    glm::mat4& Mvcswcs);

__global__ void first_hit(Ray* rays, int n, Hit* d_hits, Object* d_obs);

__device__ glm::vec3 mirror_direction(const glm::vec3& L, const glm::vec3& N);

__device__ glm::vec3 local_illumination(const glm::vec3& V, const glm::vec3& N,
    const glm::vec3& L,
    const Material& mat, const glm::vec3& ls);
__device__ bool in_shadow(Hit* shadowHits, Hit& hit, const glm::vec3& direction,
    Object* obj, int n, Hit& test);

__device__ glm::vec3 shadow_illumination(const glm::vec3& V, const glm::vec3& N,
    const glm::vec3& L,
    const Material& mat, const glm::vec3& Clight);

__device__ void set_ray(int xDCS, int yDCS, glm::vec3& start, glm::vec3& direction,
    glm::mat4& Mvcswcs, int width);



//----------------------- create buffer and mapping buffer to cuda -------------------


    // Allocate the GL Buffer, bufferid: render buffer
    // GLubyte bufferID, texture;
    // // generate buffer id
    // glGenBuffers(1, &bufferID);
    // glGenTextures(1, &texture); 

    // // make this the current unpark buffer
    // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
    // glBindTexture(GL_TEXTURE_2D, texture);

    // //set basic parameters
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // // Create texture data
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA);

    // // Unbind texture
    // glBindTexture(GL_TEXTURE_2D, 0);

    // // allocate data for the buffer, 4 chanel, 8 bit image
    // glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLbyte)*w*h*4, NULL, GL_DYNAMIC_COPY);
    // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // GLubyte depthBuffer;
    // glGenRenderbuffers(1, &depthBuffer);
    // glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h,);
    // glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // cudaError_t status;
    // status = cudaGLRegisterBufferObject(bufferID);
    // if(!status)
    //     fprintf("Register buffer object failed");
    //cudaGLMapBufferObject()

    //------------------------------- DONE ---------------------------