#include <math.h>
#include <float.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "camera.hpp"
#include "kbui.hpp"

#include <vec3.hpp>
#include <mat4x4.hpp>
#include <gtx/string_cast.hpp>

#include "object.cuh"
#include "ray.hpp"
#include "light.hpp"
#include "hit.cuh"
#include "triangle.cuh"
#include "sphere.cuh"
#include "tokenizer.hpp"
#include "material.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "ray_kernel.cuh"
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

using namespace std;

/////////////////////////////////////////////////////
// DECLARATIONS
////////////////////////////////////////////////////

/**
 * @brief Get color of a ray passing through (x,y)DCS.
 *
 * @param dcs combine xDCS and yDCS
 * @param width windows width
 * @param rays store generated rays
 * @param color output color
 * @param obs #object on scene
 */
__global__ void ray_color(Matrix dcs_x, Matrix dcs_y, int width, Ray* rays,
    glm::mat4& Mvcswcs) {

    glm::vec3 start;
    glm::vec3 direction;
    glm::vec3 color(0, 0, 0); // temp color variable
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int index = row * width + col;
    // shared memory
    __shared__ int s_dcs_x[32][32];
    __shared__ int s_dcs_y[32][32];

    // loading data into shared memory
    if (col < width && row < width)
    {
        s_dcs_x[ty][tx] = dcs_x.elements[index];
        s_dcs_y[ty][tx] = dcs_y.elements[index];
    }

    __syncthreads();

    set_ray(s_dcs_x[ty][tx], s_dcs_y[ty][tx], start, direction, Mvcswcs, width);
    Ray* r = new Ray(start, direction, color);
    rays[index] = *r;
}

/////////////////////////////////////////////////////////
// Initialize a ray starting at (x y)DCS.
// Parameters:
// xDCS, yDCS: the pixel's coordinates
// start: the ray's starting point
// direction: the ray's direction vector
/////////////////////////////////////////////////////////
__device__ void set_ray(int xDCS, int yDCS, glm::vec3& start, glm::vec3& direction,
    glm::mat4& Mvcswcs, int width) {

    float xVCS, yVCS, zVCS;
    float dX, dY;
    dX = (c_ambient[4] - c_ambient[3]) / width;
    dY = (c_ambient[6] - c_ambient[5]) / width;
    float x = (xDCS + 0.5) * dX;
    xVCS = c_ambient[3] + (xDCS + 0.5) * dX;
    yVCS = c_ambient[5] + (yDCS + 0.5) * dY;
    zVCS = -c_ambient[7];
    glm::vec4 pVCS(xVCS, yVCS, zVCS, 1);
    glm::vec4 pWCS = Mvcswcs * pVCS;
    start = pWCS;
    glm::vec3 eye(c_ambient[8], c_ambient[9], c_ambient[10]);
    direction = glm::normalize(start - eye);
}

/**
 * @brief
 *
 * @param rays ray arr
 * @param n #scene objects
 * @param d_hits device hit arr
 * @param d_obs device scene objects
 * @param p #pixel
 */
__global__ void first_hit(Ray* rays, int n, Hit* d_hits, Object* d_obs) {
    int len = 0;
    Hit tempHit; // temp hit variable
    int width = 500;
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int index = row * width + col;

    glm::vec3 start = rays[index].get_start();
    glm::vec3 direction = rays[index].get_direction();

    for (int i = 0; i < n; i++)
    {
        if (d_obs[i].intersects(start, direction, tempHit))
        {
            d_hits[i] = tempHit;
            len++;
        }
    }

    glm::vec3 color(0, 0, 0);
    if (len != 0)
    {
        tempHit = d_hits[0];
        //cout << hitList.size() << endl;
        for (int i = 1; i < len; i++)
        {
            if (tempHit.getDistance() > d_hits[i].getDistance())
            {
                tempHit = d_hits[i];
            }
        }

        // ------------------------ calculate color --------------------
        Object* ob = tempHit.getObject();
        Hit test; // check if this hit is in the shadow
        for (int i = 0; i < 2; i++)
        {
            glm::vec3 light_color(c_lights[i * 6], c_lights[i * 6 + 1], c_lights[i * 6 + 2]);
            glm::vec3 light_pos(c_lights[i * 6 + 3], c_lights[i * 6 + 4], c_lights[i * 6 + 5]);

            glm::vec3 lightDir = glm::normalize(light_pos - tempHit.hitPoint());
            if (glm::dot(tempHit.normal(), lightDir) > 0)
            {
                if (in_shadow(d_hits, tempHit, lightDir, d_obs, n, test))
                {
                    color += shadow_illumination(-direction, tempHit.normal(),
                        lightDir, test.getObject()->get_material(), light_color);
                }
                else
                {
                    color += local_illumination(-direction, tempHit.normal(),
                        lightDir, ob->get_material(), light_color);
                }
            }
        }
        glm::vec3 ambient_light(c_ambient[0], c_ambient[1], c_ambient[2]);
        color += ob->get_material().get_ambient() * ambient_light;
        rays[index].setColor(color);
        // -------------------- end -------------------
    }
    else
    {
        color = glm::vec3(0.3, 0.4, 0.4);
        rays[index].setColor(color);
    }
}

__device__ glm::vec3 mirror_direction(const glm::vec3& L, const glm::vec3& N) {
    float NL = glm::dot(N, L);
    return (2.0f * NL) * N - L;
}

///////////////////////////////////////////////////////////////////////
// Compute the Phong local illumination color.
//
// Parameters:
// V: the direction towards the eye
// N: the "up" direction of the surface
// L: the direction towards the light
// mat: the surface material
// Clight: the light's color
//
// YOU MUST IMPLEMENT THIS FUNCTION
///////////////////////////////////////////////////////////////////////
__device__ glm::vec3 local_illumination(const glm::vec3& V, const glm::vec3& N,
    const glm::vec3& L,
    const Material& mat, const glm::vec3& Clight) {


    glm::vec3 R = mirror_direction(L, N);
    //glm::vec3 Ca = ambient_light;
    //glm::vec3 ka = mat.get_ambient(); the ambient reflectane of the surface
    glm::vec3 kd = mat.get_diffuse(); //the diffuse reflectance
    glm::vec3 ks = mat.get_specular(); //the specular reflectance
    float n = mat.get_shininess();
    glm::vec3 Id;
    glm::vec3 Is;
    glm::vec3 C;
    glm::vec3 ambient_light(c_ambient[0], c_ambient[1], c_ambient[2]);

    Id = Clight * kd * glm::dot(N, L);
    if (glm::dot(V, N) < 0)
    {
        return ambient_light;
    }

    if (glm::dot(R, V) < 0)
    {
        return Id;
    }
    else
    {
        Is = Clight * ks * glm::pow(glm::dot(R, V), n);
    }

    C = Id + Is;

    return C;
}

__device__ glm::vec3 shadow_illumination(const glm::vec3& V, const glm::vec3& N,
    const glm::vec3& L,
    const Material& mat, const glm::vec3& Clight) {


    glm::vec3 R = mirror_direction(L, N);
    //glm::vec3 Ca = ambient_light;
    //glm::vec3 ka = mat.get_ambient(); the ambient reflectane of the surface
    glm::vec3 kd = mat.get_diffuse(); //the diffuse reflectance
    glm::vec3 ks = mat.get_specular(); //the specular reflectance
    float n = mat.get_shininess();
    glm::vec3 Id;
    glm::vec3 Is;
    glm::vec3 C;
    glm::vec3 ambient_light(c_ambient[0], c_ambient[1], c_ambient[2]);

    Id = Clight * kd * glm::dot(N, L);
    if (glm::dot(V, N) < 0)
    {
        return ambient_light;
    }

    if (glm::dot(R, V) < 0)
    {
        return Id;
    }
    else
    {
        Is = Clight * ks * glm::pow(glm::dot(R, V), n);
    }

    C = Id + Is;

    return C;
}


__device__ bool in_shadow(Hit* shadowHits, Hit& hit, const glm::vec3& direction,
    Object* obj, int n, Hit& test) {
    int len = 0;// shadowHits size
    for (int i = 0; i < n; i++)
    {
        if (obj[i].intersects(hit.hitPoint(), direction, test))
        {
            shadowHits[i] = test;
            len++;
        }
    }

    if (len != 0)
    {
        for (int i = 0; i < len; i++)
        {
            if (test.getDistance() < shadowHits[i].getDistance())
            {
                test = shadowHits[i];
            }
        }
        return true;
    }
    return false;
}