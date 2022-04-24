#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vec3.hpp>

#include "object.cuh"
#include "material.cuh"
#include "hit.cuh"

class Sphere : public virtual Object {
public:
    Sphere(const glm::vec3& center, float radius, const Material& color);
    __device__ bool intersects(const glm::vec3& start,
        const glm::vec3& direction,
        Hit& hit);

private:
    glm::vec3 center;
    float radius;
};

#endif
