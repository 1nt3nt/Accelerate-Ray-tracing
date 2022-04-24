#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vec3.hpp>

#include "object.cuh"
#include "hit.cuh"

class Triangle : public virtual Object {
public:
    Triangle(const glm::vec3& v1,
        const glm::vec3& v3,
        const glm::vec3& v4,
        const Material& color);
    void setNormal();
    __device__ bool intersects(const glm::vec3& start,
        const glm::vec3& direction,
        Hit& hit);

private:
    glm::vec3 v1, v2, v3;
    glm::vec3 N;
};

#endif
