#include "ray.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Ray::Ray(const glm::vec3& start,
    const glm::vec3& direction,
    const glm::vec3& color) {
    st = start;
    dir = direction;
    c = color;
}

__device__ void Ray::set(const glm::vec3& start,
    const glm::vec3& direction,
    const glm::vec3& color)
{
    st = start;
    dir = direction;
    c = color;
}

__device__ void Ray::setColor(const glm::vec3& color)
{
    c = color;
}
