#if !defined(_RAY_H_)
#define _RAY_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vec3.hpp>

class Ray {
public:
    __host__ __device__ Ray();
    __host__ __device__ Ray(const glm::vec3& start,
        const glm::vec3& direction,
        const glm::vec3& color);
    __device__ glm::vec3 get_start() const { return st; };
    __device__ glm::vec3 get_direction() const { return dir; };
    __host__ __device__ glm::vec3 get_color() const { return c; };
    __device__ void set(const glm::vec3& start, const glm::vec3& direction,
        const glm::vec3& color);
    __device__ void setColor(const glm::vec3& color);

private:
    glm::vec3 st, dir, c;
};

#endif
