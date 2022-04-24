#if !defined(_HIT_H_)
#define _HIT_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vec3.hpp>

class Object;

class Hit {
public:
    __host__ __device__ Hit();
    __host__ __device__ Hit(const glm::vec3& hitpoint,
        const glm::vec3& normal, Object* obj, double d);
    __device__ void set(const glm::vec3& hitpoint,
        const glm::vec3& normal, Object* obj, double d);
    __device__ glm::vec3 hitPoint() { return p; };
    __device__ glm::vec3 normal() { return N; };
    __device__ Object* getObject() { return obj; };
    __device__ double getDistance() { return dist; };

    // private:
    glm::vec3 p;
    glm::vec3 N;
    Object* obj;
    double dist;
};

#endif
