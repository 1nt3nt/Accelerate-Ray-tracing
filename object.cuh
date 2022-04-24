#if !defined(_OBJECT_H_)

#define _OBJECT_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "material.cuh"
#include "hit.cuh"

enum ObjectType { NO_OBJECT, SPHERE, TRIANGLE };

class Object {
public:
    Object(const Material& newColor);
    virtual __device__ bool intersects(const glm::vec3& start,
        const glm::vec3& direction,
        Hit& hit) = 0;
    Material __device__ get_material() { return color; };

    //#protected:
    Material color;

};

#endif
