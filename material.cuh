#if !defined(_MATERIAL_H_)
#define _MATERIAL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vec3.hpp>

class Material {
public:
    Material();
    Material(const glm::vec3& ambient,
        const glm::vec3& diffuse,
        const glm::vec3& specular,
        int shininess);
    void set(const glm::vec3& ambient,
        const glm::vec3& diffuse,
        const glm::vec3& specular,
        int shininess);
    __device__ glm::vec3 get_ambient() const { return ka; };
    __device__ glm::vec3 get_diffuse() const { return kd; };
    __device__ glm::vec3 get_specular() const { return ks; };
    __device__ int get_shininess() const { return n; };

private:
    glm::vec3 ka, kd, ks;
    int n;
};

#endif
