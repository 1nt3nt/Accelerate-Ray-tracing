#include "hit.cuh"

Hit::Hit() {
    p = glm::vec3(0, 0, 0);
    N = glm::vec3(1, 0, 0);
    obj = 0;
    dist = 0;
}

Hit::Hit(const glm::vec3& hitpoint,
    const glm::vec3& normal, Object* obj, double d) {
    p = hitpoint;
    N = normal;
    this->obj = obj;
    dist = d;
}

__device__ void Hit::set(const glm::vec3& hitpoint,
    const glm::vec3& normal, Object* obj, double d) {
    p = hitpoint;
    N = normal;
    this->obj = obj;
    dist = d;
}

