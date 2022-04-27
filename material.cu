#include "material.cuh"

Material::Material() {
    ka = glm::vec3(.5f, .5f, .5f);
    kd = glm::vec3(.5f, .5f, .5f);
    ks = glm::vec3(.5f, .5f, .5f);
    n = 10;
}

Material::Material(const glm::vec3& ambient,
    const glm::vec3& diffuse,
    const glm::vec3& specular,
    int shininess) {
    ka = ambient;
    kd = diffuse;
    ks = specular;
    n = shininess;
}

void Material::set(const glm::vec3& ambient,
    const glm::vec3& diffuse,
    const glm::vec3& specular,
    int shininess) {
    ka = ambient;
    kd = diffuse;
    ks = specular;
    n = shininess;
}


