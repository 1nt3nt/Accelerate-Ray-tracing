#include "light.hpp"

Light::Light() {
    p = glm::vec3(0, 0, 0);
}

Light::Light(const glm::vec3& color, const glm::vec3& position) {
    c = color;
    p = position;
}

void Light::set(const glm::vec3& color, const glm::vec3& position) {
    c = color;
    p = position;
}
