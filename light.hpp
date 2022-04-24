
#if !defined(_LIGHT_H_)
#define _LIGHT_H_

#include <vec3.hpp>

class Light {
public:
    Light();
    Light(const glm::vec3& color, const glm::vec3& position);
    void set(const glm::vec3& color, const glm::vec3& position);
    glm::vec3 get_color() { return c; };
    glm::vec3 get_pos() { return p; };

private:
    glm::vec3 p;
    glm::vec3 c;
};

#endif
