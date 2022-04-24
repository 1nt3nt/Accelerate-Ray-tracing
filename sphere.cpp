#include "sphere.cuh"
#include <math.h>
#include <vec3.hpp>
#include <gtx/string_cast.hpp>

#include <fstream>
#include <iostream>
using namespace std;

extern bool debugOn;
#define EPSILON 0.00001

Sphere::Sphere(const glm::vec3& center, float radius,
    const Material& color) : Object(color) {
    this->center = center;
    this->radius = radius;
}

__device__ bool Sphere::intersects(const glm::vec3& start,
    const glm::vec3& direction, Hit& hit) {
    double _a, _b, _c, _d;
    float t_1, t_2;
    glm::vec3 p;
    _a = glm::dot(direction, direction);
    _b = glm::dot(direction * 2.0f, (start - this->center));
    _c = glm::dot((start - this->center), (start - this->center)) - glm::pow(this->radius, 2);
    _d = glm::pow(_b, 2) - 4 * _a * _c;

    if (_d > 0)
    {
        t_1 = (-_b + glm::sqrt(_d)) / (2 * _a);
        t_2 = (-_b - glm::sqrt(_d)) / (2 * _a);

        if (t_1 > 0 && t_2 > 0)
        {
            if (t_1 < t_2)
            {
                p = start + t_1 * direction;
                glm::vec3 normal = glm::normalize(p - this->center);
                hit.set(p, normal, this, t_1);
            }
            else
            {
                p = start + t_2 * direction;
                glm::vec3 normal = glm::normalize(p - this->center);
                hit.set(p, normal, this, t_2);
            }
        }

        if (t_1 > 0 && t_2 < 0)
        {
            p = start + t_1 * direction;
            glm::vec3 normal = glm::normalize(p - this->center);
            hit.set(p, normal, this, t_1);
        }

        if (t_2 > 0 && t_1 < 0)
        {
            p = start + t_2 * direction;
            glm::vec3 normal = glm::normalize(p - this->center);
            hit.set(p, normal, this, t_2);
        }
        return true;
    }

    return false;
}
