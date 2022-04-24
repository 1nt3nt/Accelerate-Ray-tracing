#include "triangle.cuh"

#include <math.h>
#include <fstream>
#include <iostream>
#include <vec3.hpp>
#include <gtx/string_cast.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

extern bool debugOn;
#define EPSILON 0.00001

Triangle::Triangle(const glm::vec3& v1,
    const glm::vec3& v2,
    const glm::vec3& v3,
    const Material& color) : Object(color) {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
}

__device__ bool Triangle::intersects(const glm::vec3& start,
    const glm::vec3& direction, Hit& hit) {
    //get t, u v
    float t, u, v;
    float a, b, c, d, e, f, g, h, i, k, l, m;
    glm::vec3 p;

    a = direction.operator[](0);
    b = v1.operator[](0) - v2.operator[](0);
    c = v1.operator[](0) - v3.operator[](0);
    d = direction.operator[](1);
    e = v1.operator[](1) - v2.operator[](1);
    f = v1.operator[](1) - v3.operator[](1);
    g = direction.operator[](2);
    h = v1.operator[](2) - v2.operator[](2);
    i = v1.operator[](2) - v3.operator[](2);
    k = v1.operator[](0) - start.operator[](0);
    l = v1.operator[](1) - start.operator[](1);
    m = v1.operator[](2) - start.operator[](2);

    // t
    glm::mat3 matrix1(k, l, m,
        b, e, h,
        c, f, i);

    glm::mat3 matrix2(a, d, g,
        b, e, h,
        c, f, i);
    t = glm::determinant(matrix1) / glm::determinant(matrix2);
    //cout << "t: " << t << endl;

    // u
    matrix1 = glm::mat3(a, d, g,
        k, l, m,
        c, f, i);
    u = glm::determinant(matrix1) / glm::determinant(matrix2);
    //cout << "u: " << u << endl;

    // v
    matrix1 = glm::mat3(a, d, g,
        b, e, h,
        k, l, m);
    v = glm::determinant(matrix1) / glm::determinant(matrix2);
    //cout << "v: "<< v << endl;

    if (t >= 0 && u <= 1 && u >= 0 && v <= 1 && v >= 0 && u + v >= 0 && u + v <= 1)
    {
        p = v1 + u * (v2 - v1) + v * (v3 - v1);
        glm::vec3 normal = glm::normalize(glm::cross((v2 - v1), (v3 - v1)));
        hit.set(p, normal, this, t);
        return true;
    }

    return false;
}
