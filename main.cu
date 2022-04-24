
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "camera.hpp"
#include "kbui.hpp"
#include "object.cuh"
#include "ray.hpp"
#include "light.hpp"
#include "hit.cuh"
#include "triangle.cuh"
#include "sphere.cuh"
#include "tokenizer.hpp"
#include "material.cuh"
#include "ray_kernel.cuh"

#include <vec3.hpp>
#include <mat4x4.hpp>
#include <gtx/string_cast.hpp>

using namespace std;

KBUI the_ui;
Camera cam;

/////////////////////////////////////////////////////
// DECLARATIONS
////////////////////////////////////////////////////

// Forward declarations for functions in this file
void init_UI();
void setup_camera();
void check_for_resize();
void read_scene(const char* sceneFile);

void render();
void camera_changed();
void cam_param_changed(float);
bool get_was_window_resized();
void reset_camera(float);
void init_scene();
Matrix AllocateMatrix(int height, int width, int init);
Matrix AllocateDeviceMatrix(const Matrix M);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void mouse_button_callback(GLFWwindow* window, int button,
    int action, int mods);
static void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key,
    int scancode, int action, int mods);
void display();
int main(int argc, char* argv[]);

// USEFUL Flag:
// When the user clicks on a pixel, the mouse_button_callback does two things:
//   1. sets this flag.
//   2. calls ray_color() on that pixel
// This lets you check all your intersection code, for ONE ray of your choosing.
bool debugOn = false;

// FRAME BUFFER Declarations.
// The initial image is 500 x 500 x 3 bytes (3 bytes per pixel)
int winWidth = 500;
int winHeight = 500;
GLubyte* img = NULL;   // image is allocated by check_for_resize(), not here.

// These are the camera parameters.
// The camera position and orientation:
glm::vec3 eye;
glm::vec3 lookat;
glm::vec3 vup;

// The camera's HOME parameters, used in reset_camera()
glm::vec3 eye_home(1.4, 1.6, 4.6);
glm::vec3 lookat_home(0, 0, 0);
glm::vec3 vup_home(0, 1, 0);

// The clipping frustum
float clipL = -1;
float clipR = +1;
float clipB = -1;
float clipT = +1;
float clipN = 2;

// The camera's HOME frustum, also used in reset_camera()
float clip_home[5] = { -1, +1, -1, +1, 2 };

float ambient_fraction_home = 0.2; // how much of lights is ambient

vector<Object*> scene_objects; // list of objects in the scene
vector<Light> scene_lights; // list of lights in the scene
vector<Material> materials; // list of available materials
glm::vec3 ambient_light; // indirect light that shines when all lights blocked
float ambient_fraction = 0.2; // how much of lights is ambient

glm::mat4 Mvcswcs;  // the inverse of the view matrix.
vector<Hit> hits;     // list of hit records for current ray
vector<Hit> shadowHits; // list of hit records for shadow ray

// Used to trigger render() when camera has changed.
bool frame_buffer_stale = true;

// Rays which miss all objects have this color.
const glm::vec3 background_color(0.3, 0.4, 0.4); // dark blue

// Shadows on/off
float show_shadows = 0.0;


//////////////////////////////////////////////////////////////////////
// Compute Mvcstowcs.
//////////////////////////////////////////////////////////////////////
void setup_camera() {

    // Please leave this line in place.
    check_for_resize();
    Mvcswcs = glm::inverse(glm::lookAt(eye, lookat, vup));
}


//////////////////////////////////////////////////////
// This function sets up a simple scene.
/////////////////////////////////////////////////////
void read_scene(const char* filename) {
    float r, g, b;
    float x, y, z;
    int num_materials;
    int num_lights;
    int num_objects;

    Tokenizer toker(filename);

    while (!toker.eof()) {
        string keyword = toker.next_string();

        // cout << "keyword:" << keyword << "\n";

        if (keyword == string("")) {
            continue; // skip blank lines
        }
        else if (keyword == string("camera_eye"))
        {
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            eye = glm::vec3(x, y, z);
        }
        else if (keyword == string("camera_lookat"))
        {
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            lookat = glm::vec3(x, y, z);
        }
        else if (keyword == string("camera_vup")) {
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            vup = glm::vec3(x, y, z);
        }
        else if (keyword == string("camera_clip")) {
            clipL = toker.next_number();
            clipR = toker.next_number();
            clipB = toker.next_number();
            clipT = toker.next_number();
            clipN = toker.next_number();
        }
        else if (keyword == string("camera_ambient_fraction")) {
            ambient_fraction = toker.next_number();
        }
        else if (keyword == string("#materials")) {
            num_materials = toker.next_number();
            materials.reserve(num_materials);
        }
        else if (keyword == string("#lights")) {
            num_lights = toker.next_number();
            scene_lights.reserve(num_lights);
        }
        else if (keyword == string("#objects")) {
            num_objects = toker.next_number();
            scene_objects.reserve(num_objects);
        }
        else if (keyword == string("material"))
        {
            glm::vec3 ambient, diffuse, specular;
            int shininess;

            toker.match("ambient");
            r = toker.next_number();
            g = toker.next_number();
            b = toker.next_number();
            ambient = glm::vec3(r, g, b);

            toker.match("diffuse");
            r = toker.next_number();
            g = toker.next_number();
            b = toker.next_number();
            diffuse = glm::vec3(r, g, b);

            toker.match("specular");
            r = toker.next_number();
            g = toker.next_number();
            b = toker.next_number();
            specular = glm::vec3(r, g, b);

            toker.match("shininess");
            shininess = toker.next_number();

            Material* m = new Material(ambient, diffuse, specular, shininess);
            materials.push_back(*m);
        }
        else if (keyword == string("light"))
        {
            glm::vec3 color, position;

            toker.match("color");
            r = toker.next_number();
            g = toker.next_number();
            b = toker.next_number();
            color = glm::vec3(r, g, b);

            toker.match("position");
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            position = glm::vec3(x, y, z);

            Light* light = new Light(color, position);
            scene_lights.push_back(*light);
        }
        else if (keyword == string("sphere"))
        {
            glm::vec3 center;
            float radius;
            int materialID;

            toker.match("center");
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            center = glm::vec3(x, y, z);

            toker.match("radius");
            radius = toker.next_number();

            toker.match("material");
            materialID = toker.next_number();

            scene_objects.push_back(new Sphere(center, radius, materials[materialID]));
        }
        else if (keyword == string("triangle")) {
            glm::vec3 A, B, C;
            int materialID;

            toker.match("vertex");
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            A = glm::vec3(x, y, z);

            toker.match("vertex");
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            B = glm::vec3(x, y, z);

            toker.match("vertex");
            x = toker.next_number();
            y = toker.next_number();
            z = toker.next_number();
            C = glm::vec3(x, y, z);

            toker.match("material");
            materialID = toker.next_number();
            // .... more code needed

            scene_objects.push_back(new Triangle(A, B, C,
                materials[materialID]));
        }
        else {
            cerr << "Parse error: unrecognized keyword \""
                << keyword << "\"\n";
            exit(EXIT_FAILURE);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// If window size has changed, re-allocate the frame buffer
//////////////////////////////////////////////////////////////////////
void check_for_resize() {
    // Now, check if the frame buffer needs to be created,
    // or re-created.

    bool should_allocate = false;
    if (img == NULL) {
        // frame buffer not yet allocated.
        should_allocate = true;
    }
    else if (winWidth != cam.get_win_W() ||
        winHeight != cam.get_win_H()) {

        // frame buffer allocated, but has changed size.
        delete[] img;
        should_allocate = true;
        winWidth = cam.get_win_W();
        winHeight = cam.get_win_H();
    }

    if (should_allocate) {

        // cout << "ALLOCATING: (W H)=(" << winWidth
        //      << " " << winHeight << ")\n";

        img = new GLubyte[winWidth * winHeight * 3];
        camera_changed();
    }
}

// init = 0; x dcs
// init = 1; y dcs
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    M.elements = (int*)malloc(size * sizeof(int));

    if (init == 0)
    {
        for (unsigned int i = 0; i < M.height * M.width; i++)
        {
            M.elements[i] = i;
        }
    }
    if (init == 1)
    {
        for (unsigned int i = 0; i < M.height; i++)
        {
            for (unsigned int j = 0; j < M.width; j++)
            {
                M.elements[i * M.width + j] = i;
            }
        }
    }
    return M;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
        cudaMemcpyHostToDevice);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

/////////////////////////////////////////////////////////
// This function actually generates the ray-traced image.
/////////////////////////////////////////////////////////
void render() {
    int x, y;
    GLubyte r, g, b;
    int p;

    ambient_light = glm::vec3(0, 0, 0);
    for (auto light : scene_lights) {
        ambient_light += light.get_color() * ambient_fraction;
    }

    /// copy light vector into constant memory
    float* h_light = new float[12];
    int h_light_size = sizeof(float) * 4;
    h_light[0] = scene_lights[0].get_color().x;
    h_light[1] = scene_lights[0].get_color().y;
    h_light[2] = scene_lights[0].get_color().z;
    h_light[3] = scene_lights[0].get_pos().x;
    h_light[4] = scene_lights[0].get_pos().y;
    h_light[5] = scene_lights[0].get_pos().z;

    h_light[6] = scene_lights[1].get_color().x;
    h_light[7] = scene_lights[1].get_color().y;
    h_light[8] = scene_lights[1].get_color().z;
    h_light[9] = scene_lights[1].get_pos().x;
    h_light[10] = scene_lights[1].get_pos().y;
    h_light[11] = scene_lights[1].get_pos().z;
    cudaMemcpyToSymbol(c_lights, h_light, h_light_size);

    int am_sz = sizeof(float) * 11;
    float* h_ambient = (float*)malloc(am_sz);
    h_ambient[0] = ambient_light.x;
    h_ambient[1] = ambient_light.y;
    h_ambient[2] = ambient_light.z;
    h_ambient[3] = clipL;
    h_ambient[4] = clipR;
    h_ambient[5] = clipB;
    h_ambient[6] = clipT;
    h_ambient[7] = clipN;
    h_ambient[8] = eye.x;
    h_ambient[9] = eye.y;
    h_ambient[10] = eye.z;
    cudaMemcpyToSymbol(c_ambient, h_ambient, am_sz);

    // #host variable: 5
    // #device variable: 5    
    // input allocation
    Matrix h_dcs_x = AllocateMatrix(winHeight, winWidth, 0); // initialize host matrix
    Matrix d_dcs_x = AllocateDeviceMatrix(h_dcs_x); // allocate device matrix memory
    CopyToDeviceMatrix(h_dcs_x, d_dcs_x); // copy host to device
    Matrix h_dcs_y = AllocateMatrix(winHeight, winWidth, 1); // initialize host matrix
    Matrix d_dcs_y = AllocateDeviceMatrix(h_dcs_y); // allocate device matrix memory
    CopyToDeviceMatrix(h_dcs_y, d_dcs_y); // copy host to device

    // output allocation 
    int pic = winHeight * winWidth;
    Ray* h_res; // host ray arr 
    size_t h_res_byte = pic * sizeof(Ray);
    h_res = (Ray*)malloc(h_res_byte);
    Ray* d_res; // device ray arr
    cudaMalloc(&d_res, h_res_byte);

    // execute kernel
    dim3 dimBlock(32, 32);
    int gridX = (int)ceil((float)winWidth / 32);
    int gridY = (int)ceil((float)winHeight / 32);
    dim3 dimGrid(gridX, gridY);

    Hit* h_hits, * d_hits;
    size_t hits_byte = scene_objects.size() * sizeof(Hit);
    h_hits = (Hit*)malloc(hits_byte);

    // pointer arr of scene_objects
    size_t obs_byte = scene_objects.size() * sizeof(Object);
    Object* scene_obs = (Object*)malloc(obs_byte);
    for (int i = 0; i < scene_objects.size(); i++)
    {
        scene_obs[i] = *scene_objects[i];
    }
    Object* d_obs;
    cudaMalloc(&d_obs, obs_byte);
    cudaMemcpy(d_obs, scene_obs, obs_byte, cudaMemcpyHostToDevice);
    cudaMalloc(&d_hits, hits_byte);
    cudaMemcpy(d_hits, h_hits, hits_byte, cudaMemcpyHostToDevice);

    ray_color <<<dimGrid, dimBlock >>> (d_dcs_x, d_dcs_y, winWidth, d_res, Mvcswcs);
    cudaDeviceSynchronize();

    first_hit <<<dimGrid, dimBlock >>> (d_res, scene_objects.size(), d_hits, d_obs);
    // copy result from device to host
    cudaMemcpy(h_res, d_res, h_res_byte, cudaMemcpyDeviceToHost);

    // assign data to glm vec3
    for (y = 0; y < winHeight; y++) {
        for (x = 0; x < winWidth; x++) {

            //debugOn = (y == winHeight / 2 && x == winHeight / 2);

            if (debugOn) {
                cout << "pixel (" << x << " " << y << ")\n";
                cout.flush();
            }

            p = (y * winWidth + x) * 3;
            glm::vec3 pixel_color = h_res[y * winWidth + x].get_color();
            pixel_color = glm::clamp(pixel_color, 0.0f, 1.0f);

            r = (GLubyte)(pixel_color.r * 255.0);
            g = (GLubyte)(pixel_color.g * 255.0);
            b = (GLubyte)(pixel_color.b * 255.0);

            img[p] = r;
            img[p + 1] = g;
            img[p + 2] = b;
        }
    }

    cudaFree(d_obs);
    cudaFree(d_res);
    cudaFree(d_hits);
    free(h_res);
    free(h_hits);
    free(scene_obs);
    FreeMatrix(&h_dcs_y);
    FreeMatrix(&h_dcs_x);
    FreeDeviceMatrix(&d_dcs_x);
    FreeDeviceMatrix(&d_dcs_y);
}

//////////////////////////////////////////////////////
//
// Displays, on STDOUT, the colour of the pixel that
//  the user clicked on.
//
//////////////////////////////////////////////////////

//void mouse_button_callback(GLFWwindow* window, int button,
//    int action, int mods)
//{
//    if (button != GLFW_MOUSE_BUTTON_LEFT)
//        return;
//
//    if (action == GLFW_PRESS)
//    {
//        debugOn = true;
//
//        // Get the mouse's position.
//
//        double xpos, ypos;
//        int W, H;
//        glfwGetCursorPos(window, &xpos, &ypos);
//        glfwGetWindowSize(window, &W, &H);
//
//        // mouse position, as a fraction of the window dimensions
//        // The y mouse coord increases as you move down,
//        // but our yDCS increases as you move up.
//        double mouse_fx = xpos / float(W);
//        double mouse_fy = (W - 1 - ypos) / float(W);
//
//        int xDCS = (int)(mouse_fx * winWidth + 0.5);
//        int yDCS = (int)(mouse_fy * winHeight + 0.5);
//
//        glm::vec3 pixelColor = ray_color(xDCS, yDCS);
//
//        if (debugOn) {
//            cout << "cursorpos:" << xpos << " " << ypos << "\n";
//            cout << "Width Height: " << winWidth << " " << winHeight << "\n";
//            cout << "Window Size: " << W << " " << H << "\n";
//            cout << "Pixel at (x y)=(" << xDCS << " " << yDCS << ")\n";
//            cout << "Pixel Color = " << glm::to_string(pixelColor) << endl;
//        }
//        debugOn = false;
//    }
//}

/////////////////////////////////////////////////////////
// Call this when scene has changed, and we need to re-run
// the ray tracer.
/////////////////////////////////////////////////////////

void camera_changed() {
    float dummy = 0;
    cam_param_changed(dummy);
}

/////////////////////////////////////////////////////////
// Called when user modifies a camera parameter.
/////////////////////////////////////////////////////////
void cam_param_changed(float param) {
    setup_camera();
    frame_buffer_stale = true;
}

/////////////////////////////////////////////////////////
// Check if window was resized.
/////////////////////////////////////////////////////////

bool get_was_window_resized() {
    int new_W = cam.get_win_W();
    int new_H = cam.get_win_H();

    // cout << "window resized to " << new_W << " " << new_H << "\n";

    if (new_W != winWidth || new_H != winHeight) {
        camera_changed();
        winWidth = new_W;
        winHeight = new_H;
        return true;
    }

    return false;
}

///////////////////////////////////////////////////
// Resets the camera parameters.
///////////////////////////////////////////////////

void reset_camera(float dummy) {
    eye = eye_home;
    lookat = lookat_home;
    vup = vup_home;

    clipL = clip_home[0];
    clipR = clip_home[1];
    clipB = clip_home[2];
    clipT = clip_home[3];
    clipN = clip_home[4];

    ambient_fraction = ambient_fraction_home;

    camera_changed();
}


/////////////////////////////////////////////////////////
// Called on a GLFW error.
/////////////////////////////////////////////////////////

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

//////////////////////////////////////////////////////
// Quit if the user hits "q" or "ESC".
// All other key presses are passed to the UI.
//////////////////////////////////////////////////////

static void key_callback(GLFWwindow* window, int key,
    int scancode, int action, int mods)
{
    if (key == GLFW_KEY_Q ||
        key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (action == GLFW_RELEASE) {
        the_ui.handle_key(key);
    }
}


//////////////////////////////////////////////////////
// Show the image.
//////////////////////////////////////////////////////

void display() {
    glClearColor(.1f, .1f, .1f, 1.f);   /* set the background colour */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cam.begin_drawing();

    glRasterPos3d(0.0, 0.0, 0.0);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (frame_buffer_stale) {
        //
        // Don't re-render the scene EVERY time display() is called.
        // It might get called if the window is moved, or if it
        // is exposed.  Only re-render if the window is RESIZED.
        // Resizing triggers a call to handleReshape, which sets
        // frameBufferStale.
        //
        render();
        frame_buffer_stale = false;
    }

    //
    // This paints the current image buffer onto the screen.
    //
    glDrawPixels(winWidth, winHeight,
        GL_RGB, GL_UNSIGNED_BYTE, img);

    glFlush();
}

///////////////////////////////////////////////////////////////////
// Set up the keyboard UI.
//////////////////////////////////////////////////////////////////
void init_UI() {
    // These variables will trigger a call-back when they are changed.
    the_ui.add_variable("Eye X", &eye.x, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Eye Y", &eye.y, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Eye Z", &eye.z, -10, 10, 0.2, cam_param_changed);

    the_ui.add_variable("Shadows", &show_shadows, 0, 1, 1, cam_param_changed);
    the_ui.add_variable("Ambient Fraction", &ambient_fraction, 0, 1, 0.1,
        cam_param_changed);

    the_ui.add_variable("Ref X", &lookat.x, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Ref Y", &lookat.y, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Ref Z", &lookat.z, -10, 10, 0.2, cam_param_changed);

    the_ui.add_variable("Vup X", &vup.x, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Vup Y", &vup.y, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Vup Z", &vup.z, -10, 10, 0.2, cam_param_changed);

    the_ui.add_variable("Clip L", &clipL, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Clip R", &clipR, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Clip B", &clipB, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Clip T", &clipT, -10, 10, 0.2, cam_param_changed);
    the_ui.add_variable("Clip N", &clipN, -10, 10, 0.2, cam_param_changed);

    static float dummy2 = 0;
    the_ui.add_variable("Reset Camera", &dummy2, 0, winWidth,
        0.001, reset_camera);

    the_ui.done_init();

}

//////////////////////////////////////////////////////
// Main program.
//////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    init_UI();
    if (argc < 2) {
        cerr << "Usage:\n";
        cerr << "  rt <scene-file.txt>\n";
        cerr << "press Control-C to exit\n";
        char line[100];
        cin >> line;
        exit(EXIT_FAILURE);
    }

    read_scene(argv[1]);

    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        cerr << "glfwInit failed!\n";
        cerr << "PRESS Control-C to quit\n";
        char line[100];
        cin >> line;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(winWidth, winHeight,
        "Ray Traced Scene", NULL, NULL);

    if (!window)
    {
        cerr << "glfwCreateWindow failed!\n";
        cerr << "PRESS Control-C to quit\n";
        char line[100];
        cin >> line;

        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    int w = winWidth;
    int h = winHeight;

    cam = Camera(0, 0, w, h, w, h, window);
    setup_camera();

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(window))
    {
        cam.check_resize();
        setup_camera();

        display();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
