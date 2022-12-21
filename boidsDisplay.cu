/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

#define DEBUG_MSG 1

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>

// CUDA helper functions
#include <helper_cuda.h>
#include <vector_types.h>

#define MAX_THREADS_PER_BLOCK 1024
#define AS_INCLUDE
#include "boids.cu"

#define REFRESH_DELAY 1 // ms
#define NUM_OF_BOIDS 100
#define BOID_SIZE 0.02
#define MAX(a,b) ((a > b) ? a : b)

// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 10;       // FPS limit for sampling
unsigned long frameCount = 0;
float sim_time = 0.0;
clock_t previous_fps_update_time;
int running = 1;
clock_t last_frame_time;

Fish* in_fishes;
Fish* out_fishes;
int* neighbour_cell_buffer;

bool runSimulation(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void copyFishesToVbo(struct cudaGraphicsResource **vbo_resource);

__global__ void copy_to_vbo(float4* pos, Fish fishes, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= numElements){
        return;
    }

    // scale to [-1, 1]
    float u = (float)fishes.x[i] / SCENE_SIZE;
    u = u*2.0f - 1.0f;
    float v = (float)fishes.y[i] / SCENE_SIZE;
    v = v*2.0f - 1.0f;
    float w = 0;
    #ifdef USE_3D
    w = (float)fishes.z[i] / SCENE_SIZE;
    w = w*2.0f - 1.0f;
    #endif

    #ifdef USE_3D
    float len_v = (float)length(fishes.vx[i], fishes.vy[i], fishes.vz[i]);
    #else
    float len_v = (float)length(fishes.vx[i], fishes.vy[i]);
    #endif

    float dx = fishes.vx[i] / len_v * BOID_SIZE;
    float dy = fishes.vy[i] / len_v * BOID_SIZE;
    float dz = 0;
    #ifdef USE_3D
    dz = fishes.vz[i] / len_v * BOID_SIZE;
    #endif

    // write output vertices
    if(dx * dy == 0){
        pos[3*i]     = make_float4(u + dx, v + dy, w + dz, 1.0f);
        pos[3*i + 1] = make_float4(u - BOID_SIZE/3, v, w, 1.0f);
        pos[3*i + 2] = make_float4(u + BOID_SIZE/3, v, w, 1.0f);
    }
    else{
        pos[3*i]     = make_float4(u + dx, v + dy, w + dz, 1.0f);
        pos[3*i + 1] = make_float4(u - dy/3, v + dx/3, w, 1.0f);
        pos[3*i + 2] = make_float4(u + dy/3, v - dx/3, w, 1.0f);
    }
}

int main(int argc, char **argv)
{
    #if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
    #endif

    printf("starting simulation...\n");

    runSimulation(argc, argv);

    printf("simulation completed\n");
    exit(EXIT_SUCCESS);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount >= fpsLimit)
    {
        clock_t now = clock();
        float avgFPS = fpsCount / ((now - previous_fps_update_time) / ((float)CLOCKS_PER_SEC));
        fpsCount = 0;
        previous_fps_update_time = now;
        fpsLimit = (int)MAX(avgFPS, 2);

        char fps[256];
        sprintf(fps, "Boids simulation: %3.1f fps", avgFPS);
        glutSetWindowTitle(fps);
    }
}

bool initGL(int *argc, char **argv)
{
    T("initGL()");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Boids simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

bool runSimulation(int argc, char **argv)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (!initGL(&argc, argv))
    {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutCloseFunc(cleanup);

    T("createVBO()");
    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    initSimulation(&in_fishes, &out_fishes, &neighbour_cell_buffer, NUM_OF_BOIDS);

    // run the cuda part
    copyFishesToVbo(&cuda_vbo_resource);

    // init time variables
    previous_fps_update_time = clock();
    last_frame_time = clock();

    // start rendering mainloop
    glutMainLoop();

    return true;
}

void copyFishesToVbo(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    
    copy_to_vbo<<<1, NUM_OF_BOIDS>>>(dptr, *in_fishes, NUM_OF_BOIDS);
    deviceCheckErrors("copy_to_vbo");

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = NUM_OF_BOIDS * 4 * sizeof(float) * 3; // triangle per boid
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// Display callback
void display()
{
    T("display()");
    if(running == 0){
        return;
    }

    // calculate dt
    clock_t now = clock();
    double dt = (now - last_frame_time) / ((double)CLOCKS_PER_SEC); 
    last_frame_time = now;

    // run CUDA kernel to generate vertex positions
    advance(in_fishes, out_fishes, NUM_OF_BOIDS, neighbour_cell_buffer, dt);
    T("copyFishesToVbo()");
    copyFishesToVbo(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, NUM_OF_BOIDS * 3);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    // Advance time
    sim_time += dt;

    computeFPS();
    T("display() finished");
    DEBUG("\n");
}

// Callback that posts redisplay signals
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void cleanup()
{
    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }

    freeFishes(in_fishes);
    freeFishes(out_fishes);
    deviceFree(neighbour_cell_buffer);
}

// Keyboard events handler
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27):  // ESC
            glutDestroyWindow(glutGetWindow());
            return;
        case(32):   // SPACE
            running = 1-running;    // 1 => 0, 0 => 1
            return;
    }
}

// Mouse event handlers
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}