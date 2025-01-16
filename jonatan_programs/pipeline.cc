#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread>
#include <chrono>

// Constants
constexpr int DEFAULT_NUM_FRAMES = 60; // Default number of frames
constexpr int NUM_VERTICES = 1024;
constexpr int NUM_FRAGMENTS = 2048;

// Buffers
float** vertexBuffers;
float** transformedVertexBuffers;
float** fragmentBuffers;
float** frameBuffers;

// Simulate vertex processing (vertex shader)
__global__ void vertexShader(float* input, float* output, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        output[idx * 3 + 0] = input[idx * 3 + 0] * 0.5f + 1.0f;
        output[idx * 3 + 1] = input[idx * 3 + 1] * 0.5f + 1.0f;
        output[idx * 3 + 2] = input[idx * 3 + 2] * 0.5f + 1.0f;
    }
}

// Simulate rasterization
__global__ void rasterizer(float* transformedVertices, float* fragments, int numFragments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFragments) {
        fragments[idx * 2 + 0] = transformedVertices[(idx % NUM_VERTICES) * 3 + 0] + 0.1f;
        fragments[idx * 2 + 1] = transformedVertices[(idx % NUM_VERTICES) * 3 + 1] + 0.1f;
    }
}

// Simulate fragment shading (fragment shader)
__global__ void fragmentShader(float* fragments, float* frameBuffer, int numFragments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFragments) {
        float color = fragments[idx * 2 + 0] * 0.8f + fragments[idx * 2 + 1] * 0.6f;
        frameBuffer[idx] = color;
    }
}

void checkHIPError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << msg << ": " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    int numFrames = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_NUM_FRAMES;

    if (numFrames <= 0) {
        std::cerr << "Invalid number of frames. Must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "Processing " << numFrames << " frames." << std::endl;

    // Allocate memory for each frame's buffers
    vertexBuffers = new float*[numFrames];
    transformedVertexBuffers = new float*[numFrames];
    fragmentBuffers = new float*[numFrames];
    frameBuffers = new float*[numFrames];

    for (int i = 0; i < numFrames; ++i) {
        checkHIPError(hipMalloc(&vertexBuffers[i], NUM_VERTICES * 3 * sizeof(float)), "Allocating vertex buffer");
        checkHIPError(hipMalloc(&transformedVertexBuffers[i], NUM_VERTICES * 3 * sizeof(float)), "Allocating transformed vertex buffer");
        checkHIPError(hipMalloc(&fragmentBuffers[i], NUM_FRAGMENTS * 2 * sizeof(float)), "Allocating fragment buffer");
        checkHIPError(hipMalloc(&frameBuffers[i], NUM_FRAGMENTS * sizeof(float)), "Allocating frame buffer");
    }

    // Generate random vertex data for the first frame
    std::vector<float> hostVertexBuffer(NUM_VERTICES * 3);
    for (int i = 0; i < NUM_VERTICES * 3; ++i) {
        hostVertexBuffer[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < numFrames; ++i) {
        checkHIPError(hipMemcpy(vertexBuffers[i], hostVertexBuffer.data(), NUM_VERTICES * 3 * sizeof(float), hipMemcpyHostToDevice), "Copying vertex data");
    }

    // Create streams for each frame
    hipStream_t* streams = new hipStream_t[numFrames];
    for (int i = 0; i < numFrames; ++i) {
        hipStreamCreate(&streams[i]);
    }

    // Launch and synchronize pipeline stages dynamically
    for (int frame = 0; frame < numFrames; ++frame) {
        std::cout << "Launching pipeline for frame " << frame << std::endl;

        // Vertex shader
        vertexShader<<<(NUM_VERTICES + 255) / 256, 256, 0, streams[frame]>>>(
            vertexBuffers[frame], transformedVertexBuffers[frame], NUM_VERTICES);

        // Rasterizer
        rasterizer<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, streams[frame]>>>(
            transformedVertexBuffers[frame], fragmentBuffers[frame], NUM_FRAGMENTS);

        // Fragment shader
        fragmentShader<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, streams[frame]>>>(
            fragmentBuffers[frame], frameBuffers[frame], NUM_FRAGMENTS);

        // Use a separate thread to synchronize the frame and log readiness
        std::thread([frame, &streams]() {
            hipStreamSynchronize(streams[frame]);
            std::cout << "Frame " << frame << " is ready!" << std::endl;
        }).detach();
    }

    // Cleanup
    for (int i = 0; i < numFrames; ++i) {
        std::cout << "Destroying stream for frame " << i << std::endl;
        hipStreamDestroy(streams[i]);
        hipFree(vertexBuffers[i]);
        hipFree(transformedVertexBuffers[i]);
        hipFree(fragmentBuffers[i]);
        hipFree(frameBuffers[i]);
    }
}
