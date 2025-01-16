#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>

// Constants
constexpr int DEFAULT_NUM_FRAMES = 60; // Default number of frames
constexpr int NUM_VERTICES = 1024;
constexpr int NUM_FRAGMENTS = 2048;

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

    // Generate random vertex data
    std::vector<float> hostVertexBuffer(NUM_VERTICES * 3);
    for (int i = 0; i < NUM_VERTICES * 3; ++i) {
        hostVertexBuffer[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Launch and process frames dynamically
    std::vector<std::thread> syncThreads;
    for (int frame = 0; frame < numFrames; ++frame) {
        std::cout << "Launching pipeline for frame " << frame << std::endl;

        // Allocate memory and create a stream for this frame
        float *vertexBuffer, *transformedVertexBuffer, *fragmentBuffer, *frameBuffer;
        hipStream_t stream;
        hipStreamCreate(&stream);
        checkHIPError(hipMalloc(&vertexBuffer, NUM_VERTICES * 3 * sizeof(float)), "Allocating vertex buffer");
        checkHIPError(hipMalloc(&transformedVertexBuffer, NUM_VERTICES * 3 * sizeof(float)), "Allocating transformed vertex buffer");
        checkHIPError(hipMalloc(&fragmentBuffer, NUM_FRAGMENTS * 2 * sizeof(float)), "Allocating fragment buffer");
        checkHIPError(hipMalloc(&frameBuffer, NUM_FRAGMENTS * sizeof(float)), "Allocating frame buffer");

        // Copy vertex data to the device
        checkHIPError(hipMemcpy(vertexBuffer, hostVertexBuffer.data(), NUM_VERTICES * 3 * sizeof(float), hipMemcpyHostToDevice), "Copying vertex data");

        // Launch pipeline stages
        vertexShader<<<(NUM_VERTICES + 255) / 256, 256, 0, stream>>>(vertexBuffer, transformedVertexBuffer, NUM_VERTICES);
        rasterizer<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, stream>>>(transformedVertexBuffer, fragmentBuffer, NUM_FRAGMENTS);
        fragmentShader<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, stream>>>(fragmentBuffer, frameBuffer, NUM_FRAGMENTS);

        // Synchronize, clean up, and destroy the stream in a separate thread
        syncThreads.emplace_back([frame, vertexBuffer, transformedVertexBuffer, fragmentBuffer, frameBuffer, stream]() {
            hipStreamSynchronize(stream);
            std::cout << "Frame " << frame << " is ready!" << std::endl;

            // Free memory for this frame
            hipFree(vertexBuffer);
            hipFree(transformedVertexBuffer);
            hipFree(fragmentBuffer);
            hipFree(frameBuffer);

            // Destroy the stream
            hipStreamDestroy(stream);
            std::cout << "Stream for frame " << frame << " destroyed." << std::endl;
        });
    }

    // Join all synchronization threads
    for (auto& thread : syncThreads) {
        thread.join();
    }

    std::cout << "All frames processed." << std::endl;

    return 0;
}
