#include <hip/hip_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <cstdlib>
#include <mutex>
#include <condition_variable>

// Constants
constexpr int DEFAULT_NUM_FRAMES = 60; // Default number of frames
constexpr int NUM_VERTICES = 1024;
constexpr int NUM_FRAGMENTS = 2048;
constexpr int MAX_CONCURRENT_FRAMES = 10; // Maximum number of frames processed concurrently

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

    std::cout << "Processing " << numFrames << " frames with max " << MAX_CONCURRENT_FRAMES << " concurrent frames." << std::endl;

    // Generate random vertex data
    std::vector<float> hostVertexBuffer(NUM_VERTICES * 3);
    for (int i = 0; i < NUM_VERTICES * 3; ++i) {
        hostVertexBuffer[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Mutex and condition variable for concurrency control
    std::mutex mtx;
    std::condition_variable cv;
    int activeFrames = 0;

    // Launch frames dynamically
    for (int frame = 0; frame < numFrames; ++frame) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return activeFrames < MAX_CONCURRENT_FRAMES; });
        ++activeFrames;
        lock.unlock();

        std::thread([frame, &hostVertexBuffer, &mtx, &cv, &activeFrames]() {
            float *vertexBuffer, *transformedVertexBuffer, *fragmentBuffer, *frameBuffer;
            hipStream_t stream;
            hipStreamCreate(&stream);

            // Allocate memory and copy data
            checkHIPError(hipMalloc(&vertexBuffer, NUM_VERTICES * 3 * sizeof(float)), "Allocating vertex buffer");
            checkHIPError(hipMalloc(&transformedVertexBuffer, NUM_VERTICES * 3 * sizeof(float)), "Allocating transformed vertex buffer");
            checkHIPError(hipMalloc(&fragmentBuffer, NUM_FRAGMENTS * 2 * sizeof(float)), "Allocating fragment buffer");
            checkHIPError(hipMalloc(&frameBuffer, NUM_FRAGMENTS * sizeof(float)), "Allocating frame buffer");
            checkHIPError(hipMemcpy(vertexBuffer, hostVertexBuffer.data(), NUM_VERTICES * 3 * sizeof(float), hipMemcpyHostToDevice), "Copying vertex data");

            // Launch pipeline
            vertexShader<<<(NUM_VERTICES + 255) / 256, 256, 0, stream>>>(vertexBuffer, transformedVertexBuffer, NUM_VERTICES);
            rasterizer<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, stream>>>(transformedVertexBuffer, fragmentBuffer, NUM_FRAGMENTS);
            fragmentShader<<<(NUM_FRAGMENTS + 255) / 256, 256, 0, stream>>>(fragmentBuffer, frameBuffer, NUM_FRAGMENTS);

            hipStreamSynchronize(stream);
            std::cout << "Frame " << frame << " is ready, and currently " << activeFrames << " frames are being processed." << std::endl;

            // Cleanup
            hipFree(vertexBuffer);
            hipFree(transformedVertexBuffer);
            hipFree(fragmentBuffer);
            hipFree(frameBuffer);
            hipStreamDestroy(stream);

            // Update active frames count
            std::unique_lock<std::mutex> lock(mtx);
            --activeFrames;
            lock.unlock();
            cv.notify_one();
        }).detach();
    }

    // Wait for all threads to finish
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return activeFrames == 0; });
    }

    std::cout << "All frames processed." << std::endl;

    return 0;
}
