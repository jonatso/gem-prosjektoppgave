#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>
#include <mutex>
#include <condition_variable>
#include <thread>

// Constants
constexpr int DEFAULT_NUM_FRAMES = 60; // Default number of frames
constexpr int NUM_VERTICES = 1024;
constexpr int NUM_TRIANGLES = NUM_VERTICES / 3;
constexpr int NUM_FRAGMENTS = 2048;
constexpr int TEXTURE_WIDTH = 128;
constexpr int TEXTURE_HEIGHT = 128;
constexpr int TILE_SIZE = 64;
constexpr int MAX_CONCURRENT_FRAMES = 10;

// Vertex Shader: Simulates transforming vertices
__global__ void vertexShader(float* input, float* output, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        for (int i = 0; i < 4; ++i) {
            output[idx * 4 + i] = input[idx * 4 + i] * 1.1f + 0.5f;
        }
    }
}

// Rasterizer: Simulates generating fragments from triangles
__global__ void rasterizer(float* vertices, float* fragments, int numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTriangles) {
        int baseIdx = idx * 12; // 3 vertices * 4 floats per vertex
        float centroidX = 0.0f, centroidY = 0.0f;

        for (int i = 0; i < 3; ++i) {
            centroidX += vertices[baseIdx + i * 4];
            centroidY += vertices[baseIdx + i * 4 + 1];
        }
        centroidX /= 3.0f;
        centroidY /= 3.0f;

        fragments[idx * 2] = centroidX;
        fragments[idx * 2 + 1] = centroidY;
    }
}

// Fragment Shader: Simulates texturing and shading
__global__ void fragmentShader(float* fragments, float* texture, float* framebuffer, int numFragments) {
    __shared__ float tile[TILE_SIZE * 2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFragments) {
        float x = fragments[idx * 2];
        float y = fragments[idx * 2 + 1];

        int texX = static_cast<int>((x + 1.0f) * 0.5f * (TEXTURE_WIDTH - 1));
        int texY = static_cast<int>((y + 1.0f) * 0.5f * (TEXTURE_HEIGHT - 1));
        texX = min(max(texX, 0), TEXTURE_WIDTH - 1);
        texY = min(max(texY, 0), TEXTURE_HEIGHT - 1);

        float texValue = texture[texY * TEXTURE_WIDTH + texX];

        int tileIdx = threadIdx.x % TILE_SIZE;
        tile[tileIdx * 2] = x * texValue;
        tile[tileIdx * 2 + 1] = y * texValue;

        __syncthreads();

        framebuffer[idx * 3 + 0] = tile[tileIdx * 2];
        framebuffer[idx * 3 + 1] = tile[tileIdx * 2 + 1];
        framebuffer[idx * 3 + 2] = texValue;
    }
}

void checkHIPError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << msg << ": " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int numFrames = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_NUM_FRAMES;
    if (numFrames <= 0) {
        std::cerr << "Invalid number of frames. Must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "Processing " << numFrames << " frames with max " << MAX_CONCURRENT_FRAMES << " concurrent frames." << std::endl;

    // Generate vertex data
    std::vector<float> hostVertexBuffer(NUM_VERTICES * 4);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> vertexDist(-1.0f, 1.0f);
    for (float& val : hostVertexBuffer) {
        val = vertexDist(generator);
    }

    // Generate texture data
    std::vector<float> hostTexture(TEXTURE_WIDTH * TEXTURE_HEIGHT);
    std::uniform_real_distribution<float> textureDist(0.0f, 1.0f);
    for (float& val : hostTexture) {
        val = textureDist(generator);
    }

    // Allocate texture memory
    float *texture;
    checkHIPError(hipMalloc(&texture, TEXTURE_WIDTH * TEXTURE_HEIGHT * sizeof(float)), "Allocating texture");
    checkHIPError(hipMemcpy(texture, hostTexture.data(), TEXTURE_WIDTH * TEXTURE_HEIGHT * sizeof(float), hipMemcpyHostToDevice), "Copying texture data");

    // Concurrency control
    std::mutex mtx;
    std::condition_variable cv;
    int activeFrames = 0;

    // Launch frames
    for (int frame = 0; frame < numFrames; ++frame) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return activeFrames < MAX_CONCURRENT_FRAMES; });
            ++activeFrames;
        }

        std::thread([frame, &mtx, &cv, &activeFrames, &hostVertexBuffer, texture]() {
            float *vertexBuffer, *transformedVertexBuffer, *fragmentBuffer, *framebuffer;
            hipStream_t stream;
            hipStreamCreate(&stream);

            // Allocate per-frame buffers
            checkHIPError(hipMalloc(&vertexBuffer, NUM_VERTICES * 4 * sizeof(float)), "Allocating vertex buffer");
            checkHIPError(hipMalloc(&transformedVertexBuffer, NUM_VERTICES * 4 * sizeof(float)), "Allocating transformed vertex buffer");
            checkHIPError(hipMalloc(&fragmentBuffer, NUM_TRIANGLES * 2 * sizeof(float)), "Allocating fragment buffer");
            checkHIPError(hipMalloc(&framebuffer, NUM_FRAGMENTS * 3 * sizeof(float)), "Allocating framebuffer");

            // Copy vertex data to the device
            checkHIPError(hipMemcpy(vertexBuffer, hostVertexBuffer.data(), NUM_VERTICES * 4 * sizeof(float), hipMemcpyHostToDevice), "Copying vertex data");

            // Launch vertex shader
            hipLaunchKernelGGL(vertexShader, dim3((NUM_VERTICES + 255) / 256), dim3(256), 0, stream, vertexBuffer, transformedVertexBuffer, NUM_VERTICES);

            // Launch rasterizer
            hipLaunchKernelGGL(rasterizer, dim3((NUM_TRIANGLES + 255) / 256), dim3(256), 0, stream, transformedVertexBuffer, fragmentBuffer, NUM_TRIANGLES);

            // Launch fragment shader
            hipLaunchKernelGGL(fragmentShader, dim3((NUM_FRAGMENTS + 255) / 256), dim3(256), 0, stream, fragmentBuffer, texture, framebuffer, NUM_FRAGMENTS);

            // Synchronize and clean up
            hipStreamSynchronize(stream);
            std::cout << "Frame " << frame << " processed." << std::endl;

            hipFree(vertexBuffer);
            hipFree(transformedVertexBuffer);
            hipFree(fragmentBuffer);
            hipFree(framebuffer);
            hipStreamDestroy(stream);

            {
                std::unique_lock<std::mutex> lock(mtx);
                --activeFrames;
            }
            cv.notify_one();
        }).detach();
    }

    // Wait for all frames to finish
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return activeFrames == 0; });
    }

    hipFree(texture);
    std::cout << "All frames processed successfully." << std::endl;

    return 0;
}
