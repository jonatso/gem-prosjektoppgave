// First test of a custom GPU pipeline, to test running HIP kernels on the GPU.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Constants for framebuffer size
const int WIDTH = 800;
const int HEIGHT = 600;

// Vertex structure
struct Vertex {
    float x, y, z;
};

// Color structure
struct Color {
    float r, g, b;
};

// Simple transformation kernel (vertex shader)
__global__ void vertexShader(Vertex *vertices, int numVertices, float scaleX, float scaleY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        // printf("Vertex %d before transformation: (%f, %f, %f)\n", idx, vertices[idx].x, vertices[idx].y, vertices[idx].z);
        vertices[idx].x *= scaleX;
        vertices[idx].y *= scaleY;
        // printf("Vertex %d after transformation: (%f, %f, %f)\n", idx, vertices[idx].x, vertices[idx].y, vertices[idx].z);
    }
}

// Rasterization kernel (rasterizer)
__global__ void rasterizer(Vertex *vertices, int *indices, int numTriangles, Color *framebuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTriangles) {
        // Extract vertex indices for the triangle
        int v0 = indices[idx * 3 + 0];
        int v1 = indices[idx * 3 + 1];
        int v2 = indices[idx * 3 + 2];

        // printf("Processing triangle %d with vertices (%d, %d, %d)\n", idx, v0, v1, v2);

        // Compute triangle bounding box
        Vertex tri[3] = {vertices[v0], vertices[v1], vertices[v2]};
        int minX = fminf(fminf(tri[0].x, tri[1].x), tri[2].x);
        int maxX = fmaxf(fmaxf(tri[0].x, tri[1].x), tri[2].x);
        int minY = fminf(fminf(tri[0].y, tri[1].y), tri[2].y);
        int maxY = fmaxf(fmaxf(tri[0].y, tri[1].y), tri[2].y);

        // printf("Triangle %d bounding box: minX=%d, maxX=%d, minY=%d, maxY=%d\n", idx, minX, maxX, minY, maxY);

        // Clamp to framebuffer size
        minX = max(minX, 0);
        maxX = min(maxX, WIDTH - 1);
        minY = max(minY, 0);
        maxY = min(maxY, HEIGHT - 1);

        // Simple rasterization loop
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                // Placeholder: color the triangle area (no barycentric interpolation)
                framebuffer[y * WIDTH + x] = {1.0f, 0.0f, 0.0f}; // Red color
            }
        }
        // printf("Finished rasterizing triangle %d\n", idx);
    }
}

int main() {
    // Initialize vertices and indices for a single triangle
    std::vector<Vertex> hostVertices = {{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}};
    std::vector<int> hostIndices = {0, 1, 2};

    // Allocate device memory
    Vertex *deviceVertices;
    int *deviceIndices;
    Color *deviceFramebuffer;
    hipMalloc(&deviceVertices, hostVertices.size() * sizeof(Vertex));
    hipMalloc(&deviceIndices, hostIndices.size() * sizeof(int));
    hipMalloc(&deviceFramebuffer, WIDTH * HEIGHT * sizeof(Color));

    // Copy data to device
    hipMemcpy(deviceVertices, hostVertices.data(), hostVertices.size() * sizeof(Vertex), hipMemcpyHostToDevice);
    hipMemcpy(deviceIndices, hostIndices.data(), hostIndices.size() * sizeof(int), hipMemcpyHostToDevice);

    // Launch vertex shader
    dim3 blockSize(256);
    dim3 gridSize((hostVertices.size() + blockSize.x - 1) / blockSize.x);
    std::cout << "Launching vertex shader..." << std::endl;
    vertexShader<<<gridSize, blockSize>>>(deviceVertices, hostVertices.size(), 400.0f, 300.0f);
    hipDeviceSynchronize();

    // Launch rasterizer
    gridSize = dim3((hostIndices.size() / 3 + blockSize.x - 1) / blockSize.x);
    std::cout << "Launching rasterizer..." << std::endl;
    rasterizer<<<gridSize, blockSize>>>(deviceVertices, deviceIndices, hostIndices.size() / 3, deviceFramebuffer);
    hipDeviceSynchronize();

    // Copy framebuffer to host
    std::vector<Color> hostFramebuffer(WIDTH * HEIGHT);
    hipMemcpy(hostFramebuffer.data(), deviceFramebuffer, WIDTH * HEIGHT * sizeof(Color), hipMemcpyDeviceToHost);

    // Output framebuffer to console (as ASCII art for simplicity)
    for (int y = 0; y < HEIGHT; y += 20) {
        for (int x = 0; x < WIDTH; x += 20) {
            if (hostFramebuffer[y * WIDTH + x].r > 0.5f)
                std::cout << "#";
            else
                std::cout << ".";
        }
        std::cout << std::endl;
    }

    // Clean up
    hipFree(deviceVertices);
    hipFree(deviceIndices);
    hipFree(deviceFramebuffer);

    return 0;
}