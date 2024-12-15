#define STB_IMAGE_IMPLEMENTATION  // Enable the implementation
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  // Enable implementation for stb_image_write
#include "stb_image_write.h"
#include "grayscale.h"
#include <stdio.h>
#include <stdlib.h>  // For memory allocation

int main() {
    int width, height, channels;
    // Load the image file into memory (you can use a PNG, JPG, BMP, etc.)
    unsigned char *img = stbi_load("image.jpg", &width, &height, &channels, 0);

    if (img == NULL) {
        printf("Failed to load image\n");
        return 1;
    }

    // Print image details
    printf("Image loaded: Width = %d, Height = %d, Channels = %d\n", width, height, channels);

    // Allocate memory
    unsigned char* img_d;
    unsigned char* grayImg_d;
    unsigned char* grayImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    cudaMalloc((void**)&img_d, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&grayImg_d, width * height * sizeof(unsigned char));

    // Copy the image to the device
    cudaMemcpy(img_d, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Execute the grayscale kernel
    dim3 grid(ceil(width / 32.0), ceil(height / 32.0), 1);
    dim3 block(32, 32, 1);
    toGrayscale<<<grid, block>>>(grayImg_d, img_d, width, height);

    // Copy results back
    cudaMemcpy(grayImg, grayImg_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Write the image to a JPEG file
    int newChannels = 1;
    if (!stbi_write_jpg("grayscale.jpg", width, height, newChannels, grayImg, 100)) {
        printf("Error: Could not write the image!\n");
        stbi_image_free(img);
        return 1;
    }

    printf("Image successfully written to grayscale.jpg\n");

    // Free the image memory after use, and CUDA memory
    stbi_image_free(img);
    cudaFree(img_d);
    cudaFree(grayImg_d);
    free(grayImg);

    return 0;
}
