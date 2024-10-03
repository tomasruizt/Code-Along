#define STB_IMAGE_IMPLEMENTATION  // Enable the implementation
#include "stb_image.h"            // Include the stb_image header
#include "grayscale.h"
#include <stdio.h>                // Standard I/O library for printing
#include <stdlib.h>               // For memory allocation

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

    // You can access the image data through the 'img' pointer now
    // For example, pixel at (x, y) would be at img[(y * width + x) * channels]

    // Free the image memory after use
    stbi_image_free(img);

    return 0;
}
