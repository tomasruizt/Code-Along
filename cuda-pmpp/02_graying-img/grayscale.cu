__global__
void toGrayscale(unsigned char *output, unsigned char *input, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];
        output[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}