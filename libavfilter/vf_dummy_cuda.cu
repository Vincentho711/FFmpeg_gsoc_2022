/*
 * Copyright (c) 2022, Dummy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cuda/vector_helpers.cuh"
#include <math.h>

extern "C" {

__global__ void Process_uchar(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_U, cudaTextureObject_t src_tex_V,
                              uchar *dst_Y, uchar *dst_U, uchar *dst_V,
                              int width, int height, int pitch,
                              int width_uv, int height_uv, int pitch_uv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width)
        return;
    dst_Y[y*pitch + x] = tex2D<float>(src_tex_Y, x, y) * 255;

    if (y >= height_uv || x >= width_uv)
        return;
    dst_U[y*pitch_uv + x] = tex2D<float>(src_tex_U, x, y) * 255;
    dst_V[y*pitch_uv + x] = tex2D<float>(src_tex_V, x, y) * 255;
}

__global__ void Process_uchar2(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_UV, cudaTextureObject_t unused1,
                               uchar *dst_Y, uchar2 *dst_UV, uchar *unused2,
                               int width, int height, int pitch,
                               int width_uv, int height_uv, int pitch_uv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width)
        return;
    dst_Y[y*pitch + x] = tex2D<float>(src_tex_Y, x, y) * 255;

    if (y >= height_uv || x >= width_uv)
        return;
    float2 uv = tex2D<float2>(src_tex_UV, x, y) * 255;
    dst_UV[y*pitch_uv + x] = make_uchar2(uv.x, uv.y);
}

// Gaussian Blur
// Generate (kernel_dim x kernel_dim) Gaussian kernel
__host__ void generate_gaussian(double *kernel, int kernel_dim, double stdev)
{
    double constant = 1 / (2.0 * M_PI * pow(stdev, 2) );
    int radius = floor(kernel_dim / 2.0);

    for (int i = -radius; i < radius + 1; ++i)
		for (int j = -radius; j < radius + 1; ++j)
			kernel[(i + radius) * kernel_dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (pow(stdev, 2) * 2)));
}

// Gaussian convolution
__global__ void gaussian_blur(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_U, cudaTextureObject_t src_tex_V,
                              uchar *dst_Y, uchar *dst_U, uchar *dst_V,
                              int width, int height, int pitch,
                              int width_uv, int height_uv, int pitch_uv,
                              double *kernel, int kernel_dim)
{
    // Work out true x, y dim with border
    int true_x_dim = blockDim.x - (kernel_dim - 1);
    int true_y_dim = blockDim.y - (kernel_dim - 1);

    int x = (blockIdx.x * true_x_dim) + threadIdx.x;
	int y = (blockIdx.y * true_y_dim) + threadIdx.y;
    
    // Shared memory
    extern __shared__ double y_buffer[];
    extern __shared__ double u_buffer[];
    extern __shared__ double v_buffer[];

    // Handles Y
    if (x < width && y < height)
    {
        // Load src into buffer 
        y_buffer[threadIdx.y * pitch + threadIdx.x] = tex2D<float>(src_tex_Y, x, y);
        
    }
    else
    {
        y_buffer[threadIdx.y * pitch + threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Handles U,V
    if (x < width_uv && y < height_uv)
    {
        // Load src into buffer 
        u_buffer[threadIdx.y * pitch + threadIdx.x] = tex2D<float>(src_tex_U, x, y);
        v_buffer[threadIdx.y * pitch + threadIdx.x] = tex2D<float>(src_tex_V, x, y);
        
    }
    else
    {
        u_buffer[threadIdx.y * pitch + threadIdx.x] = 0.0;
        v_buffer[threadIdx.y * pitch + threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Convolve buffers with kernel
    if (threadIdx.x < true_x_dim && threadIdx.y < true_y_dim) 
    {
        double y_sum = 0.0;
        double u_sum = 0.0;
        double v_sum = 0.0;
        for (int i = 0; i < kernel_dim; ++i)
        {
            for (int j = 0; j < kernel_dim; ++j)
            {
                y_sum += y_buffer[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * kernel[(i * kernel_dim) + j];
                u_sum += u_buffer[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * kernel[(i * kernel_dim) + j];
                v_sum += v_buffer[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * kernel[(i * kernel_dim) + j];
            }
        }
        // Convert back to byte and output to dst
        dst_Y[x * pitch + y] = y_sum * 255;
        dst_U[x * pitch + y] = u_sum * 255;
        dst_V[x * pitch + y] = v_sum * 255;
    }
    
}
    

}
