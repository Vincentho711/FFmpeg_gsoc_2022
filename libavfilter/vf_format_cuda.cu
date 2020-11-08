/*
 * This file is part of FFmpeg.
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

static const ushort mask_10bit = 0xFFC0;
static const ushort mask_16bit = 0xFFFF;

static inline __device__ ushort conv_8to16(uchar in, ushort mask)
{
    return ((ushort)in | ((ushort)in << 8)) & mask;
}

static inline __device__ ushort2 conv_8to16(uchar2 in, ushort mask)
{
    return make_ushort2(
        conv_8to16(in.x, mask),
        conv_8to16(in.y, mask)
    );
}

static inline __device__ uchar conv_16to8(ushort in)
{
    return in >> 8;
}

static inline __device__ uchar2 conv_16to8(ushort2 in)
{
    return make_uchar2(
        conv_16to8(in.x),
        conv_16to8(in.y)
    );
}

static inline __device__ uchar conv_10to8(ushort in)
{
    return in >> 8;
}

static inline __device__ uchar2 conv_10to8(ushort2 in)
{
    return make_uchar2(
        conv_10to8(in.x),
        conv_10to8(in.y)
    );
}

static inline __device__ ushort conv_10to16(ushort in)
{
    return in | (in >> 10);
}

static inline __device__ ushort2 conv_10to16(ushort2 in)
{
    return make_ushort2(
        conv_10to16(in.x),
        conv_10to16(in.y)
    );
}

static inline __device__ ushort conv_16to10(ushort in)
{
    return in & mask_10bit;
}

static inline __device__ ushort2 conv_16to10(ushort2 in)
{
    return make_ushort2(
        conv_16to10(in.x),
        conv_16to10(in.y)
    );
}

template<typename T>
static inline __device__ T conv_444to420(const T *src, int pitch, int x, int y)
{
    unsigned tmp = (unsigned)src[ y      * pitch +  x] +
                   (unsigned)src[(y + 1) * pitch +  x] +
                   (unsigned)src[ y      * pitch + (x + 1)] +
                   (unsigned)src[(y + 1) * pitch + (x + 1)];
    return tmp / 4;
}

static inline __device__ ushort conv_444to420p16(const uchar *src, int pitch, int x, int y, ushort mask)
{
    unsigned tmp = (unsigned)conv_8to16(src[ y      * pitch +  x], mask_16bit) +
                   (unsigned)conv_8to16(src[(y + 1) * pitch +  x], mask_16bit) +
                   (unsigned)conv_8to16(src[ y      * pitch + (x + 1)], mask_16bit) +
                   (unsigned)conv_8to16(src[(y + 1) * pitch + (x + 1)], mask_16bit);
    return (tmp / 4) & mask;
}

template<typename T>
static inline __device__ T conv_420to444(const T *src, int width, int height, int pitch, int x, int y)
{
    int x1 = x / 2;
    int y1 = y / 2;
    int x2 = min(x1 + 1, width - 1);
    int y2 = min(y1 + 1, height - 1);

    intT tmp;
    vec_set_scalar(tmp, 0);
    tmp += to_intN<T, intT>(src[y1 * pitch + x1]);
    tmp += to_intN<T, intT>(src[y1 * pitch + x2]);
    tmp += to_intN<T, intT>(src[y2 * pitch + x1]);
    tmp += to_intN<T, intT>(src[y2 * pitch + x2]);

    return from_intN<T, intT>(tmp / 4);
}

template<typename T, typename O>
static inline __device__ O conv_420to444p16(const T *src, int width, int height, int pitch, int x, int y, ushort mask)
{
    int x1 = x / 2;
    int y1 = y / 2;
    int x2 = min(x1 + 1, (width / 2) - 1);
    int y2 = min(y1 + 1, (height / 2) - 1);

    intT tmp;
    vec_set_scalar(tmp, 0);
    tmp += to_intN<O, intT>(conv_8to16(src[y1 * pitch + x1], mask_16bit));
    tmp += to_intN<O, intT>(conv_8to16(src[y1 * pitch + x2], mask_16bit));
    tmp += to_intN<O, intT>(conv_8to16(src[y2 * pitch + x1], mask_16bit));
    tmp += to_intN<O, intT>(conv_8to16(src[y2 * pitch + x2], mask_16bit));

    return from_intN<O, intT>((tmp / 4) & mask);
}

#define FIX_PITCH(name) name ## _pitch /= sizeof(*name)

// yuv420p->X
extern "C" {

__global__ void Convert_yuv420p_nv12(int width, int height,
                                     uchar  *dst_y  , int dst_y_pitch , const uchar *src_y, int src_y_pitch,
                                     uchar2 *dst_uv , int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                     uchar  *unused0, int unused1     , const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);

        dst_uv[y * dst_uv_pitch + x] = make_uchar2(
            src_u[y * src_u_pitch + x],
            src_v[y * src_v_pitch + x]
        );
    }
}

__global__ void Convert_yuv420p_yuv444p(int width, int height,
                                        uchar *dst_y, int dst_y_pitch, const uchar *src_y, int src_y_pitch,
                                        uchar *dst_u, int dst_u_pitch, const uchar *src_u, int src_u_pitch,
                                        uchar *dst_v, int dst_v_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[y * dst_y_pitch + x] = src_y[y  * src_y_pitch + x];
        dst_u[y * dst_u_pitch + x] = conv_420to444(src_u, width, height, src_u_pitch, x, y);
        dst_v[y * dst_v_pitch + x] = conv_420to444(src_v, width, height, src_v_pitch, x, y);
    }
}

__global__ void Convert_yuv420p_p010le(int width, int height,
                                       ushort  *dst_y,  int dst_y_pitch,  const uchar *src_y, int src_y_pitch,
                                       ushort2 *dst_uv, int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                       ushort2 *unuse0, int unused_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_10bit);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);

        dst_uv[y * dst_uv_pitch + x] = make_ushort2(
            conv_8to16(src_u[y * src_u_pitch + x], mask_10bit),
            conv_8to16(src_v[y * src_v_pitch + x], mask_10bit)
        );
    }
}

__global__ void Convert_yuv420p_p016le(int width, int height,
                                       ushort  *dst_y,  int dst_y_pitch,  const uchar *src_y, int src_y_pitch,
                                       ushort2 *dst_uv, int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                       ushort2 *unuse0, int unused_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_16bit);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);

        dst_uv[y * dst_uv_pitch + x] = make_ushort2(
            conv_8to16(src_u[y * src_u_pitch + x], mask_16bit),
            conv_8to16(src_v[y * src_v_pitch + x], mask_16bit)
        );
    }
}

__global__ void Convert_yuv420p_yuv444p16le(int width, int height,
                                            ushort *dst_y, int dst_y_pitch, const uchar *src_y, int src_y_pitch,
                                            ushort *dst_u, int dst_u_pitch, const uchar *src_u, int src_u_pitch,
                                            ushort *dst_v, int dst_v_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(dst_u);
        FIX_PITCH(dst_v);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y  * src_y_pitch + x],  mask_16bit);
        dst_u[y * dst_u_pitch + x] = conv_420to444p16<uchar, ushort>(src_u, width, height, src_u_pitch, x, y, mask_16bit);
        dst_v[y * dst_v_pitch + x] = conv_420to444p16<uchar, ushort>(src_v, width, height, src_v_pitch, x, y, mask_16bit);
    }
}

}

// nv12->X
extern "C" {

__global__ void Convert_nv12_yuv420p(int width, int height,
                                     uchar *dst_y, int dst_y_pitch, const uchar  *src_y,  int src_y_pitch,
                                     uchar *dst_u, int dst_u_pitch, const uchar2 *src_uv, int src_uv_pitch,
                                     uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(src_uv);

        const uchar2 &uv = src_uv[y * src_uv_pitch + x];
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_nv12_yuv444p(int width, int height,
                                     uchar *dst_y, int dst_y_pitch, const uchar  *src_y,  int src_y_pitch,
                                     uchar *dst_u, int dst_u_pitch, const uchar2 *src_uv, int src_uv_pitch,
                                     uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_uv);

        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];

        uchar2 uv = conv_420to444(src_uv, width, height, src_uv_pitch, x, y);
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_nv12_p010le(int width, int height,
                                    ushort  *dst_y,  int dst_y_pitch,  const uchar  *src_y,  int src_y_pitch,
                                    ushort2 *dst_uv, int dst_uv_pitch, const uchar2 *src_uv, int src_uv_pitch,
                                    ushort2 *unuse0, int unused_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_10bit);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(src_uv);
        FIX_PITCH(dst_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_8to16(src_uv[y * src_uv_pitch + x], mask_10bit);
    }
}

__global__ void Convert_nv12_p016le(int width, int height,
                                    ushort  *dst_y,  int dst_y_pitch,  const uchar  *src_y,  int src_y_pitch,
                                    ushort2 *dst_uv, int dst_uv_pitch, const uchar2 *src_uv, int src_uv_pitch,
                                    ushort2 *unuse0, int unused_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_16bit);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(src_uv);
        FIX_PITCH(dst_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_8to16(src_uv[y * src_uv_pitch + x], mask_16bit);
    }
}

__global__ void Convert_nv12_yuv444p16le(int width, int height,
                                         ushort *dst_y, int dst_y_pitch, const uchar  *src_y,  int src_y_pitch,
                                         ushort *dst_u, int dst_u_pitch, const uchar2 *src_uv, int src_uv_pitch,
                                         ushort *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_uv);
        FIX_PITCH(dst_y);
        FIX_PITCH(dst_u);
        FIX_PITCH(dst_v);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_16bit);

        ushort2 uv = conv_420to444p16<uchar2, ushort2>(src_uv, width, height, src_uv_pitch, x, y, mask_16bit);
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

}

// yuv444p->X
extern "C" {

__global__ void Convert_yuv444p_yuv420p(int width, int height,
                                        uchar *dst_y, int dst_y_pitch, const uchar *src_y, int src_y_pitch,
                                        uchar *dst_u, int dst_u_pitch, const uchar *src_u, int src_u_pitch,
                                        uchar *dst_v, int dst_v_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;

            dst_u[y2 * dst_u_pitch + x2] = conv_444to420(src_u, src_u_pitch, x, y);
            dst_v[y2 * dst_v_pitch + x2] = conv_444to420(src_v, src_v_pitch, x, y);
        }
    }
}

__global__ void Convert_yuv444p_nv12(int width, int height,
                                     uchar  *dst_y , int dst_y_pitch,  const uchar *src_y, int src_y_pitch,
                                     uchar2 *dst_uv, int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                     uchar2 *unused, int unused_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_uchar2(
                conv_444to420(src_u, src_u_pitch, x, y),
                conv_444to420(src_v, src_v_pitch, x, y)
            );
        }
    }
}

__global__ void Convert_yuv444p_p010le(int width, int height,
                                       ushort  *dst_y,  int dst_y_pitch,  const uchar *src_y, int src_y_pitch,
                                       ushort2 *dst_uv, int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                       ushort2 *unused, int unused_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_10bit);

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_ushort2(
                conv_444to420p16(src_u, src_u_pitch, x, y, mask_10bit),
                conv_444to420p16(src_v, src_v_pitch, x, y, mask_10bit)
            );
        }
    }
}

__global__ void Convert_yuv444p_p016le(int width, int height,
                                       ushort  *dst_y,  int dst_y_pitch,  const uchar *src_y, int src_y_pitch,
                                       ushort2 *dst_uv, int dst_uv_pitch, const uchar *src_u, int src_u_pitch,
                                       ushort2 *unused, int unused_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_16bit);

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_ushort2(
                conv_444to420p16(src_u, src_u_pitch, x, y, mask_16bit),
                conv_444to420p16(src_v, src_v_pitch, x, y, mask_16bit)
            );
        }
    }
}

__global__ void Convert_yuv444p_yuv444p16le(int width, int height,
                                            ushort *dst_y, int dst_y_pitch, const uchar *src_y, int src_y_pitch,
                                            ushort *dst_u, int dst_u_pitch, const uchar *src_u, int src_u_pitch,
                                            ushort *dst_v, int dst_v_pitch, const uchar *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(dst_u);
        FIX_PITCH(dst_v);

        dst_y[y * dst_y_pitch + x] = conv_8to16(src_y[y * src_y_pitch + x], mask_16bit);
        dst_u[y * dst_u_pitch + x] = conv_8to16(src_u[y * src_u_pitch + x], mask_16bit);
        dst_v[y * dst_v_pitch + x] = conv_8to16(src_v[y * src_v_pitch + x], mask_16bit);
    }
}

}

// p010le->X
extern "C" {

__global__ void Convert_p010le_yuv420p(int width, int height,
                                       uchar *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                       uchar *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                       uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_10to8(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(src_uv);

        uchar2 uv = conv_10to8(src_uv[y * src_uv_pitch + x]);
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_p010le_nv12(int width, int height,
                                    uchar  *dst_y,  int dst_y_pitch,  const ushort  *src_y,  int src_y_pitch,
                                    uchar2 *dst_uv, int dst_uv_pitch, const ushort2 *src_uv, int src_uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_10to8(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);
        FIX_PITCH(src_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_10to8(src_uv[y * src_uv_pitch + x]);
    }
}

__global__ void Convert_p010le_yuv444p(int width, int height,
                                       uchar *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                       uchar *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                       uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        FIX_PITCH(src_uv);

        dst_y[y * dst_y_pitch + x] = conv_10to8(src_y[y * src_y_pitch + x]);

        uchar2 uv = conv_10to8(conv_420to444(src_uv, width, height, src_uv_pitch, x, y));
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_p010le_p016le(int width, int height,
                                      ushort  *dst_y,  int dst_y_pitch,  const ushort  *src_y,  int src_y_pitch,
                                      ushort2 *dst_uv, int dst_uv_pitch, const ushort2 *src_uv, int src_uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_10to16(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);
        FIX_PITCH(src_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_10to16(src_uv[y * src_uv_pitch + x]);
    }
}

__global__ void Convert_p010le_yuv444p16le(int width, int height,
                                           ushort *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                           ushort *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                           ushort *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(dst_u);
        FIX_PITCH(dst_v);
        FIX_PITCH(src_y);
        FIX_PITCH(src_uv);

        dst_y[y * dst_y_pitch + x] = conv_10to16(src_y[y * src_y_pitch + x]);

        ushort2 uv = conv_10to16(conv_420to444(src_uv, width, height, src_uv_pitch, x, y));
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

}

// p016le->X
extern "C" {

__global__ void Convert_p016le_yuv420p(int width, int height,
                                       uchar *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                       uchar *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                       uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(src_uv);

        uchar2 uv = conv_16to8(src_uv[y * src_uv_pitch + x]);
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_p016le_nv12(int width, int height,
                                    uchar  *dst_y,  int dst_y_pitch,  const ushort  *src_y,  int src_y_pitch,
                                    uchar2 *dst_uv, int dst_uv_pitch, const ushort2 *src_uv, int src_uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);
        FIX_PITCH(src_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_16to8(src_uv[y * src_uv_pitch + x]);
    }
}

__global__ void Convert_p016le_yuv444p(int width, int height,
                                       uchar *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                       uchar *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                       uchar *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        FIX_PITCH(src_uv);

        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);

        uchar2 uv = conv_16to8(conv_420to444(src_uv, width, height, src_uv_pitch, x, y));
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

__global__ void Convert_p016le_p010le(int width, int height,
                                      ushort  *dst_y,  int dst_y_pitch,  const ushort  *src_y,  int src_y_pitch,
                                      ushort2 *dst_uv, int dst_uv_pitch, const ushort2 *src_uv, int src_uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_16to10(src_y[y * src_y_pitch + x]);
    }

    if (x < width / 2 && y < height / 2) {
        FIX_PITCH(dst_uv);
        FIX_PITCH(src_uv);

        dst_uv[y * dst_uv_pitch + x] = conv_16to10(src_uv[y * src_uv_pitch + x]);
    }
}

__global__ void Convert_p016le_yuv444p16le(int width, int height,
                                           ushort *dst_y, int dst_y_pitch, const ushort  *src_y,  int src_y_pitch,
                                           ushort *dst_u, int dst_u_pitch, const ushort2 *src_uv, int src_uv_pitch,
                                           ushort *dst_v, int dst_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(dst_y);
        FIX_PITCH(dst_u);
        FIX_PITCH(dst_v);
        FIX_PITCH(src_y);
        FIX_PITCH(src_uv);

        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];

        ushort2 uv = conv_420to444(src_uv, width, height, src_uv_pitch, x, y);
        dst_u[y * dst_u_pitch + x] = uv.x;
        dst_v[y * dst_v_pitch + x] = uv.y;
    }
}

}

// yuv444p16le->X
extern "C" {

__global__ void Convert_yuv444p16le_yuv420p(int width, int height,
                                            uchar *dst_y, int dst_y_pitch, const ushort *src_y, int src_y_pitch,
                                            uchar *dst_u, int dst_u_pitch, const ushort *src_u, int src_u_pitch,
                                            uchar *dst_v, int dst_v_pitch, const ushort *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);

        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(src_u);
            FIX_PITCH(src_v);

            dst_u[y2 * dst_u_pitch + x2] = conv_16to8(conv_444to420(src_u, src_u_pitch, x, y));
            dst_v[y2 * dst_v_pitch + x2] = conv_16to8(conv_444to420(src_v, src_v_pitch, x, y));
        }
    }
}

__global__ void Convert_yuv444p16le_nv12(int width, int height,
                                         uchar  *dst_y , int dst_y_pitch,  const ushort *src_y, int src_y_pitch,
                                         uchar2 *dst_uv, int dst_uv_pitch, const ushort *src_u, int src_u_pitch,
                                         uchar2 *unused, int unused_pitch, const ushort *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(src_u);
            FIX_PITCH(src_v);
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_uchar2(
                conv_16to8(conv_444to420(src_u, src_u_pitch, x, y)),
                conv_16to8(conv_444to420(src_v, src_v_pitch, x, y))
            );
        }
    }
}

__global__ void Convert_yuv444p16le_yuv444p(int width, int height,
                                            uchar *dst_y, int dst_y_pitch, const ushort *src_y, int src_y_pitch,
                                            uchar *dst_u, int dst_u_pitch, const ushort *src_u, int src_u_pitch,
                                            uchar *dst_v, int dst_v_pitch, const ushort *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        FIX_PITCH(src_u);
        FIX_PITCH(src_v);

        dst_y[y * dst_y_pitch + x] = conv_16to8(src_y[y * src_y_pitch + x]);
        dst_u[y * dst_u_pitch + x] = conv_16to8(src_u[y * src_u_pitch + x]);
        dst_v[y * dst_v_pitch + x] = conv_16to8(src_v[y * src_v_pitch + x]);
    }
}

__global__ void Convert_yuv444p16le_p010le(int width, int height,
                                           ushort  *dst_y,  int dst_y_pitch,  const ushort *src_y, int src_y_pitch,
                                           ushort2 *dst_uv, int dst_uv_pitch, const ushort *src_u, int src_u_pitch,
                                           ushort2 *unused, int unused_pitch, const ushort *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        FIX_PITCH(dst_y);
        dst_y[y * dst_y_pitch + x] = conv_16to10(src_y[y * src_y_pitch + x]);

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(src_u);
            FIX_PITCH(src_v);
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_ushort2(
                conv_16to10(conv_444to420(src_u, src_u_pitch, x, y)),
                conv_16to10(conv_444to420(src_v, src_v_pitch, x, y))
            );
        }
    }
}

__global__ void Convert_yuv444p16le_p016le(int width, int height,
                                           ushort  *dst_y,  int dst_y_pitch,  const ushort *src_y, int src_y_pitch,
                                           ushort2 *dst_uv, int dst_uv_pitch, const ushort *src_u, int src_u_pitch,
                                           ushort2 *unused, int unused_pitch, const ushort *src_v, int src_v_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        FIX_PITCH(src_y);
        FIX_PITCH(dst_y);
        dst_y[y * dst_y_pitch + x] = src_y[y * src_y_pitch + x];

        if ((x & 1) == 0 && (y & 1) == 0) {
            int x2 = x / 2;
            int y2 = y / 2;
            FIX_PITCH(src_u);
            FIX_PITCH(src_v);
            FIX_PITCH(dst_uv);

            dst_uv[y2 * dst_uv_pitch + x2] = make_ushort2(
                conv_444to420(src_u, src_u_pitch, x, y),
                conv_444to420(src_v, src_v_pitch, x, y)
            );
        }
    }
}

}
