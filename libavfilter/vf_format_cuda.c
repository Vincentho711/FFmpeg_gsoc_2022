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

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

#include "cuda/load_helper.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
    AV_PIX_FMT_YUV444P16,
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct CUDAFormatContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;

    AVFrame *tmp_frame;
    int passthrough;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;

    CUcontext   cu_ctx;
    CUmodule    cu_module;
    CUfunction  cu_func_convert;
    CUstream    cu_stream;
} CUDAFormatContext;

static av_cold int cudaformat_init(AVFilterContext *ctx)
{
    CUDAFormatContext *s = ctx->priv;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudaformat_uninit(AVFilterContext *ctx)
{
    CUDAFormatContext *s = ctx->priv;

    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int cudaformat_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);
    if (!pix_fmts)
        return AVERROR(ENOMEM);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_hwframe_ctx(CUDAFormatContext *s, AVBufferRef *device_ctx, int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width     = FFALIGN(width,  32);
    out_ctx->height    = FFALIGN(height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = width;
    s->frame->height = height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, int width, int height)
{
    CUDAFormatContext *s = ctx->priv;

    AVHWFramesContext *in_frames_ctx;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    int ret;

    /* check that we have a hw context */
    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (s->format == AV_PIX_FMT_NONE) ? in_format : s->format;

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(out_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    s->in_fmt = in_format;
    s->out_fmt = out_format;

    if (s->passthrough && in_format == out_format) {
        s->frames_ctx = av_buffer_ref(ctx->inputs[0]->hw_frames_ctx);
        if (!s->frames_ctx)
            return AVERROR(ENOMEM);
    } else {
        s->passthrough = 0;

        ret = init_hwframe_ctx(s, in_frames_ctx->device_ref, width, height);
        if (ret < 0)
            return ret;
    }

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudaformat_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAFormatContext *s = ctx->priv;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;
    const char *in_fmt_name, *out_fmt_name;
    char buf[64];
    int ret;

    extern const unsigned char ff_vf_format_cuda_ptx_data[];
    extern const unsigned int ff_vf_format_cuda_ptx_len;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, device_hwctx, &s->cu_module, ff_vf_format_cuda_ptx_data, ff_vf_format_cuda_ptx_len);
    if (ret < 0)
        goto fail;

    in_fmt_name = av_get_pix_fmt_name(s->in_fmt);
    out_fmt_name = av_get_pix_fmt_name(s->out_fmt);
    snprintf(buf, sizeof(buf), "Convert_%s_%s", in_fmt_name, out_fmt_name);

    if (s->in_fmt != s->out_fmt) {
        av_log(ctx, AV_LOG_DEBUG, "Loading conversion kernel: %s\n", buf);
        ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_convert, s->cu_module, buf));
        if (ret < 0)
            goto fail;
    }

    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    av_log(ctx, AV_LOG_VERBOSE, "%s -> %s%s\n",
           in_fmt_name, out_fmt_name, s->passthrough ? " (passthrough)" : "");

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int call_conversion_kernel(AVFilterContext *ctx,
                                  AVFrame *out, AVFrame *in)
{
    CUDAFormatContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int ret;

    CUdeviceptr dst_devptrs[] = { (CUdeviceptr)out->data[0], (CUdeviceptr)out->data[1],
                                  (CUdeviceptr)out->data[2], (CUdeviceptr)out->data[3] };
    CUdeviceptr src_devptrs[] = { (CUdeviceptr)in->data[0],  (CUdeviceptr)in->data[1],
                                  (CUdeviceptr)in->data[2],  (CUdeviceptr)in->data[3] };
    void *args[] = { &in->width, &in->height,
                     &dst_devptrs[0], &out->linesize[0], &src_devptrs[0], &in->linesize[0],
                     &dst_devptrs[1], &out->linesize[1], &src_devptrs[1], &in->linesize[1],
                     &dst_devptrs[2], &out->linesize[2], &src_devptrs[2], &in->linesize[2],
                     &dst_devptrs[3], &out->linesize[3], &src_devptrs[3], &in->linesize[3] };

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_convert,
                                      DIV_UP(out->width, BLOCKX), DIV_UP(out->height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args, NULL));
    if (ret < 0)
        return ret;

    return 0;
}

static int cudaformat_convert(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAFormatContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *src = in;
    int ret;

    if (s->in_fmt != s->out_fmt)
        ret = call_conversion_kernel(ctx, s->frame, src);
    else
        ret = av_hwframe_transfer_data(s->frame, src, 0);
    if (ret < 0)
        return ret;

    src = s->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->tmp_frame);

    s->frame->width  = outlink->w;
    s->frame->height = outlink->h;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudaformat_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    CUDAFormatContext        *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    if (s->passthrough)
        return ff_filter_frame(outlink, in);

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = cudaformat_convert(ctx, out, in);
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static AVFrame *cudaformat_get_video_buffer(AVFilterLink *inlink, int w, int h)
{
    CUDAFormatContext *s = inlink->dst->priv;

    return s->passthrough ?
        ff_null_get_video_buffer   (inlink, w, h) :
        ff_default_get_video_buffer(inlink, w, h);
}

#define OFFSET(x) offsetof(CUDAFormatContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "format", "Output pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, AV_PIX_FMT_NONE, INT_MAX, FLAGS },
    { "passthrough", "Do not process frames at all if parameters match", OFFSET(passthrough), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL },
};

static const AVClass cudaformat_class = {
    .class_name = "cudaformat",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudaformat_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudaformat_filter_frame,
        .get_video_buffer = cudaformat_get_video_buffer,
    },
    { NULL }
};

static const AVFilterPad cudaformat_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudaformat_config_props,
    },
    { NULL }
};

const AVFilter ff_vf_format_cuda = {
    .name      = "format_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated video format conversion"),

    .init          = cudaformat_init,
    .uninit        = cudaformat_uninit,
    .query_formats = cudaformat_query_formats,

    .priv_size = sizeof(CUDAFormatContext),
    .priv_class = &cudaformat_class,

    .inputs    = cudaformat_inputs,
    .outputs   = cudaformat_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
