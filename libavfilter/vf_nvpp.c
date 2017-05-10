/*
 * Nvidia CUVID video processor
 * Copyright (c) 2017 Timo Rothenpieler <timo@rothenpieler.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "compat/cuda/dynlink_loader.h"

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "scale.h"
#include "video.h"

typedef struct NvppContext {
    const AVClass *class;

    CUvideodecoder cudecoder;

    AVBufferRef *hwdevice;
    AVBufferRef *hwframe;

    CudaFunctions *cudl;
    CuvidFunctions *cvdl;

    char *w_expr;
    char *h_expr;
} NvppContext;

static int check_cu(AVFilterContext *avctx, CUresult err, const char *func)
{
    NvppContext *ctx = avctx->priv;
    const char *err_name;
    const char *err_string;

    av_log(avctx, AV_LOG_TRACE, "Calling %s\n", func);

    if (err == CUDA_SUCCESS)
        return 0;

    ctx->cudl->cuGetErrorName(err, &err_name);
    ctx->cudl->cuGetErrorString(err, &err_string);

    av_log(avctx, AV_LOG_ERROR, "%s failed", func);
    if (err_name && err_string)
        av_log(avctx, AV_LOG_ERROR, " -> %s: %s", err_name, err_string);
    av_log(avctx, AV_LOG_ERROR, "\n");

    return AVERROR_EXTERNAL;
}

#define CHECK_CU(x) check_cu(avctx, (x), #x)

static int sw_format_is_supported(enum AVPixelFormat fmt)
{
    if (fmt == AV_PIX_FMT_NV12) //TODO: more formats!
        return 1;

    return 0;
}

static int nvpp_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *avctx = inlink->dst;
    NvppContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFrame *out = NULL;
    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;
    AVCUDADeviceContext *device_hwctx = device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CUVIDPICPARAMS pic_params;
    CUVIDPROCPARAMS vpp_params;
    CUdeviceptr mapped_frame = 0;
    unsigned int mapped_pitch = 0;
    int eret, ret = 0;

    ret = CHECK_CU(ctx->cudl->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto error;
    }

    ret = av_hwframe_get_buffer(ctx->hwframe, out, 0);
    if (ret < 0)
        goto error;

    memset(&pic_params, 0, sizeof(pic_params));

    pic_params.CurrPicIdx = 0;
    pic_params.PicWidthInMbs = (in->width + 15) / 16;
    pic_params.FrameHeightInMbs = (in->height + 15) / 16;

    ret = CHECK_CU(ctx->cvdl->cuvidDecodePicture(ctx->cudecoder, &pic_params));
    if (ret < 0)
        goto error;

    memset(&vpp_params, 0, sizeof(vpp_params));

    vpp_params.progressive_frame = 1;

    vpp_params.raw_input_dptr = (CUdeviceptr)in->data[0];
    vpp_params.raw_input_pitch = in->linesize[0];
    vpp_params.raw_input_format = cudaVideoCodec_NV12;

    vpp_params.raw_output_dptr = (CUdeviceptr)out->data[0];
    vpp_params.raw_output_pitch = out->linesize[0];


    ret = CHECK_CU(ctx->cvdl->cuvidMapVideoFrame(ctx->cudecoder, pic_params.CurrPicIdx, &mapped_frame, &mapped_pitch, &vpp_params));
    if (ret < 0)
        goto error;

    ret = CHECK_CU(ctx->cvdl->cuvidUnmapVideoFrame(ctx->cudecoder, mapped_frame));
    if (ret < 0)
        goto error;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto error;

    av_frame_free(&in);
    ret = ff_filter_frame(outlink, out);
    goto end;

error:
    av_frame_free(&in);
    av_frame_free(&out);
end:
    eret = CHECK_CU(ctx->cudl->cuCtxPopCurrent(&dummy));
    if (eret < 0)
        return eret;
    return ret;
}

static void nvpp_uninit(AVFilterContext *avctx)
{
    NvppContext *ctx = avctx->priv;

    if (ctx->cudecoder)
        ctx->cvdl->cuvidDestroyDecoder(ctx->cudecoder);

    ctx->cudl = NULL;

    av_buffer_unref(&ctx->hwframe);
    av_buffer_unref(&ctx->hwdevice);

    cuvid_free_functions(&ctx->cvdl);
}

static int nvpp_config_props(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    NvppContext *ctx = avctx->priv;
    AVHWFramesContext *in_hwframe_ctx;
    AVHWDeviceContext *device_ctx;
    AVCUDADeviceContext *device_hwctx;
    AVHWFramesContext *out_hwframe_ctx;

    CUcontext dummy, cuda_ctx;
    CUVIDDECODECREATEINFO cuinfo;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;

    int w, h;
    int ret;

    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw_frames_ctx provided on input\n");
        return AVERROR(EINVAL);
    }

    ret = ff_scale_eval_dimensions(avctx, ctx->w_expr, ctx->h_expr,
                                   inlink, outlink, &w, &h);
    if (ret < 0)
        return ret;

    outlink->w = w;
    outlink->h = h;

    in_hwframe_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;

    ctx->hwdevice = av_buffer_ref(in_hwframe_ctx->device_ref);
    if (!ctx->hwdevice)
        return AVERROR(ENOMEM);

    device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;
    device_hwctx = device_ctx->hwctx;

    ctx->cudl = device_hwctx->internal->cuda_dl;
    cuda_ctx = device_hwctx->cuda_ctx;

    in_format = in_hwframe_ctx->sw_format;
    out_format = in_format; //TODO: Maybe make configurable?

    if (!sw_format_is_supported(in_format)) {
        av_log(avctx, AV_LOG_ERROR, "input pixel format is not supported.\n");
        return AVERROR(EINVAL);
    }

    if (!sw_format_is_supported(out_format)) {
        av_log(avctx, AV_LOG_ERROR, "output pixel format is not supported.\n");
        return AVERROR(EINVAL);
    }

    ctx->hwframe = av_hwframe_ctx_alloc(ctx->hwdevice);
    if (!ctx->hwframe)
        return AVERROR(ENOMEM);

    out_hwframe_ctx = (AVHWFramesContext*)ctx->hwframe->data;

    out_hwframe_ctx->format = AV_PIX_FMT_CUDA;
    out_hwframe_ctx->sw_format = out_format;
    out_hwframe_ctx->width = outlink->w;
    out_hwframe_ctx->height = outlink->h;

    ret = av_hwframe_ctx_init(ctx->hwframe);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "av_hwframe_ctx_init failed\n");
        return ret;
    }

    memset(&cuinfo, 0, sizeof(cuinfo));

    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave; //TODO: add deinterlacing support
    cuinfo.ulNumDecodeSurfaces = 1;
    cuinfo.ulNumOutputSurfaces = 1;

    if (in_format == AV_PIX_FMT_NV12) {
        cuinfo.CodecType = cudaVideoCodec_NV12;
        cuinfo.ChromaFormat = cudaVideoChromaFormat_420;
        cuinfo.bitDepthMinus8 = 0;
    } else {
        return AVERROR_BUG;
    }

    cuinfo.ulWidth = inlink->w;
    cuinfo.ulHeight = inlink->h;

    cuinfo.ulTargetWidth = outlink->w;
    cuinfo.ulTargetHeight = outlink->h;

    cuinfo.display_area.left = 0; //TODO: cropping
    cuinfo.display_area.top = 0;
    cuinfo.display_area.right = outlink->w;
    cuinfo.display_area.bottom = outlink->h;

    cuinfo.target_rect.left = 0;
    cuinfo.target_rect.top = 0;
    cuinfo.target_rect.right = outlink->w;
    cuinfo.target_rect.bottom = outlink->h;

    ret = CHECK_CU(ctx->cudl->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(ctx->cvdl->cuvidCreateDecoder(&ctx->cudecoder, &cuinfo));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(ctx->cudl->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    outlink->hw_frames_ctx = av_buffer_ref(ctx->hwframe);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0; //TODO cleanup on error
}

static int nvpp_init(AVFilterContext *avctx)
{
    NvppContext *ctx = avctx->priv;
    int ret;

    ret = cuvid_load_functions(&ctx->cvdl, avctx);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "failed loading nvcuvid\n");
        return ret;
    }

    return 0;
}

static int nvpp_query_formats(AVFilterContext *avctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(avctx, pix_fmts);
}

#define OFFSET(x) offsetof(NvppContext, x)
#define VF (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "w", "Output video width",  OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = "iw" }, .flags = VF },
    { "h", "Output video height", OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = "ih" }, .flags = VF },
    { NULL }
};

static const AVClass nvpp_class = {
    .class_name = "nvpp",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT
};

static const AVFilterPad nvpp_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = nvpp_filter_frame,
    },
    { NULL }
};

static const AVFilterPad nvpp_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = nvpp_config_props,
    },
    { NULL }
};

AVFilter ff_vf_nvpp = {
    .name      = "nvpp",
    .description = NULL_IF_CONFIG_SMALL("Nvidia CUVID video processor"),

    .init          = nvpp_init,
    .uninit        = nvpp_uninit,
    .query_formats = nvpp_query_formats,

    .priv_size  = sizeof(NvppContext),
    .priv_class = &nvpp_class,

    .inputs    = nvpp_inputs,
    .outputs   = nvpp_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};
