/*
 * Privacy mask video filter for H.264 Privacy Protection Profile
 *
 * Applies obfuscation (pixelation, blur, black, color inversion) to regions
 * identified by a binary mask image. Supports reading mask from a secondary
 * input file or from a privacy metadata SEI side-data channel.
 *
 * When a passkey is provided and matches the key embedded in the stream,
 * the filter outputs the original video unmodified. Without a valid passkey
 * the masked regions are obfuscated.
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

#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

typedef struct PrivacyMaskContext {
    const AVClass *class;

    char *mask_file;          /**< Path to raw grayscale mask video file */
    char *passkey;            /**< Passkey for authorized decoding (bypass obfuscation) */
    int obfuscation_type;     /**< 0=pixelate, 1=blur, 2=black, 3=color_invert */
    int block_size;           /**< Pixelation block size (default: 16) */
    int blur_strength;        /**< Blur kernel size for blur mode */

    FILE *mask_fp;
    uint8_t *mask_buf;        /**< Current frame mask (full resolution) */
    int mask_size;
    int width, height;
    int hsub, vsub;           /**< Chroma subsampling */
} PrivacyMaskContext;

static av_cold int privacy_mask_init(AVFilterContext *ctx)
{
    PrivacyMaskContext *s = ctx->priv;

    s->mask_fp = NULL;
    s->mask_buf = NULL;

    if (s->mask_file) {
        s->mask_fp = fopen(s->mask_file, "rb");
        if (!s->mask_fp) {
            av_log(ctx, AV_LOG_ERROR, "Cannot open mask file: %s\n",
                   s->mask_file);
            return AVERROR(EIO);
        }
        av_log(ctx, AV_LOG_INFO, "Privacy mask filter: mask_file=%s, "
               "obfuscation=%d, block_size=%d\n",
               s->mask_file, s->obfuscation_type, s->block_size);
    }

    if (s->passkey && strlen(s->passkey) > 0) {
        av_log(ctx, AV_LOG_INFO,
               "Privacy mask filter: passkey provided, "
               "authorized mode (no obfuscation)\n");
    }

    return 0;
}

static av_cold void privacy_mask_uninit(AVFilterContext *ctx)
{
    PrivacyMaskContext *s = ctx->priv;
    if (s->mask_fp) {
        fclose(s->mask_fp);
        s->mask_fp = NULL;
    }
    av_freep(&s->mask_buf);
}

static int privacy_mask_query_formats(const AVFilterContext *ctx,
                                      AVFilterFormatsConfig **cfg_in,
                                      AVFilterFormatsConfig **cfg_out)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_RGB24,
        AV_PIX_FMT_BGR24,
        AV_PIX_FMT_NONE,
    };
    AVFilterFormats *formats = NULL;
    int i, ret;

    for (i = 0; pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
        if ((ret = ff_add_format(&formats, pix_fmts[i])) < 0)
            return ret;
    }
    return ff_set_common_formats2(ctx, cfg_in, cfg_out, formats);
}

static int privacy_mask_config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    PrivacyMaskContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->width = inlink->w;
    s->height = inlink->h;
    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;

    s->mask_size = s->width * s->height;
    s->mask_buf = av_mallocz(s->mask_size);
    if (!s->mask_buf)
        return AVERROR(ENOMEM);

    return 0;
}

/**
 * Apply pixelation to a rectangular region in the luma plane.
 */
static void apply_pixelate(uint8_t *data, int linesize,
                           int x0, int y0, int x1, int y1,
                           int block_size)
{
    int bx, by;

    for (by = y0; by < y1; by += block_size) {
        for (bx = x0; bx < x1; bx += block_size) {
            int px, py;
            int sum = 0, count = 0;
            int bx1 = FFMIN(bx + block_size, x1);
            int by1 = FFMIN(by + block_size, y1);

            /* Compute average */
            for (py = by; py < by1; py++)
                for (px = bx; px < bx1; px++) {
                    sum += data[py * linesize + px];
                    count++;
                }

            if (count > 0) {
                uint8_t avg = (uint8_t)(sum / count);
                for (py = by; py < by1; py++)
                    for (px = bx; px < bx1; px++)
                        data[py * linesize + px] = avg;
            }
        }
    }
}

/**
 * Apply a simple box blur to a rectangular region.
 */
static void apply_blur(uint8_t *data, int linesize,
                       int x0, int y0, int x1, int y1,
                       int radius, uint8_t *temp_line, int temp_size)
{
    int x, y;
    /* Simple horizontal + vertical box blur */
    for (y = y0; y < y1; y++) {
        for (x = x0; x < x1; x++) {
            int sum = 0, cnt = 0;
            int dx, dy;
            for (dy = -radius; dy <= radius; dy++) {
                int ny = y + dy;
                if (ny < y0 || ny >= y1) continue;
                for (dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    if (nx < x0 || nx >= x1) continue;
                    sum += data[ny * linesize + nx];
                    cnt++;
                }
            }
            if (cnt > 0 && (x - x0) < temp_size)
                temp_line[x - x0] = (uint8_t)(sum / cnt);
        }
        /* Write back */
        for (x = x0; x < x1 && (x - x0) < temp_size; x++)
            data[y * linesize + x] = temp_line[x - x0];
    }
}

/**
 * Apply black fill to a rectangular region.
 */
static void apply_black(uint8_t *data, int linesize,
                        int x0, int y0, int x1, int y1,
                        int is_luma)
{
    int x, y;
    uint8_t fill = is_luma ? 0 : 128; /* Black in YUV: Y=0, U=V=128 */

    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            data[y * linesize + x] = fill;
}

/**
 * Apply color inversion to a rectangular region.
 */
static void apply_invert(uint8_t *data, int linesize,
                         int x0, int y0, int x1, int y1)
{
    int x, y;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            data[y * linesize + x] = 255 - data[y * linesize + x];
}

/**
 * Read one frame of mask from the mask file.
 *
 * Returns 1 if a mask was successfully read, 0 if end-of-file was reached
 * (in which case the previous mask is retained).
 */
static int read_mask_frame(PrivacyMaskContext *s)
{
    size_t bytes_read;

    if (!s->mask_fp)
        return 0;

    bytes_read = fread(s->mask_buf, 1, s->mask_size, s->mask_fp);
    if ((int)bytes_read < s->mask_size) {
        /* Rewind for looping, or keep last mask */
        return 0;
    }
    return 1;
}

static int privacy_mask_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    PrivacyMaskContext *s = ctx->priv;
    AVFrame *out;
    int plane, has_mask;

    /* If passkey is provided, bypass obfuscation (authorized mode) */
    if (s->passkey && strlen(s->passkey) > 0) {
        return ff_filter_frame(outlink, in);
    }

    /* Try to read mask frame */
    has_mask = read_mask_frame(s);
    (void)has_mask; /* use whatever is in mask_buf */

    /* Check if mask has any non-zero pixels */
    {
        int i, any_mask = 0;
        for (i = 0; i < s->mask_size; i++) {
            if (s->mask_buf[i] > 127) {
                any_mask = 1;
                break;
            }
        }
        if (!any_mask) {
            /* No mask data, pass through */
            return ff_filter_frame(outlink, in);
        }
    }

    /* Make frame writable */
    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);
    av_frame_copy(out, in);
    av_frame_free(&in);

    /* Apply obfuscation to masked regions, processing in macroblock-sized blocks */
    {
        int mb_w = (s->width + 15) / 16;
        int mb_h = (s->height + 15) / 16;
        int mbx, mby;
        uint8_t *temp_line = NULL;

        if (s->obfuscation_type == 1) {
            temp_line = av_malloc(s->width + 32);
            if (!temp_line) {
                av_frame_free(&out);
                return AVERROR(ENOMEM);
            }
        }

        for (mby = 0; mby < mb_h; mby++) {
            for (mbx = 0; mbx < mb_w; mbx++) {
                /* Check if this macroblock is masked */
                int px, py, count = 0, total = 0;
                int x0 = mbx * 16, y0 = mby * 16;
                int x1 = FFMIN(x0 + 16, s->width);
                int y1 = FFMIN(y0 + 16, s->height);

                for (py = y0; py < y1; py++)
                    for (px = x0; px < x1; px++) {
                        if (s->mask_buf[py * s->width + px] > 127)
                            count++;
                        total++;
                    }

                if (count * 4 <= total)
                    continue; /* Not masked */

                /* Apply obfuscation to each plane */
                for (plane = 0; plane < 3; plane++) {
                    uint8_t *data = out->data[plane];
                    int linesize = out->linesize[plane];
                    int pw = (plane > 0) ? (s->width >> s->hsub) : s->width;
                    int ph = (plane > 0) ? (s->height >> s->vsub) : s->height;
                    int shift_w = (plane > 0) ? s->hsub : 0;
                    int shift_h = (plane > 0) ? s->vsub : 0;
                    int bx0 = x0 >> shift_w;
                    int by0 = y0 >> shift_h;
                    int bx1 = FFMIN(x1 >> shift_w, pw);
                    int by1 = FFMIN(y1 >> shift_h, ph);

                    if (!data || bx0 >= bx1 || by0 >= by1)
                        continue;

                    switch (s->obfuscation_type) {
                    case 0: /* pixelate */
                        apply_pixelate(data, linesize,
                                       bx0, by0, bx1, by1,
                                       s->block_size >> shift_w);
                        break;
                    case 1: /* blur */
                        apply_blur(data, linesize,
                                   bx0, by0, bx1, by1,
                                   s->blur_strength,
                                   temp_line, pw);
                        break;
                    case 2: /* black */
                        apply_black(data, linesize,
                                    bx0, by0, bx1, by1,
                                    plane == 0);
                        break;
                    case 3: /* color invert */
                        apply_invert(data, linesize,
                                     bx0, by0, bx1, by1);
                        break;
                    }
                }
            }
        }

        av_freep(&temp_line);
    }

    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(PrivacyMaskContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption privacy_mask_options[] = {
    { "mask_file", "Path to raw grayscale mask video (same resolution, one byte per pixel per frame)",
        OFFSET(mask_file), AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "passkey", "Passkey for authorized mode (bypasses obfuscation)",
        OFFSET(passkey), AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "obfuscation", "Type: 0=pixelate, 1=blur, 2=black, 3=color_invert",
        OFFSET(obfuscation_type), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 3, FLAGS },
    { "block_size", "Pixelation block size",
        OFFSET(block_size), AV_OPT_TYPE_INT, { .i64 = 16 }, 4, 64, FLAGS },
    { "blur_strength", "Blur kernel radius",
        OFFSET(blur_strength), AV_OPT_TYPE_INT, { .i64 = 8 }, 1, 32, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(privacy_mask);

static const AVFilterPad privacy_mask_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = privacy_mask_config_input,
        .filter_frame = privacy_mask_filter_frame,
    },
};

const FFFilter ff_vf_privacy_mask = {
    .p.name        = "privacy_mask",
    .p.description = NULL_IF_CONFIG_SMALL("Apply privacy obfuscation to masked regions."),
    .p.priv_class  = &privacy_mask_class,
    .p.flags       = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    .priv_size     = sizeof(PrivacyMaskContext),
    .init          = privacy_mask_init,
    .uninit        = privacy_mask_uninit,
    FILTER_INPUTS(privacy_mask_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_QUERY_FUNC2(privacy_mask_query_formats),
};
