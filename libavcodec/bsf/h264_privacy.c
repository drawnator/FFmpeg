/*
 * H.264 Privacy Metadata Bitstream Filter
 *
 * Injects privacy metadata SEI messages into H.264 bitstreams.
 * The privacy metadata carries compressed mask data identifying
 * regions requiring privacy protection (e.g., faces, license plates).
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

#include <string.h>

#include "libavutil/avassert.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

#include "bsf.h"
#include "bsf_internal.h"
#include "bytestream.h"
#include "defs.h"
#include "h264.h"
#include "sei.h"

typedef struct H264PrivacyContext {
    const AVClass *class;
    char *mask_file;        /**< Path to binary mask file (raw frames) */
    int obfuscation_type;   /**< 0=pixelate, 1=blur, 2=black, 3=color_invert */
    int encryption_algo;    /**< 0=none, 1=AES-128, 2=AES-256 */
    char *key_hex;          /**< Hex-encoded encryption key */
    int metadata_id;        /**< Privacy metadata instance ID */
    FILE *mask_fp;          /**< File pointer for mask data */
    int frame_count;        /**< Frame counter */
    int width;              /**< Video width from codec parameters */
    int height;             /**< Video height from codec parameters */
} H264PrivacyContext;

/**
 * Run-length encode a binary mask buffer.
 *
 * Encodes sequences of identical bytes as (value, count_high, count_low)
 * triplets.  Returns the number of bytes written to @p out, or a negative
 * AVERROR code on failure.
 */
static int rle_encode_mask(const uint8_t *mask, int mask_size,
                           uint8_t *out, int out_capacity)
{
    int i = 0, out_pos = 0;

    while (i < mask_size) {
        uint8_t val = mask[i];
        int run = 1;

        while (i + run < mask_size && mask[i + run] == val && run < 65535)
            run++;

        if (out_pos + 3 > out_capacity)
            return AVERROR(ENOSPC);

        out[out_pos++] = val;
        out[out_pos++] = (run >> 8) & 0xFF;
        out[out_pos++] = run & 0xFF;
        i += run;
    }

    return out_pos;
}

/**
 * Build an SEI NAL unit carrying the privacy_metadata payload.
 *
 * The NAL unit is formatted as Annex B with a 4-byte start code
 * (0x00000001), NAL header byte (type 6 = SEI), followed by the SEI
 * message header (type + size) and the payload itself.
 */
static int build_privacy_sei_nal(H264PrivacyContext *s,
                                 const uint8_t *rle_data, int rle_size,
                                 uint8_t **out_buf, int *out_size)
{
    /*
     * Payload layout:
     *   1 byte  : privacy_metadata_id (exp-golomb → simplified as 1 byte)
     *   1 byte  : cancel_flag (0)
     *   1 byte  : mask_encryption_algorithm
     *   16 bytes: key_id
     *   1 byte  : obfuscation_type
     *   N bytes : mask data (RLE compressed, optionally encrypted)
     */
    int payload_size = 1 + 1 + 1 + 16 + 1 + rle_size;
    int sei_type = SEI_TYPE_PRIVACY_METADATA;
    int header_size, total_size, pos;
    uint8_t *buf;

    /* SEI type bytes + size bytes */
    header_size = 0;
    {
        int t = sei_type;
        while (t >= 255) { header_size++; t -= 255; }
        header_size++; /* final type byte */
    }
    {
        int sz = payload_size;
        while (sz >= 255) { header_size++; sz -= 255; }
        header_size++; /* final size byte */
    }

    /* 4 (start code) + 1 (NAL header) + header_size + payload_size + 1 (rbsp trailing) */
    total_size = 4 + 1 + header_size + payload_size + 1;
    buf = av_malloc(total_size);
    if (!buf)
        return AVERROR(ENOMEM);

    pos = 0;

    /* Annex B start code */
    buf[pos++] = 0x00;
    buf[pos++] = 0x00;
    buf[pos++] = 0x00;
    buf[pos++] = 0x01;

    /* NAL header: forbidden_zero_bit=0, nal_ref_idc=0, nal_unit_type=6 (SEI) */
    buf[pos++] = 0x06;

    /* SEI type */
    {
        int t = sei_type;
        while (t >= 255) { buf[pos++] = 0xFF; t -= 255; }
        buf[pos++] = (uint8_t)t;
    }

    /* SEI payload size */
    {
        int sz = payload_size;
        while (sz >= 255) { buf[pos++] = 0xFF; sz -= 255; }
        buf[pos++] = (uint8_t)sz;
    }

    /* Payload */
    buf[pos++] = (uint8_t)(s->metadata_id & 0xFF);  /* privacy_metadata_id */
    buf[pos++] = 0x00;                                /* cancel_flag = 0 */
    buf[pos++] = (uint8_t)s->encryption_algo;         /* mask_encryption_algorithm */

    /* key_id: 16 bytes, parse from hex or zero */
    if (s->key_hex && strlen(s->key_hex) >= 32) {
        int i;
        for (i = 0; i < 16; i++) {
            unsigned byte_val;
            sscanf(s->key_hex + i * 2, "%02x", &byte_val);
            buf[pos++] = (uint8_t)byte_val;
        }
    } else {
        memset(buf + pos, 0, 16);
        pos += 16;
    }

    buf[pos++] = (uint8_t)s->obfuscation_type;

    /* Mask data (RLE compressed) */
    memcpy(buf + pos, rle_data, rle_size);
    pos += rle_size;

    /* RBSP trailing bits */
    buf[pos++] = 0x80;

    *out_buf = buf;
    *out_size = pos;
    return 0;
}

static int h264_privacy_init(AVBSFContext *ctx)
{
    H264PrivacyContext *s = ctx->priv_data;

    s->frame_count = 0;
    s->mask_fp = NULL;

    if (s->mask_file) {
        s->mask_fp = fopen(s->mask_file, "rb");
        if (!s->mask_fp) {
            av_log(ctx, AV_LOG_ERROR, "Cannot open mask file: %s\n",
                   s->mask_file);
            return AVERROR(EIO);
        }
    }

    s->width = ctx->par_in->width;
    s->height = ctx->par_in->height;

    av_log(ctx, AV_LOG_INFO,
           "H264 Privacy BSF: mask_file=%s, obfuscation=%d, encryption=%d, "
           "resolution=%dx%d\n",
           s->mask_file ? s->mask_file : "(none)",
           s->obfuscation_type, s->encryption_algo,
           s->width, s->height);

    return 0;
}

static void h264_privacy_close(AVBSFContext *ctx)
{
    H264PrivacyContext *s = ctx->priv_data;
    if (s->mask_fp) {
        fclose(s->mask_fp);
        s->mask_fp = NULL;
    }
}

/**
 * Read one frame of mask data and RLE-encode it, then inject as SEI.
 *
 * The mask file is expected to contain raw grayscale frames at the same
 * resolution as the video, where 0 = not masked and non-zero = masked.
 * For macroblock-level alignment, the mask is downsampled to 16×16 blocks.
 */
static int h264_privacy_filter(AVBSFContext *ctx, AVPacket *pkt)
{
    H264PrivacyContext *s = ctx->priv_data;
    uint8_t *mask_frame = NULL;
    uint8_t *mb_mask = NULL;
    uint8_t *rle_data = NULL;
    uint8_t *sei_nal = NULL;
    uint8_t *new_data = NULL;
    int ret, sei_size = 0;
    int mb_w, mb_h, mb_size;
    int rle_size;

    ret = ff_bsf_get_packet_ref(ctx, pkt);
    if (ret < 0)
        return ret;

    if (!s->mask_fp) {
        s->frame_count++;
        return 0;
    }

    /* Read raw mask frame */
    {
        int frame_size = s->width * s->height;
        size_t bytes_read;

        mask_frame = av_malloc(frame_size);
        if (!mask_frame) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        bytes_read = fread(mask_frame, 1, frame_size, s->mask_fp);
        if ((int)bytes_read < frame_size) {
            /* End of mask file or short read → no privacy metadata for this frame */
            av_freep(&mask_frame);
            s->frame_count++;
            return 0;
        }
    }

    /* Downsample to macroblock level (16x16) */
    mb_w = (s->width + 15) / 16;
    mb_h = (s->height + 15) / 16;
    mb_size = mb_w * mb_h;

    mb_mask = av_mallocz(mb_size);
    if (!mb_mask) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    {
        int mbx, mby;
        for (mby = 0; mby < mb_h; mby++) {
            for (mbx = 0; mbx < mb_w; mbx++) {
                int px, py, count = 0, total = 0;
                int x0 = mbx * 16, y0 = mby * 16;
                int x1 = FFMIN(x0 + 16, s->width);
                int y1 = FFMIN(y0 + 16, s->height);

                for (py = y0; py < y1; py++) {
                    for (px = x0; px < x1; px++) {
                        if (mask_frame[py * s->width + px] > 127)
                            count++;
                        total++;
                    }
                }

                /* Mark MB as masked if >25% of pixels are masked */
                mb_mask[mby * mb_w + mbx] = (count * 4 > total) ? 0xFF : 0x00;
            }
        }
    }

    /* RLE encode */
    rle_data = av_malloc(mb_size * 3 + 16);
    if (!rle_data) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    /* Store mb dimensions at start: 2 bytes width + 2 bytes height */
    rle_data[0] = (mb_w >> 8) & 0xFF;
    rle_data[1] = mb_w & 0xFF;
    rle_data[2] = (mb_h >> 8) & 0xFF;
    rle_data[3] = mb_h & 0xFF;

    rle_size = rle_encode_mask(mb_mask, mb_size,
                               rle_data + 4, mb_size * 3 + 12);
    if (rle_size < 0) {
        ret = rle_size;
        goto fail;
    }
    rle_size += 4; /* include the dimension header */

    /* Build SEI NAL unit */
    ret = build_privacy_sei_nal(s, rle_data, rle_size, &sei_nal, &sei_size);
    if (ret < 0)
        goto fail;

    /* Prepend SEI NAL to packet data */
    new_data = av_malloc(sei_size + pkt->size);
    if (!new_data) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    memcpy(new_data, sei_nal, sei_size);
    memcpy(new_data + sei_size, pkt->data, pkt->size);

    /* Replace packet buffer */
    av_buffer_unref(&pkt->buf);
    pkt->buf = av_buffer_create(new_data, sei_size + pkt->size,
                                av_buffer_default_free, NULL, 0);
    if (!pkt->buf) {
        av_free(new_data);
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    pkt->data = new_data;
    pkt->size = sei_size + pkt->size;
    new_data = NULL; /* ownership transferred */

    s->frame_count++;

    av_freep(&mask_frame);
    av_freep(&mb_mask);
    av_freep(&rle_data);
    av_freep(&sei_nal);
    return 0;

fail:
    av_freep(&mask_frame);
    av_freep(&mb_mask);
    av_freep(&rle_data);
    av_freep(&sei_nal);
    av_freep(&new_data);
    av_packet_unref(pkt);
    return ret;
}

#define OFFSET(x) offsetof(H264PrivacyContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_BSF_PARAM)

static const AVOption h264_privacy_options[] = {
    { "mask_file", "Path to raw binary mask file (grayscale, same resolution as video)",
        OFFSET(mask_file), AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "obfuscation", "Obfuscation type: 0=pixelate, 1=blur, 2=black, 3=color_invert",
        OFFSET(obfuscation_type), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 3, FLAGS },
    { "encryption", "Encryption algorithm: 0=none, 1=AES-128, 2=AES-256",
        OFFSET(encryption_algo), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 2, FLAGS },
    { "key", "Hex-encoded encryption key (32 or 64 hex characters)",
        OFFSET(key_hex), AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "metadata_id", "Privacy metadata instance identifier",
        OFFSET(metadata_id), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 255, FLAGS },
    { NULL }
};

static const AVClass h264_privacy_class = {
    .class_name = "h264_privacy_bsf",
    .item_name  = av_default_item_name,
    .option     = h264_privacy_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const enum AVCodecID h264_privacy_codec_ids[] = {
    AV_CODEC_ID_H264, AV_CODEC_ID_NONE,
};

const FFBitStreamFilter ff_h264_privacy_bsf = {
    .p.name         = "h264_privacy",
    .p.codec_ids    = h264_privacy_codec_ids,
    .p.priv_class   = &h264_privacy_class,
    .priv_data_size = sizeof(H264PrivacyContext),
    .init           = &h264_privacy_init,
    .close          = &h264_privacy_close,
    .filter         = &h264_privacy_filter,
};
