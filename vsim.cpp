
/*          GGML          Copyright (c) 2023 Nolano.org */
/*          IMAX3         Copyright (C) 2013- by NAIST  */
/*                          Primary writer: Y.Nakashima */
/*                                 nakashim@is.naist.jp */
/* vsim.cpp 2024/3/14 */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <signal.h>
#include <unistd.h>
#include "ggml.h"
#include "utils.h"
#include "monitor.h"
#include "./emax7.h"
#define NO_EMAX7LIB_BODY
#include "./emax7lib.c"

extern "C" {
  void init_xmax();
}

// default hparams (GPT-J 6B)
struct gptneox_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t use_parallel_residual = 1; // 1 = true, 0 = false
    int32_t f16     = 1;
};

struct gptneox_layer {
    // input_layernorm
    struct ggml_tensor * input_layernorm_weight;
    struct ggml_tensor * input_layernorm_bias;

    // post_attention_layernorm
    struct ggml_tensor * post_attention_layernorm_weight;
    struct ggml_tensor * post_attention_layernorm_bias;

    // attention
    struct ggml_tensor * c_attn_q_proj_w;
    struct ggml_tensor * c_attn_k_proj_w;
    struct ggml_tensor * c_attn_v_proj_w;

    struct ggml_tensor * c_attn_q_proj_bias;
    struct ggml_tensor * c_attn_k_proj_bias;
    struct ggml_tensor * c_attn_v_proj_bias;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_bias;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w_trans;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gptneox_model {
    gptneox_hparams hparams;

    // final normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // word embedding

    struct ggml_tensor * lmh_g; // language model head
    // struct ggml_tensor * lmh_b; // language model bias

    std::vector<gptneox_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool gptneox_model_load(const std::string & fname, gptneox_model & model, gpt_vocab & vocab, int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        // fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.use_parallel_residual,     sizeof(hparams.use_parallel_residual));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: use_parallel_residual = %d\n", __func__, hparams.use_parallel_residual);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = model.hparams.n_vocab;
        // fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // wte

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype);         // lmh_g
        // ctx_size +=        n_vocab*ggml_type_sizef(GGML_TYPE_F32); // lmh_b

        { // Transformer layers
            { // Layernorms
                ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // input_layernorm_weight
                ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // input_layernorm_bias

                ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // post_attention_layernorm_weight
                ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // post_attention_layernorm_bias
            }

            { // Attention layer
                ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_q_proj_w
                ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_k_proj_w
                ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_v_proj_w

                ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_q_proj_bias
                ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_k_proj_bias
                ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_v_proj_bias

                ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_proj_w
                ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_proj_bias
            }

            { // Feedforward layer
                ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
                ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

                ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w_trans
                ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b
            }
        }

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 16*n_layer)*256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte    = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.lmh_g  = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        // model.lmh_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.wte;

        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;

        model.tensors["embed_out.weight"] = model.lmh_g;
        // model.tensors["lm_head.bias"]   = model.lmh_b;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            // Layernorms
            layer.input_layernorm_weight          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.input_layernorm_bias            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.post_attention_layernorm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.post_attention_layernorm_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // Attention
            layer.c_attn_q_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_k_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_v_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);

            layer.c_attn_q_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.c_attn_k_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.c_attn_v_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_proj_w         = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_proj_bias      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // Feedforward
            layer.c_mlp_fc_w            = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w_trans    = ggml_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_proj_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            // Layernorms
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"]          = layer.input_layernorm_weight;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]            = layer.input_layernorm_bias;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = layer.post_attention_layernorm_weight;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]   = layer.post_attention_layernorm_bias;

            // Attention
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query.weight"]   = layer.c_attn_q_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query.bias"]     = layer.c_attn_q_proj_bias;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.key.weight"]     = layer.c_attn_k_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.key.bias"]       = layer.c_attn_k_proj_bias;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.value.weight"]   = layer.c_attn_v_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.value.bias"]     = layer.c_attn_v_proj_bias;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"]   = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]     = layer.c_attn_proj_bias;

            // Feedforward
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"]    = layer.c_mlp_fc_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]      = layer.c_mlp_fc_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"]    = layer.c_mlp_proj_w_trans;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]      = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (0) {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                        {
                            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                            return false;
                        }
            };

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-NeoX model requires about 16MB of memory per input token.
//
bool gptneox_eval(
        const gptneox_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);

    // for (int il = 0; il < 1; ++il) {
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // input norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = input_layernorm_weight*cur + input_layernorm_bias
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].input_layernorm_weight, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].input_layernorm_bias, cur));

        }

        // self-attention
        {
            // Weight
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_q_proj_w, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_k_proj_w, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].c_attn_v_proj_w, cur);

            // Add bias
            Qcur = ggml_add(ctx0, Qcur, ggml_repeat(ctx0, model.layers[il].c_attn_q_proj_bias, Qcur));
            Kcur = ggml_add(ctx0, Kcur, ggml_repeat(ctx0, model.layers[il].c_attn_k_proj_bias, Kcur));
            Vcur = ggml_add(ctx0, Vcur, ggml_repeat(ctx0, model.layers[il].c_attn_v_proj_bias, Vcur));

            // // // // cur = ggml_add(ctx0, cur, Qcur);
            // // // // cur = ggml_add(ctx0, cur, Kcur);
            // // // // cur = ggml_add(ctx0, cur, Vcur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_gptneox_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_gptneox_rope(ctx0,// "change Qcur" in line 2270.)
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_proj_w, cur);

            // projection (then bias)
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_attn_proj_bias, cur), cur);
        }

        struct ggml_tensor * inpFF;

        if (hparams.use_parallel_residual == 0) {
            printf("use_parallel_residual == 0\n");
            // This takes the self-attention residual output as input to Feedforward
            inpFF = ggml_add(ctx0, cur, inpL);

            // post attention layer norm
            {
                inpFF = ggml_norm(ctx0, inpFF);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, inpFF),
                        inpFF),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, inpFF));
            }

            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, inpFF), inpFF);

                inpFF = ggml_gelu(ctx0, inpFF);

                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, inpFF), inpFF);
            }

            // inpL = inpFF + inpL
            inpL = ggml_add(ctx0, inpFF, inpL);

        } else if (hparams.use_parallel_residual == 1) {
            // printf("use_parallel_residual == 1\n");
            // This is independent of the self-attention result, so it could be done in parallel to the self-attention

            // post attention layer norm
            {
                inpFF = ggml_norm(ctx0, inpL);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                inpFF = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, inpFF),
                        inpFF),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, inpFF));
            }


            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, inpFF), inpFF);

                // GELU activation
                inpFF = ggml_gelu(ctx0, inpFF);

                // projection
                // inpFF = proj_w*inpFF + proj_b
                inpFF = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, inpFF);

                inpFF = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, inpFF), inpFF);
            }

            // inpL = inpL + inpFF + cur
            inpFF = ggml_add(ctx0, cur, inpFF);
            inpL = ggml_add(ctx0, inpL, inpFF);
        } else {
            printf("use_parallel_residual == %d\n", hparams.use_parallel_residual);
            assert(0);
        }
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main_gptneox(gpt_params params) {
    monitor_time_start(0, T_MAIN_GPTNEOX); /* Nakashima */

    std::mt19937 rng(params.seed);

    gpt_vocab vocab;
    gptneox_model model;
    // load the model
    {
      const int n_ctx = 512; // TODO: set context from user input ??
      monitor_time_start(0, T_LOAD); /* Nakashima */
      if (!gptneox_model_load(params.model, model, vocab, n_ctx)) {  // TODO: set context from user input ??
	fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
	return 1;
      }
      monitor_time_end(0, T_LOAD); /* Nakashima */
    }

    int n_past = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::whitespace_tokenize(params.prompt); //TODO: set bos to true?

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    // for (int i = 0; i < (int) embd_inp.size(); i++) {
    //     printf("%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    // }
    printf("\n");
    printf("sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    init_xmax(); /* Nakashima */
  
    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    monitor_time_start(0, T_EVAL); /* Nakashima */
    gptneox_eval(model, params.n_threads, 0, { 1, 2, 3, 4, 5 }, logits, mem_per_token);
    monitor_time_end(0, T_EVAL); /* Nakashima */

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    printf(" embd.size()=%d embd_inp.size()=%d params.n_predict=%d", embd.size(), embd_inp.size(), params.n_predict);
    printf("\n<|BEGIN> ");
    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
      // predict
      if (embd.size() > 0) {
	monitor_time_start(0, T_PREDICT); /* Nakashima */
	if (!gptneox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) { // update logits
	  printf("Failed to predict\n");
	  return 1;
	}
	monitor_time_end(0, T_PREDICT); /* Nakashima */
      }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            if (params.return_logits) {
                printf("logits: ");
                for (int i = 0; i < n_vocab; i++) {
                    // Upto 8 decimal places
                    printf("%.8f ", logits[i]);
                }
                printf(" <END|>\n");
                // Stdout should flush before returning
                fflush(stdout);
                return 0;
            }

            {
	      monitor_time_start(0, T_SAMPLE); /* Nakashima */

	      id = sample_top_p_top_k_repeat_penalty(
						     vocab,
						     logits.data() + (logits.size() - n_vocab),
						     last_n_tokens,
						     repeat_penalty,
						     top_k,
						     top_p,
						     temp,
						     rng);

	      // // print
	      // printf("\ngenerated token: '%s' (%d)\n", vocab.id_to_token[id].c_str(), id);

	      last_n_tokens.erase(last_n_tokens.begin());
	      last_n_tokens.push_back(id);

	      monitor_time_end(0, T_SAMPLE); /* Nakashima */
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                if (params.return_logits) {
                    printf("logits: ");
                    for (int i = 0; i < model.hparams.n_vocab; i++) {
                        // Upto 8 decimal places
                        printf("%.8f ", logits[i]);
                    }
                    printf("\n");
                }
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            if (!params.return_logits) {
                printf(" %d ", id);
            }
            // printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 2) {
            break;
        }
    }
    printf(" <END|>\n");

    monitor_time_end(0, T_MAIN_GPTNEOX); /* Nakashima */

    // report timing
    {
      show_time_sep();
    }

    ggml_free(model.ctx);

    return 0;
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

extern "C" {
void x11_open(Uint), x11_update(), BGR_to_X(), BOX_to_X();
int  x11_checkevent();
}

int  WD, HT, BITMAP, SCRWD, SCRHT, VECWD, VECHT, VECSTEP;
int  enable_x11, enable_x11vec, NTHREAD;

void copy_I_to_BGR(Uint *to, Uint batch, Uint w, Uint h, Uint *from)
{
  int i, j, k;
  for (i=0; i<batch; i++) {
    int x = (i%10)*w;       /* 28 */
    int y = (i/10)*h;       /* 28 */
    for (j=0; j<h; j++) {   /* 28 */
      for (k=0; k<w; k++) { /* 28 */
	to[(y+j)*WD+x+k] = *from++;
      }
    }
  }
}

void copy_BGR(Uint *to, Uint *from)
{
  int i;
  for (i=0; i<HT*WD; i++)
    *to++ = *from++;
}

void clear_BGR(Uint *to)
{
  int i;
  for (i=0; i<HT*WD; i++)
    *to++ = 0;
}

int main(int argc, char ** argv)
{
  WD            = 320;
  HT            = 240;
  BITMAP        = WD*HT;
  SCRWD         = 3;
  SCRHT         = 2;
  VECWD         = 3;//FC_DEPTH;
  VECHT         = 5;
  VECSTEP       = 4;
  enable_x11    = 0; /* default off */
  enable_x11vec = 0; /* default off */
  NTHREAD       = 1; /* default 1   */

  if (enable_x11)
    x11_open(enable_x11vec); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/

  gpt_params params; // We say "gpt", but it's actually any LLM
  // params.model = "models/ggml-model-bloomz-7b1-f16-q4_0.bin";
  // params.prompt = "Je vais";

  // loop through argv and print all the arguments, one per line
  for (int i = 0; i < argc; i++) {
    printf("argv[%d] = %s\n", i, argv[i]);
  }

  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }
  // return if params.model or params.prompt are empty
  if (params.model.empty() || params.prompt.empty()) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  printf("%s: seed = %d\n", __func__, params.seed);

  if (params.prompt.empty()) {
    return 1;
  }

  // Get the model type from argv[1]
  std::string model_type = argv[1];
  printf("model_type: %s\n", model_type.c_str());

  if (params.return_logits) {
    printf("********************************\n");
    printf("*** return_logits mode ***\n");
    printf("*** setting sampling to greedy ***\n");
    printf("********************************\n");
    // model_type should be either gptj or gptneox or bloom
    // if (model_type != "gptj" && model_type != "gptneox" && model_type != "bloom") {
    //     printf("model_type: %s, should be either gptj or gptneox or bloom\n", model_type.c_str());
    //     assert(false);
    // }
  }

  if (model_type == "gptneox") {
    return main_gptneox(params);
  } else {
    printf("Unknown model type: %s\n", model_type.c_str());
    return 1;
  }
}