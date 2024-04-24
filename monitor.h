#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

void reset_time();
void show_time();

void monitor_time_start(int, int);
void monitor_time_end(int, int);
void show_time_sep(void);

typedef enum {
  T_MAIN_GPTNEOX,
  T_LOAD,
  T_EVAL,
  T_PREDICT,
  T_INIT_GELU,
  T_INIT_GSTAT,
  T_COMPUTE_INIT,
  T_COMPUTE_NODES,
  T_COMPUTE_FORWARD,
  T_COMPUTE_FORWARD_DUP,
  T_COMPUTE_FORWARD_ADD,
  T_COMPUTE_FORWARD_SUB,
  T_COMPUTE_FORWARD_MUL,
  T_COMPUTE_FORWARD_DIV,
  T_COMPUTE_FORWARD_SQR,
  T_COMPUTE_FORWARD_SQRT,
  T_COMPUTE_FORWARD_SUM,
  T_COMPUTE_FORWARD_MEAN,
  T_COMPUTE_FORWARD_REPEAT,
  T_COMPUTE_FORWARD_ABS,
  T_COMPUTE_FORWARD_SGN,
  T_COMPUTE_FORWARD_NEG,
  T_COMPUTE_FORWARD_STEP,
  T_COMPUTE_FORWARD_RELU,
  T_COMPUTE_FORWARD_GELU,
  T_COMPUTE_FORWARD_SILU,
  T_COMPUTE_FORWARD_NORM,
  T_COMPUTE_FORWARD_MUL_MAT,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_ACC,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_INI,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_FIN,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00,
  T_VEC_DOT_Q4_0,
  IMAX_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00,
  IMAX_CPYIN,
  IMAX_CPYOUT,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_LT_NB00,
  T_COMPUTE_FORWARD_MUL_MAT_Q4_1_F32,
  T_COMPUTE_FORWARD_MUL_MAT_F16_F32,
  T_COMPUTE_FORWARD_MUL_MAT_F32,
  T_COMPUTE_FORWARD_SCALE,
  T_COMPUTE_FORWARD_CPY,
  T_COMPUTE_FORWARD_RESHAPE,
  T_COMPUTE_FORWARD_VIEW,
  T_COMPUTE_FORWARD_PERMUTE,
  T_COMPUTE_FORWARD_TRANSPOSE,
  T_COMPUTE_FORWARD_GET_ROWS,
  T_COMPUTE_FORWARD_DIAG_MASK_INF,
  T_COMPUTE_FORWARD_SOFT_MAX,
  T_COMPUTE_FORWARD_ROPE,
  T_COMPUTE_FORWARD_GPTNEOX_ROPE,
  T_COMPUTE_FORWARD_ALIBI,
  T_COMPUTE_FORWARD_CONV_1D_1S,
  T_COMPUTE_FORWARD_CONV_1D_2S,
  T_COMPUTE_FORWARD_FLASH_ATTN,
  T_COMPUTE_FORWARD_FLASH_FF,
  T_OPT,
  T_SAMPLE,
  MONITOREND} monitor_types;

#ifdef  __cplusplus
}
#endif
