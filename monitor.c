
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* monitor.c 2019/10/18 */

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

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>
#include "ggml.h"
#include "monitor.h"

extern int    NTHREAD; 
double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

double last_times[MAX_NTHREAD][MONITOREND];
double sep_times[MAX_NTHREAD][MONITOREND];

unsigned long last_ticks[MAX_NTHREAD][MONITOREND];
unsigned long sep_ticks[MAX_NTHREAD][MONITOREND];

void monitor_time_start(int th, int id) {
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  last_times[th][id] = tv.tv_sec + tv.tv_usec/1000000.0;

  times(&utms);
  last_ticks[th][id] = utms.tms_utime;
}

void monitor_time_end(int th, int id) {
  struct timeval tv;
  struct tms    utms;
  double now;
  unsigned long now_ticks;
  
  gettimeofday(&tv, NULL);
  now = tv.tv_sec + tv.tv_usec/1000000.0;
  sep_times[th][id] += now - last_times[th][id];

  times(&utms);
  now_ticks = utms.tms_utime;
  sep_ticks[th][id] += now_ticks - last_ticks[th][id];
}

void show_time() {
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tms = tv.tv_sec+tv.tv_usec/1000000.0;
  printf("====TOTAL-EXEC-TIME(w/o IO) %g sec===\n", (double)(tms - tmssave));

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));

  printf("====PARENT(w/ IO)===\n");
  getrusage(RUSAGE_SELF, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, (int)rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, (int)rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", (int)rusage.ru_maxrss);        /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", (int)(rusage.ru_ixrss/ticks)); /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", (int)(rusage.ru_idrss/ticks)); /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", (int)(rusage.ru_isrss/ticks)); /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", (int)rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", (int)rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", (int)rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", (int)rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", (int)rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", (int)rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", (int)rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", (int)rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", (int)rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", (int)rusage.ru_nivcsw);          /* involuntary context switches */

  printf("====CHILD(w/ IO)===\n");
  getrusage(RUSAGE_CHILDREN, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, (int)rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, (int)rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", (int)rusage.ru_maxrss);        /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", (int)(rusage.ru_ixrss/ticks)); /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", (int)(rusage.ru_idrss/ticks)); /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", (int)(rusage.ru_isrss/ticks)); /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", (int)rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", (int)rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", (int)rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", (int)rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", (int)rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", (int)rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", (int)rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", (int)rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", (int)rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", (int)rusage.ru_nivcsw);          /* involuntary context switches */
}

const char *monitor_names[MONITOREND] = {
  "T_MAIN_GPTNEOX",
  " T_LOAD",
  " T_EVAL",
  " T_PREDICT",
  "  T_GGML_INIT_GELU",
  "  T_GGML_INIT_GSTAT",
  "  T_COMPUTE_INIT",
  "  T_COMPUTE_NODES",
  "   T_COMPUTE_FORWARD",
  "    T_COMPUTE_FORWARD_DUP",
  "    T_COMPUTE_FORWARD_ADD",
  "    T_COMPUTE_FORWARD_SUB",
  "    T_COMPUTE_FORWARD_MUL",
  "    T_COMPUTE_FORWARD_DIV",
  "    T_COMPUTE_FORWARD_SQR",
  "    T_COMPUTE_FORWARD_SQRT",
  "    T_COMPUTE_FORWARD_SUM",
  "    T_COMPUTE_FORWARD_MEAN",
  "    T_COMPUTE_FORWARD_REPEAT",
  "    T_COMPUTE_FORWARD_ABS",
  "    T_COMPUTE_FORWARD_SGN",
  "    T_COMPUTE_FORWARD_NEG",
  "    T_COMPUTE_FORWARD_STEP",
  "    T_COMPUTE_FORWARD_RELU",
  "    T_COMPUTE_FORWARD_GELU",
  "    T_COMPUTE_FORWARD_SILU",
  "    T_COMPUTE_FORWARD_NORM",
  "    T_COMPUTE_FORWARD_MUL_MAT",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_ACC",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_INI",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_FIN",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00",
  "    T_GGML_VEC_DOT_Q4_0",
  "    IMAX_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00",
  "    IMAX_CPYIN",
  "    IMAX_CPYOUT",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_LT_NB00",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_1_F32",
  "    T_COMPUTE_FORWARD_MUL_MAT_Q4_F16_F32",
  "    T_COMPUTE_FORWARD_MUL_MAT_F32",
  "    T_COMPUTE_FORWARD_SCALE",
  "    T_COMPUTE_FORWARD_CPY",
  "    T_COMPUTE_FORWARD_RESHAPE",
  "    T_COMPUTE_FORWARD_VIEW",
  "    T_COMPUTE_FORWARD_PERMUTE",
  "    T_COMPUTE_FORWARD_TRANSPOSE",
  "    T_COMPUTE_FORWARD_GET_ROWS",
  "    T_COMPUTE_FORWARD_DIAG_MASK_INF",
  "    T_COMPUTE_FORWARD_SOFT_MAX",
  "    T_COMPUTE_FORWARD_ROPE",
  "    T_COMPUTE_FORWARD_GPTNEOX_ROPE",
  "    T_COMPUTE_FORWARD_ALIBI",
  "    T_COMPUTE_FORWARD_CONV_1D_1S",
  "    T_COMPUTE_FORWARD_CONV_1D_2S",
  "    T_COMPUTE_FORWARD_FLASH_ATTN",
  "    T_COMPUTE_FORWARD_FLASH_FF",
  "  T_GGML_OPT",
  " T_SAMPLE",
 };

void print_sep(int i)
{
  int t;

  printf("%-54s:", monitor_names[i]);
  for (t=0; t<NTHREAD; t++) {
    printf(" %5.1fs(%5.1f%%)",
	 (double)sep_times[t][i],
	 (double)sep_times[t][i]*100
	 /(sep_times[0][T_MAIN_GPTNEOX]));
  }
  printf("\n");
}

void show_time_sep(void) {
  int t;

  printf("%-54s:", " ");
  for (t=0; t<NTHREAD; t++)
    printf("   TH%02.2d        ", t);
  printf("\n");
  print_sep(T_MAIN_GPTNEOX);
  print_sep(T_LOAD);
  print_sep(T_EVAL);
  print_sep(T_PREDICT);
  print_sep(T_INIT_GELU);
  print_sep(T_INIT_GSTAT);
  print_sep(T_COMPUTE_INIT);
  print_sep(T_COMPUTE_NODES);
  print_sep(T_COMPUTE_FORWARD);
  print_sep(T_COMPUTE_FORWARD_DUP);
  print_sep(T_COMPUTE_FORWARD_ADD);
  print_sep(T_COMPUTE_FORWARD_SUB);
  print_sep(T_COMPUTE_FORWARD_MUL);
  print_sep(T_COMPUTE_FORWARD_DIV);
  print_sep(T_COMPUTE_FORWARD_SQR);
  print_sep(T_COMPUTE_FORWARD_SQRT);
  print_sep(T_COMPUTE_FORWARD_SUM);
  print_sep(T_COMPUTE_FORWARD_MEAN);
  print_sep(T_COMPUTE_FORWARD_REPEAT);
  print_sep(T_COMPUTE_FORWARD_ABS);
  print_sep(T_COMPUTE_FORWARD_SGN);
  print_sep(T_COMPUTE_FORWARD_NEG);
  print_sep(T_COMPUTE_FORWARD_STEP);
  print_sep(T_COMPUTE_FORWARD_RELU);
  print_sep(T_COMPUTE_FORWARD_GELU);
  print_sep(T_COMPUTE_FORWARD_SILU);
  print_sep(T_COMPUTE_FORWARD_NORM);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_ACC);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_INI);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_FIN);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00);
  print_sep(T_VEC_DOT_Q4_0);
  print_sep(IMAX_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_GE_NB00);
  print_sep(IMAX_CPYIN);
  print_sep(IMAX_CPYOUT);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_0_F32_NB01_LT_NB00);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_Q4_1_F32);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_F16_F32);
  print_sep(T_COMPUTE_FORWARD_MUL_MAT_F32);
  print_sep(T_COMPUTE_FORWARD_SCALE);
  print_sep(T_COMPUTE_FORWARD_CPY);
  print_sep(T_COMPUTE_FORWARD_RESHAPE);
  print_sep(T_COMPUTE_FORWARD_VIEW);
  print_sep(T_COMPUTE_FORWARD_PERMUTE);
  print_sep(T_COMPUTE_FORWARD_TRANSPOSE);
  print_sep(T_COMPUTE_FORWARD_GET_ROWS);
  print_sep(T_COMPUTE_FORWARD_DIAG_MASK_INF);
  print_sep(T_COMPUTE_FORWARD_SOFT_MAX);
  print_sep(T_COMPUTE_FORWARD_ROPE);
  print_sep(T_COMPUTE_FORWARD_GPTNEOX_ROPE);
  print_sep(T_COMPUTE_FORWARD_ALIBI);
  print_sep(T_COMPUTE_FORWARD_CONV_1D_1S);
  print_sep(T_COMPUTE_FORWARD_CONV_1D_2S);
  print_sep(T_COMPUTE_FORWARD_FLASH_ATTN);
  print_sep(T_COMPUTE_FORWARD_FLASH_FF);
  print_sep(T_OPT);
  print_sep(T_SAMPLE);
}
