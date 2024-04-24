
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

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
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#ifdef CBLAS_GEMM
#include "cblas.h"
#endif
#include "ggml.h"
#include "monitor.h"
#include "./emax7.h"
#include "./emax7lib.c"

void *memcpy();

Uchar   *membase;
int     memsize;
int     memalign;

void    *i_m0A[EMAX_NLANE]; /* for imax_ggml_compute_forward_mul_mat_q4_0_f32 on ACAP_PL */
void    *i_m0B[EMAX_NLANE]; /* for imax_ggml_compute_forward_mul_mat_q4_0_f32 on ACAP_PL */
void    *i_m0C[EMAX_NLANE]; /* for imax_ggml_compute_forward_mul_mat_q4_0_f32 on ACAP_PL */
int     i_m0A_max_size;
int     i_m0B_max_size;
int     i_m0C_max_size;

#define ERRTH  (5.0E-2)
#define udiff(a,b) (((a)-(b)>=0.0?(a)-(b):(b)-(a))/((a)==0.0?1:(a)))
#define setmax(max, new) { if (max < (new)) max = (new); }

void init_xmax()
{
  int l;

  setmax(i_m0A_max_size, 40230400);
  setmax(i_m0B_max_size,    16000);
  setmax(i_m0C_max_size,   251440);
  setmax(memsize, (i_m0A_max_size+i_m0B_max_size+i_m0C_max_size)*sizeof(int));
  memalign = 32;

#if defined(EMAX7)
#if defined(ARMZYNQ)
  if ((NLANE = emax7_open(2)) == NULL) /* EMAX7 MACRO_PIPELIING (vsim-zynq,vsim-zynq.emax7*) */
    exit(1);
  if (memsize*NLANE > DDR_MMAP_SIZE) {
    printf("memsize*NLANE: %08.8x exceeds DDR_MMAP_SIZE=%08.8x\n", (Uint)memsize*NLANE, (Uint)DDR_MMAP_SIZE);
    exit(1);
  }
#else
  NLANE = 2; /* EMAX7 MACRO_PIPELIING (vsim-bsd.emax7nc,vsim-cent.emax7nc) */
#endif
#endif

#if defined(EMAX7) && defined(ARMZYNQ)
  membase = emax_info[0].ddr_mmap;
  /*{int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}*/
#elif __linux__ == 1
  if (posix_memalign(&membase, memalign, memsize*NLANE) != 0) {
    printf("posix_memalign: filed to allocate size=%08.8x\n", (Uint)memsize*NLANE);
    exit(1);
  }
#else
  if ((membase = (void*)malloc(memsize*NLANE+memalign)) == NULL) {
    printf("malloc: filed to allocate size=%08.8x\n", (Uint)memsize*NLANE+memalign);
    exit(1);
  }
  if ((Ull)membase & (Ull)(memalign-1))
    membase = (void*)(((Ull)membase & ~(Ull)(memalign-1))+memalign);
#endif

  printf("membase: %08.8x\n", (Uint)membase);

  for (l=0; l<NLANE; l++) {
    i_m0A[l] = (Uint*)membase+memsize/sizeof(int)*l;
    i_m0B[l] = (Uint*)i_m0A[l] + i_m0A_max_size;
    i_m0C[l] = (Uint*)i_m0B[l] + i_m0B_max_size;
    printf("i_m0A[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0A[l], (Uint)i_m0A[l]+i_m0A_max_size*sizeof(int)-1);
    printf("i_m0B[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0B[l], (Uint)i_m0B[l]+i_m0B_max_size*sizeof(int)-1);
    printf("i_m0C[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0C[l], (Uint)i_m0C[l]+i_m0C_max_size*sizeof(int)-1);
  }

#if defined(EMAX7) && (defined(ARMSIML) || defined(ARMZYNQ))
  { int i;
    int stat;
    for (i=0; i<NLANE; i++) {
      emax7[i].dma_ctrl  = emax_info[i].dma_mmap;
      emax7[i].reg_ctrl  = emax_info[i].reg_mmap;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
#if defined(ARMZYNQ)
      usleep(1);
#endif
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].adtr = emax_info[i].ddr_mmap - emax_info[i].lmm_phys;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].dmrp = 0LL;
      stat = ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].stat>>8 & 0xf;
      switch (stat) {
      case  0:EMAX_DEPTH =  8;break;
      case  1:EMAX_DEPTH = 16;break;
      case  2:EMAX_DEPTH = 32;break;
      case  3:EMAX_DEPTH = 64;break;
      default:
        printf("init_xmax: illegal stat=%x for setting EMAX_DEPTH\n", stat);
        exit(1);
      }

      stat = ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].stat>>12 & 0xf;
      switch (stat) {
      case  0:LMM_SIZE =  32768;break;
      case  1:LMM_SIZE =  65536;break;
      case  2:LMM_SIZE = 131072;break;
      case  3:LMM_SIZE = 262144;break;
      case  4:LMM_SIZE = 524288;break;
      default:
        printf("init_xmax: illegal stat=%x for setting LMM_SIZE\n", stat);
        exit(1);
      }
      printf("EMAX7[%d].DEPTH=%d LMM_SIZE=%d\n", i, EMAX_DEPTH, LMM_SIZE);
    }
  }
  printf("EMAX7: NLANE=%d DEPTH=%d LMM_SIZE=%d\n", NLANE, EMAX_DEPTH, LMM_SIZE);
#endif
}

void imemcpy(Uint *dst, Uint *src, int words)
{
  union {
    Uint i[4];
    Ull  l[2];
    Dll  d;
  } buf;

  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = *src++;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    if ((Ull)src & sizeof(Uint)) {
      for(i=0; i<loop; i++) {
        buf.i[0] = *src++;
        buf.i[1] = *src++;
        buf.i[2] = *src++;
        buf.i[3] = *src++;
        *(Dll*)dst = buf.d;
        dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else if ((Ull)src & sizeof(Ull)) {
      for(i=0; i<loop; i++) {
        buf.l[0] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
        buf.l[1] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
        *(Dll*)dst = buf.d;
        dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else {
      for(i=0; i<loop; i++) {
        *(Dll*)dst = *(Dll*)src;
        src += sizeof(Dll)/sizeof(Uint);
        dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = *src++;
    words--;
  }
}

void __attribute__((optimize("O1"))) xmax_bzero(Uint *dst, int words)
{
  /* +----+-m-----+ */
  /* |3x3 |       | */
  /* |    |    src| */
  /* +----+       | */
  /* |       +----+ */
  /* |       |    | */
  /* |       | 3x3| */
  /* +-------+----+ */
  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = 0;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    for(i=0; i<loop; i++) {
#if __AARCH64EL__ == 1
      *((Dll*)dst) = 0;
#else
      ((Dll*)dst)->u[0] = 0;
      ((Dll*)dst)->u[1] = 0;
#endif
      dst += sizeof(Dll)/sizeof(Uint);
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = 0;
    words--;
  }
}

void __attribute__((optimize("O1"))) xmax_cpyin(int order, Uint *dst, int *imo, Uint *src, int batch, int ic, int im, int m, int k)
{
  /* order 0: dst[batch][ic][im*im]  <- src[batch][ic][im*im] */
  /* order 1: dst[ic][oc][im*im]     <- src[oc][ic][im*im] */
  /* order 2: dst[ic][im][batch][im] <- src[batch][ic][im*im] */
  /* order 3: dst[im][m]             <- src[im][m]            */

  switch (order) {
  case 0:
    /* num=batch+ichan                            */
    /* imi¤Î¼þÊÕ¤Ë0¤òÄÉ²Ã¤·imo¤Ë¥³¥Ô¡¼            */
    /* k=3,(IM==M)             k=2,(IM==M)        */
    /* +-------+imo-------+    +-----+--imo----+  */
    /* | 0 0 0 |       dst|    | 0 0 |      dst|  */
    /* |  +----+im=m---+  |    |  +--+--im=m---+  */
    /* | 0|3x3 |       |  |    | 0|  |         |  */
    /* | 0|    |    src|  |    +--+--+      src|  */
    /* +--+----+       |  |    |  |            |  */
    /* |  |       +----+--+    |  |            |  */
    /* |  |       |    |0 |    |  |            |  */
    /* |  |       | 3x3|0 |    |  |            |  */
    /* |  +-------+----+  |    +--+------------+  */
    /* |          | 0 0 0 |                       */
    /* +----------+-------+                       */

    /* imi¤Èimo¤ÏÆ±¤¸¥µ¥¤¥º¤Ç¥³¥Ô¡¼                                 */
    /* k=3,(IM-k)/1+1==M       k=2,(IM-k)/1+1==M    k=1,(IM==M)     */
    /* +-------+im--------+    +-----+--im-----+                    */
    /* | x x x |       dst|    | x x |      dst|                    */
    /* |  +----+-m-----+  |    |  +--+---m-----+    +--+--im=m---+  */
    /* | x|3x3 |       |  |    | x|  |         |    |  |         |  */
    /* | x|    |    src|  |    +--+--+      src|    +--+      src|  */
    /* +--+----+       |  |    |  |            |    |            |  */
    /* |  |       +----+--+    |  |            |    |            |  */
    /* |  |       |    |x |    |  |            |    |         +--+  */
    /* |  |       | 3x3|x |    |  |            |    |         |  |  */
    /* |  +-------+----+  |    +--+------------+    +---------+--+  */
    /* |          | x x x |                                         */
    /* +----------+-------+                                         */
    /* EMAX for large IM/M                                   *//*         burst_exe 6*6    ||         burst_exe 6*6    */
    /*     +-----+  +----+-+----+---------+    +-----------+ *//* 7*8... | 7*8... | 7*8... || 7*8... | 7*8... | 7*8... */
    /* unit|2    |  |7*7 | |7*7 |*IC  *100|    |2          | *//*-- -- --                  ||-- -- --                  *//* LMM=7*8*4B */
    /*  |  |*    |  |ch0 | |ch1 |         | -> |*          | *//*         -- -- --         ||         -- -- --         *//*    =244B   */
    /*  V  |2    |  +----+ +----+         |    |2          | *//*                  -- -- --||                  -- -- --*/
    /*     |*ich |  |loop=RMGRP(6)*M(6)   |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*4och */
    /*     +-och-+  +---------------------+    +6*6*och----+ *//* img0     img0     img0   || img1     img1     img1   *//*    =576B        */
    /*        32 ... lmf+lmxËè²óDMA            |    32/4   | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n, i, w = im+k-1;
      for (n=0; n<batch*ic; n++,dst+=w*w,src+=im*im) {
        for (i=0; i<k/2; i++)
          xmax_bzero(dst+i*w, w);
        for (i=k/2; i<=im+k/2-1; i++) {
          xmax_bzero (dst+i*w,               (k/2) );
          imemcpy(dst+i*w+(k/2),   src+(i-k/2)*im, im);
          if (k-1-(k/2)) xmax_bzero (dst+i*w+(k/2)+im, k-1-(k/2));
        }
        for (i=im+k/2; i<w; i++)
          xmax_bzero(dst+i*w, w);
      }
      *imo = w;
    }
    else {
      imemcpy(dst, src, batch*ic*im*im);
      *imo = im;
    }
    break;
  case 1:  /* dst[ic][oc][im*im] <- src[oc][ic][im*im] */
    {
      int i, o;
      for (o=0; o<batch; o++) {
        for (i=0; i<ic; i++)
          imemcpy(dst+(i*batch+o)*im*im, src+(o*ic+i)*im*im, im*im);
      }
      *imo = im;
    }
    break;
  case 2:
    /* EMAX for small IM/M                                   */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100  ||         burst_exe 6*100  *//* 100²èÁü¤ò1Ëç(7*700pix)¤Ë(7*100¤ò7¹Ô) */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*8*100| 7*8*100| 7*8*100|| 7*8*100| 7*8*100| 7*8*100*//* ¤Þ¤¿¤Ï7*7Ï¢Â³¥¢¥É¥ì¥¹¤ò100¥»¥Ã¥È     */
    /*  |  |2    |  ||7*7 | |7*7 |*100 *IC| -> |2          | *//*-- -- --                    -- -- --                  *//* LMM=7*8*4B*100 LMMstg2-7¤Ëload       */
    /*  |  |*    |  ||im0 | |im1 |        |    |*          | *//* top=0   -- -- --            top=1   -- -- --         *//*    =22400B(RMGRP=7¤Ç2²óºÆÍøÍÑ)<32KB  */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                  -- -- --                    -- -- --*/
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0-99  img0-99  img0-99|| img0-99  img0-99  img0-99*//*    =9600B         */
    /*        32 ... lmf+lmxËè²óDMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n1, n0, i, w = im+k-1;
      for (n1=0; n1<batch; n1++) {           /* src-data½ç */
        for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data½ç */
          int ofs  = (n0*w*batch+n1)*w;      /* Ê£¿ôimg¤Î1¹Ô¤¬Ï¢Â³,chËè¤ËÏ¢Â³ */
          int dist =  batch*w;               /* Ê£¿ôimg¤Î1¹Ô¤¬Ï¢Â³,»þ¥¢¥É¥ì¥¹¤Ï¼¡¹Ô */
          for (i=0; i<k/2; i++)
            xmax_bzero(dst+ofs+i*dist, w);
          for (i=k/2; i<=im+k/2-1; i++) {
            xmax_bzero (dst+ofs+i*dist,               (k/2) );
            imemcpy(dst+ofs+i*dist+(k/2),   src+(i-k/2)*im, im);
            if (k-1-(k/2)) xmax_bzero (dst+ofs+i*dist+(k/2)+im, k-1-(k/2));
          }
          for (i=im+k/2; i<w; i++)
            xmax_bzero(dst+ofs+i*dist, w);
        }
      }
      *imo = w;
    }
    else {
      int n1, n0, i;
      for (n1=0; n1<batch; n1++) {           /* src-data½ç */
        for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data½ç */
          int ofs  = (n0*im*batch+n1)*im;
          int dist =  batch*im;
          for (i=0; i<im; i++)
            imemcpy(dst+ofs+i*dist, src+i*im, im);
        }
      }
      *imo = im;
    }
    break;
  case 3:
    imemcpy(dst, src, im*m);
    *imo = im;
    break;
  }
}

void __attribute__((optimize("O1"))) xmax_cpyout(int order, Uint *dst, int batch, int oc, Uint *src, int m, int n, int oc4)
{
  /* order 0: dst[batch][oc][m*n] <- src[batch][oc4][m*n]  */
  /* order 1: dst[batch][oc][m*n] <- src[oc4][m][batch][n] */
  /* order 2: dst[m][n]           <- src[m][oc4=(n+3)&~3]  */

  /* +-dst--------------+    +-imo--------------+ */
  /* | OC | OC | OC |   | <- | OC4   | OC4   |  | */
  /* +------------------+    +------------------+ */
  int k, k2, k1, k0;

  switch (order) {
  case 0:
    for (k=0; k<batch; k++,dst+=oc*m*n,src+=oc4*m*n)
      imemcpy(dst, src, oc*m*n);
    break;
  case 1:
    for (k2=0; k2<batch; k2++) {
      for (k1=0; k1<oc; k1++) {
        for (k0=0; k0<m; k0++,dst+=n)
          imemcpy(dst, src+((k1*m+k0)*batch+k2)*n, n);
      }
    }
    break;
  case 2:
    if (n == oc4)
      imemcpy(dst, src, m*n);
    else {
      for (k=0; k<m; k++,dst+=n,src+=oc4)
        imemcpy(dst, src, n);
    }
    break;
  }
}

/* LMM=128KB¤Î¾ì¹ç */
/* -I0 -C1 -F1 */
/*  CNN5x5  BATCH=100 M=24 RMGRP=2  IC=1  OC=16 outloop=48   inloop=2400 Klen=  400/16384 IMlen=14000/16384 Mlen=2400/8192 */
/*  GEMM00  m=100 n=10 ka=2304(/H)  8*2*3*3*16  outloop=960  inloop=15   Blen=   10/16384  Alen=11520/16384 Clen=  50/32768*/
/* -I0 -C3 -F1 */
/*  CNN5x5  BATCH=100 M=24 RMGRP=2  IC=1  OC=16 outloop=48   inloop=2400 Klen=  400/16384 IMlen=14000/16384 Mlen=2400/8192 */
/*  CNN3x3x4          M=12 RMGRP=1  IC=16 OC=16 outloop=192  inloop=1200 Klen= 2304/16384 IMlen= 4200/16384 Mlen=1200/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=16 OC=32 outloop=96   inloop=600  Klen= 1024/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  GEMM00  m=100 n=10 ka=1152(/H)  8*2*3*3*8   outloop=240  inloop=30   Blen=   10/16384  Alen=11520/16384 Clen= 100/32768*/
/* -I1 -C4 -F1 */
/*  CNN5x5  BATCH=100 M=28 RMGRP=2  IC=3  OC=18 outloop=210  inloop=2800 Klen= 1350/16384 IMlen=16000/16384 Mlen=2800/8192 */
/*  CNN3x3x6          M=14 RMGRP=1  IC=18 OC=16 outloop=168  inloop=1400 Klen= 2592/16384 IMlen= 4800/16384 Mlen=1400/8192 */
/*  CNN2x2            M=7  RMGRP=1  IC=16 OC=32 outloop=112  inloop=700  Klen= 1024/16384 IMlen= 1600/16384 Mlen= 700/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=32 OC=64 outloop=384  inloop=600  Klen= 2048/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  GEMM00  m=100 n=10 ka=576(/H)   8*2*3*3*4   outloop=60   inloop=60   Blen=   10/16384  Alen=11520/16384 Clen= 200/32768*/
/* -I1 -C6 -F2 */
/*  CNN5x5  BATCH=100 M=28 RMGRP=2  IC=3  OC=18 outloop=210  inloop=2800 Klen= 1350/16384 IMlen=16000/16384 Mlen=2800/8192 */
/*  CNN3x3x6          M=14 RMGRP=1  IC=18 OC=16 outloop=168  inloop=1400 Klen= 2592/16384 IMlen= 4800/16384 Mlen=1400/8192 */
/*  CNN2x2            M=7  RMGRP=1  IC=16 OC=32 outloop=112  inloop=700  Klen= 1024/16384 IMlen= 1600/16384 Mlen= 700/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=32 OC=64 outloop=384  inloop=600  Klen= 2048/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  CNN2x2            M=2  RMGRP=1  IC=64 OC=64 outloop=256  inloop=200  Klen= 2048/16384 IMlen=  600/16384 Mlen= 200/8192 */
/*  CNN2x2            M=1  RMGRP=1  IC=64 OC=128outloop=256  inloop=100  Klen= 4096/16384 IMlen=  400/16384 Mlen= 100/8192 */
/*  GEMM00  m=100 n=40 ka=128(/H)   8*8*2       outloop=4    inloop=1000 Blen=   40/16384  Alen=12800/16384 Clen=4000/32768*/
/*  GEMM00  m=100 n=10 ka=40(/H)    8*5         outloop=1    inloop=300  Blen=   10/16384  Alen= 4000/16384 Clen=1000/32768*/

void xmax_sgemm00_48(int, int, int, int, int, float*, float*, float*); /* C=A*B */
void xmax_sgemm00_32(int, int, int, int, int, float*, float*, float*); /* C=A*B */
void xmax_sgemm00_40(int, int, int, int, int, float*, float*, float*); /* C=A*B */

void xmax_sgemm00(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  if (ka % 48 == 0)
    xmax_sgemm00_48(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else if (ka % 32 == 0)
    xmax_sgemm00_32(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else if (ka % 40 == 0)
    xmax_sgemm00_40(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else {
    printf("xmax_sgemm00 error: ka=%d\n", ka);
    exit(-1);
  }
}

void xmax_sgemm00_48(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     48
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMM_SIZE/4/2)/ka>100 ? 100:
          (LMM_SIZE/4/2)/ka>20  ? 20:
          (LMM_SIZE/4/2)/ka>10  ? 10:
          (LMM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMM_SIZE/4/2 || Alen > LMM_SIZE/4/2 || Clen > LMM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
           (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMM_SIZE/4/2, (Uint)Alen, LMM_SIZE/4/2, (Uint)Clen, LMM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4];} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
        b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3;
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
        for (k=0; k<H; k++)
          a[k][CHIP] = a0[CHIP]+blk+k;
        c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
        c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_48_core1(r, rm1, rp1) \
            mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
            exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_48_final(r, rp1, Force) \
            exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
            mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
            mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
            mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_48 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

            mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
            mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
            exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

            sgemm00_48_core1( 2,  1,  3);
            sgemm00_48_core1( 3,  2,  4);
            sgemm00_48_core1( 4,  3,  5);
            sgemm00_48_core1( 5,  4,  6);
            sgemm00_48_core1( 6,  5,  7);
            sgemm00_48_core1( 7,  6,  8);
            sgemm00_48_core1( 8,  7,  9);
            sgemm00_48_core1( 9,  8, 10);
            sgemm00_48_core1(10,  9, 11);
            sgemm00_48_core1(11, 10, 12);
            sgemm00_48_core1(12, 11, 13);
            sgemm00_48_core1(13, 12, 14);
            sgemm00_48_core1(14, 13, 15);
            sgemm00_48_core1(15, 14, 16);
            sgemm00_48_core1(16, 15, 17);
            sgemm00_48_core1(17, 16, 18);
            sgemm00_48_core1(18, 17, 19);
            sgemm00_48_core1(19, 18, 20);
            sgemm00_48_core1(20, 19, 21);
            sgemm00_48_core1(21, 20, 22);
            sgemm00_48_core1(22, 21, 23);
            sgemm00_48_core1(23, 22, 24);
            sgemm00_48_core1(24, 23, 25);
#if (H==24)
            sgemm00_48_final(25,     27, Force);
#endif
#if (H>24)
            sgemm00_48_core1(25, 24, 26);
            sgemm00_48_core1(26, 25, 27);
            sgemm00_48_core1(27, 26, 28);
            sgemm00_48_core1(28, 27, 29);
            sgemm00_48_core1(29, 28, 30);
            sgemm00_48_core1(30, 29, 31);
            sgemm00_48_core1(31, 30, 32);
            sgemm00_48_core1(32, 31, 33);
            sgemm00_48_core1(33, 32, 34);
            sgemm00_48_core1(34, 33, 35);
            sgemm00_48_core1(35, 34, 36);
            sgemm00_48_core1(36, 35, 37);
            sgemm00_48_core1(37, 36, 38);
            sgemm00_48_core1(38, 37, 39);
            sgemm00_48_core1(39, 38, 40);
            sgemm00_48_core1(40, 39, 41);
            sgemm00_48_core1(41, 40, 42);
            sgemm00_48_core1(42, 41, 43);
            sgemm00_48_core1(43, 42, 44);
            sgemm00_48_core1(44, 43, 45);
            sgemm00_48_core1(45, 44, 46);
            sgemm00_48_core1(46, 45, 47);
            sgemm00_48_core1(47, 46, 48);
            sgemm00_48_core1(48, 47, 49); /* 288/6 H=48 */
#endif
#if (H==48)
            /****final*****/
            sgemm00_48_final(49,     51, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm00_32(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     32
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMM_SIZE/4/2)/ka>100 ? 100:
          (LMM_SIZE/4/2)/ka>20  ? 20:
          (LMM_SIZE/4/2)/ka>10  ? 10:
          (LMM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMM_SIZE/4/2 || Alen > LMM_SIZE/4/2 || Clen > LMM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
           (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMM_SIZE/4/2, (Uint)Alen, LMM_SIZE/4/2, (Uint)Clen, LMM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4];} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
        b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3;
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
        for (k=0; k<H; k++)
          a[k][CHIP] = a0[CHIP]+blk+k;
        c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
        c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_32_core1(r, rm1, rp1) \
            mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
            exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_32_final(r, rp1, Force) \
            exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
            mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
            mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
            mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_32 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

            mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
            mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
            exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

            sgemm00_32_core1( 2,  1,  3);
            sgemm00_32_core1( 3,  2,  4);
            sgemm00_32_core1( 4,  3,  5);
            sgemm00_32_core1( 5,  4,  6);
            sgemm00_32_core1( 6,  5,  7);
            sgemm00_32_core1( 7,  6,  8);
            sgemm00_32_core1( 8,  7,  9);
            sgemm00_32_core1( 9,  8, 10);
            sgemm00_32_core1(10,  9, 11);
            sgemm00_32_core1(11, 10, 12);
            sgemm00_32_core1(12, 11, 13);
            sgemm00_32_core1(13, 12, 14);
            sgemm00_32_core1(14, 13, 15);
            sgemm00_32_core1(15, 14, 16);
            sgemm00_32_core1(16, 15, 17);
#if (H==16)
            sgemm00_32_final(17,     19, Force);
#endif
#if (H>16)
            sgemm00_32_core1(17, 16, 18);
            sgemm00_32_core1(18, 17, 19);
            sgemm00_32_core1(19, 18, 20);
            sgemm00_32_core1(20, 19, 21);
            sgemm00_32_core1(21, 20, 22);
            sgemm00_32_core1(22, 21, 23);
            sgemm00_32_core1(23, 22, 24);
            sgemm00_32_core1(24, 23, 25);
            sgemm00_32_core1(25, 24, 26);
            sgemm00_32_core1(26, 25, 27);
            sgemm00_32_core1(27, 26, 28);
            sgemm00_32_core1(28, 27, 29);
            sgemm00_32_core1(29, 28, 30);
            sgemm00_32_core1(30, 29, 31);
            sgemm00_32_core1(31, 30, 32);
            sgemm00_32_core1(32, 31, 33);
#endif
#if (H==32)
            /****final*****/
            sgemm00_32_final(33,     35, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm00_40(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     40
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMM_SIZE/4/2)/ka>100 ? 100:
          (LMM_SIZE/4/2)/ka>20  ? 20:
          (LMM_SIZE/4/2)/ka>10  ? 10:
          (LMM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMM_SIZE/4/2 || Alen > LMM_SIZE/4/2 || Clen > LMM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
           (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMM_SIZE/4/2, (Uint)Alen, LMM_SIZE/4/2, (Uint)Clen, LMM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4];} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
        b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3;
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
        for (k=0; k<H; k++)
          a[k][CHIP] = a0[CHIP]+blk+k;
        c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
        c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_40_core1(r, rm1, rp1) \
            mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
            mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
            exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_40_final(r, rp1, Force) \
            exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
            mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
            mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
            cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
            mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_40 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

            mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
            mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
            mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
            exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
            exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

            sgemm00_40_core1( 2,  1,  3);
            sgemm00_40_core1( 3,  2,  4);
            sgemm00_40_core1( 4,  3,  5);
            sgemm00_40_core1( 5,  4,  6);
            sgemm00_40_core1( 6,  5,  7);
            sgemm00_40_core1( 7,  6,  8);
            sgemm00_40_core1( 8,  7,  9);
            sgemm00_40_core1( 9,  8, 10);
            sgemm00_40_core1(10,  9, 11);
            sgemm00_40_core1(11, 10, 12);
            sgemm00_40_core1(12, 11, 13);
            sgemm00_40_core1(13, 12, 14);
            sgemm00_40_core1(14, 13, 15);
            sgemm00_40_core1(15, 14, 16);
            sgemm00_40_core1(16, 15, 17);
            sgemm00_40_core1(17, 16, 18);
            sgemm00_40_core1(18, 17, 19);
            sgemm00_40_core1(19, 18, 20);
            sgemm00_40_core1(20, 19, 21);
#if (H==20)
            sgemm00_40_final(21,     23, Force);
#endif
#if (H>20)
            sgemm00_40_core1(21, 20, 22);
            sgemm00_40_core1(22, 21, 23);
            sgemm00_40_core1(23, 22, 24);
            sgemm00_40_core1(24, 23, 25);
            sgemm00_40_core1(25, 24, 26);
            sgemm00_40_core1(26, 25, 27);
            sgemm00_40_core1(27, 26, 28);
            sgemm00_40_core1(28, 27, 29);
            sgemm00_40_core1(29, 28, 30);
            sgemm00_40_core1(30, 29, 31);
            sgemm00_40_core1(31, 30, 32);
            sgemm00_40_core1(32, 31, 33);
            sgemm00_40_core1(33, 32, 34);
            sgemm00_40_core1(34, 33, 35);
            sgemm00_40_core1(35, 34, 36);
            sgemm00_40_core1(36, 35, 37);
            sgemm00_40_core1(37, 36, 38);
            sgemm00_40_core1(38, 37, 39);
            sgemm00_40_core1(39, 38, 40);
            sgemm00_40_core1(40, 39, 41);
#endif
#if (H==40)
            /****final*****/
            sgemm00_40_final(41,     43, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm10(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, ka, 1.0f, A, m, B, n, 0.0f, C, n);
#else
  for (k=0; k<ka; k++) {
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
        if (k==0) C[row*n+col]  = A[k*m+row] * B[k*n+col];
        else      C[row*n+col] += A[k*m+row] * B[k*n+col];
      }
    }
  }
#endif

  /* ¡ú¡ú¡ú PBL1-2 ¡ú¡ú¡ú */
}

void xmax_sgemm01(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, ka, 1.0f, A, ka, B, ka, 0.0f, C, n);
#else
  for (row=0; row<m; row++) {
    for (col=0; col<n; col++) {
      for (k=0; k<ka; k++) {
        if (k==0) C[row*n+col]  = A[row*ka+k] * B[col*ka+k];
        else      C[row*n+col] += A[row*ka+k] * B[col*ka+k];
      }
    }
  }
#endif

  /* ¡ú¡ú¡ú PBL1-3 ¡ú¡ú¡ú */
}

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define QK 32

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = { QK, QK, 1, 1, 1, 1, 1, };

enum ggml_task_type {
  GGML_TASK_INIT = 0,
  GGML_TASK_COMPUTE,
  GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
  enum ggml_task_type type;
  int ith, nth;
  // work buffer for all threads
  size_t wsize;
  void * wdata;
};

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
  sizeof(float  )   + QK/2,
  sizeof(float  )*2 + QK/2,
  sizeof(int8_t ),
  sizeof(int16_t),
  sizeof(int32_t),
  sizeof(ggml_fp16_t),
  sizeof(float  ),
};

void imax_ggml_compute_forward_mul_mat_q4_0_f32(
  int THREAD, /* temporally set 0 by ggml.c */
  int LANE,   /* temporally set 0 by ggml.c */
  const struct ggml_compute_params * params,
  const struct ggml_tensor * src0,
  const struct ggml_tensor * src1,
  struct ggml_tensor * dst) {

  const int    ne00 = src0->ne[0]; const int ne01 = src0->ne[1]; const int ne02 = src0->ne[2]; const int ne03 = src0->ne[3];
  const int    ne10 = src1->ne[0]; const int ne11 = src1->ne[1]; const int ne12 = src1->ne[2]; const int ne13 = src1->ne[3];
  const int    ne0  = dst->ne[0];  const int ne1  = dst->ne[1];  const int ne2  = dst->ne[2];  const int ne3  = dst->ne[3];
  const int    ne   = ne0*ne1*ne2*ne3;
  const int    nb00 = src0->nb[0]; const int nb01 = src0->nb[1]; const int nb02 = src0->nb[2]; const int nb03 = src0->nb[3];
  const int    nb10 = src1->nb[0]; const int nb11 = src1->nb[1]; const int nb12 = src1->nb[2]; const int nb13 = src1->nb[3];
  const int    nb0  = dst->nb[0];  const int nb1  = dst->nb[1];  const int nb2  = dst->nb[2];  const int nb3  = dst->nb[3];
  const int    nr   = ne01*ne02*ne03;
  const int    ith  = params->ith;
  const int    nth  = params->nth;
  const int    nb   = ne00/QK;
  const size_t bs   = sizeof(float) + QK/2; /* 20B */

#if 0
  printf("nr=%d ne11=%d nb=%d ne0/nb0=%d/%d ne1/nb1=%d/%d ne00/nb00=%d/%d ne01/nb01=%d/%d ne10/nb10=%d/%d ne11/nb11=%d/%d\n", nr, ne11, nb, ne0, nb0, ne1, nb1, ne00, nb00, ne01, nb01, ne10, nb10, ne11, nb11);
  /*    1 nr=50288 ne11=5 nb=160 ne0/nb0=50288/4 ne1/nb1=5/201152 ne00/nb00= 5120/20 ne01/nb01=50288/ 3200 ne10/nb10= 5120/4 ne11/nb11=5/20480 */
  /*    1 nr=50288 ne11=8 nb=160 ne0/nb0=50288/4 ne1/nb1=8/201152 ne00/nb00= 5120/20 ne01/nb01=50288/ 3200 ne10/nb10= 5120/4 ne11/nb11=8/20480 */
  /*    1 nr=50288 ne11=9 nb=160 ne0/nb0=50288/4 ne1/nb1=9/201152 ne00/nb00= 5120/20 ne01/nb01=50288/ 3200 ne10/nb10= 5120/4 ne11/nb11=9/20480 */
  /*   36 nr=20480 ne11=5 nb=160 ne0/nb0=20480/4 ne1/nb1=5/ 81920 ne00/nb00= 5120/20 ne01/nb01=20480/ 3200 ne10/nb10= 5120/4 ne11/nb11=5/20480 */
  /*   36 nr=20480 ne11=8 nb=160 ne0/nb0=20480/4 ne1/nb1=8/ 81920 ne00/nb00= 5120/20 ne01/nb01=20480/ 3200 ne10/nb10= 5120/4 ne11/nb11=8/20480 */
  /*   36 nr=20480 ne11=9 nb=160 ne0/nb0=20480/4 ne1/nb1=9/ 81920 ne00/nb00= 5120/20 ne01/nb01=20480/ 3200 ne10/nb10= 5120/4 ne11/nb11=9/20480 */
  /*   36 nr= 5120 ne11=5 nb=640 ne0/nb0= 5120/4 ne1/nb1=5/ 20480 ne00/nb00=20480/20 ne01/nb01= 5120/12800 ne10/nb10=20480/4 ne11/nb11=5/81920 */
  /*   36 nr= 5120 ne11=8 nb=640 ne0/nb0= 5120/4 ne1/nb1=8/ 20480 ne00/nb00=20480/20 ne01/nb01= 5120/12800 ne10/nb10=20480/4 ne11/nb11=8/81920 */
  /*   36 nr= 5120 ne11=9 nb=640 ne0/nb0= 5120/4 ne1/nb1=9/ 20480 ne00/nb00=20480/20 ne01/nb01= 5120/12800 ne10/nb10=20480/4 ne11/nb11=9/81920 */
  /*   99 nr=50288 ne11=1 nb=160 ne0/nb0=50288/4 ne1/nb1=1/201152 ne00/nb00= 5120/20 ne01/nb01=50288/ 3200 ne10/nb10= 5120/4 ne11/nb11=1/20480 */
  /*  143 nr= 5120 ne11=8 nb=160 ne0/nb0= 5120/4 ne1/nb1=8/ 20480 ne00/nb00= 5120/20 ne01/nb01= 5120/ 3200 ne10/nb10= 5120/4 ne11/nb11=8/20480 */
  /*  143 nr= 5120 ne11=9 nb=160 ne0/nb0= 5120/4 ne1/nb1=9/ 20480 ne00/nb00= 5120/20 ne01/nb01= 5120/ 3200 ne10/nb10= 5120/4 ne11/nb11=9/20480 */
  /*  144 nr= 5120 ne11=5 nb=160 ne0/nb0= 5120/4 ne1/nb1=5/ 20480 ne00/nb00= 5120/20 ne01/nb01= 5120/ 3200 ne10/nb10= 5120/4 ne11/nb11=5/20480 */
  /* 3564 nr=20480 ne11=1 nb=160 ne0/nb0=20480/4 ne1/nb1=1/ 81920 ne00/nb00= 5120/20 ne01/nb01=20480/ 3200 ne10/nb10= 5120/4 ne11/nb11=1/20480 */
  /* 3564 nr= 5120 ne11=1 nb=640 ne0/nb0= 5120/4 ne1/nb1=1/ 20480 ne00/nb00=20480/20 ne01/nb01= 5120/12800 ne10/nb10=20480/4 ne11/nb11=1/81920 */
  /*14256 nr= 5120 ne11=1 nb=160 ne0/nb0= 5120/4 ne1/nb1=1/ 20480 ne00/nb00= 5120/20 ne01/nb01= 5120/ 3200 ne10/nb10= 5120/4 ne11/nb11=1/20480 */
#endif

/* #define EMAX7_BASELINE */
/* #define EMAX7_STEP1 */
/* #define EMAX7_STEP2 */
/* #define EMAX7_STEP3 */
/* #define EMAX7_STEP4 */
#define EMAX7_STEP4

#if !defined(EMAX7)
  // nb01 >= nb00 - src0 is not transposed. compute by src0 rows
  // rows per thread
  const int dr = (nr + nth - 1)/nth;
  // row range for this thread
  const int ir0 = dr*ith;
  const int ir1 = MIN(ir0 + dr, nr);
  void * wdata  = params->wdata;

  for (int ir = ir0; ir < ir1; ++ir) { /* 5120, 20480, 50288 */
    const int i03 = ir/(ne02*ne01);
    const int i02 = (ir - i03*ne02*ne01)/ne01;
    const int i01 = (ir - i03*ne02*ne01 - i02*ne01);
    const int i13 = i03;
    const int i12 = i02;
    const int i0  = i01;
    const int i2  = i02;
    const int i3  = i03;

    void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
    char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*GGML_TYPE_SIZE[GGML_TYPE_Q4_0])/GGML_BLCK_SIZE[GGML_TYPE_Q4_0]);
    float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

    for (int ic = 0; ic < ne11; ++ic) { /* 1, 5 */
      const uint8_t * restrict pd0 = ((const uint8_t *)src0_row + 0*bs);
      const uint8_t * restrict pd1 = ((const uint8_t *)(src1_col + (ic*ne00*GGML_TYPE_SIZE[GGML_TYPE_Q4_0])/GGML_BLCK_SIZE[GGML_TYPE_Q4_0]) + 0*bs);
      const uint8_t * restrict pb0 = ((const uint8_t *)src0_row + 0*bs + sizeof(float));
      const uint8_t * restrict pb1 = ((const uint8_t *)(src1_col + (ic*ne00*GGML_TYPE_SIZE[GGML_TYPE_Q4_0])/GGML_BLCK_SIZE[GGML_TYPE_Q4_0]) + 0*bs + sizeof(float));
      float sumf = 0.0;

      for (int i = 0; i < nb; i++) { /* 320, 1280 */
        const float d0 = *(const float *) (pd0 + i*bs);
        const float d1 = *(const float *) (pd1 + i*bs);
        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        for (int j = 0; j < QK/2; j++) { /* 16 */
          const uint8_t v0 = p0[j];
          const uint8_t v1 = p1[j];
          const float f0 = d0*((int8_t) (v0 & 0xf) - 8);
          const float f1 = d0*((int8_t) (v0 >> 4)  - 8);
          const float f2 = d1*((int8_t) (v1 & 0xf) - 8);
          const float f3 = d1*((int8_t) (v1 >> 4)  - 8);
          sumf += f0*f2 + f1*f3;
        }
      }
      dst_col[ic*ne0] = sumf;
    }
  }

#elif defined(EMAX7_BASELINE)
  if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
    exit(1);
  }
  if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
    exit(1);
  }
  for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
    const uint8_t * restrict sd = (const uint8_t *)((char *)src0->data + (ir*nb01));              /* nb01:     3200B/ir¡ú, 12800B/ir¡ü */
    float         *     dst_col = (float *)        ((char *) dst->data + (ir*nb0));               /* nb0:      4B                      */
    for (int ic = 0; ic < ne11; ic++) { /* 1,5,8,9 */
      const uint8_t * restrict wd = (const uint8_t *)((char *)params->wdata + ic*nb*nb00); /* nb*nb00:3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
      float sumf = 0.0;
      /* src0->data   sd[float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b] */
      /* param->wdata wd[float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b] */
      for (int i = 0; i < nb; i++) { /* 160, 640 */
        const float         * sdf32 = (const float *) (sd + i*bs);                 /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
        const float         * wdf32 = (const float *) (wd + i*bs);                 /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
        const uint8_t * restrict s0 = (const uint8_t*)(sd + i*bs + sizeof(float)); /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
        const uint8_t * restrict w0 = (const uint8_t*)(wd + i*bs + sizeof(float)); /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
        /*          0  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12 12 13 13 14 15 15 15 16 16 17 17 18 18 19 19 */
        /* sd:  float,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b */
        /* sdata  ^sdf32 ^s0                                                                                             */
        /*            -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */
        /*            lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi */
        /* wd:  float,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b */
        /* wdata  ^wdf32 ^w0                                                                                             */
        /*            -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */
        /*            lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi */
        /*             *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * */
        /*      sumf ¦² +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +   */
        for (int j = 0; j < QK/2; j++) { /* 16 */
          const float slo = *sdf32 * ((int8_t)(s0[j] & 0xf) - 8); const float shi = *sdf32 * ((int8_t)(s0[j] >> 4)  - 8);
          const float wlo = *wdf32 * ((int8_t)(w0[j] & 0xf) - 8); const float whi = *wdf32 * ((int8_t)(w0[j] >> 4)  - 8);
          sumf += slo*wlo + shi*whi;
        }
      }
      dst_col[ic*ne0] = sumf; /* icËè¤Ë, ne0:5120W, 20480W, 50288WÈô¤Ó¢£ ºÇ³°ir¤Ç4BËè¤Ë¥¹¥È¥¢ */
    }
  }

#elif defined(EMAX7_STEP1)
  if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
    exit(1);
  }
  if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
    exit(1);
  }
  /* output: Hi there, how are you doing? I am Open Assistant and here to help... */
  /*                   <|BEGIN>  50278  12092  2  0  50281  12764  627  13  849  403  368  2509  32 ... <END|> */
  /* output: Hi there! <|BEGIN>  50278  12092  2  0  50281  12764  627   2                              <END|> */
  Ull   CHIP, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   NBNB00        = (Ull)(nb*nb00);
  Ull   NBNB00d4      = NBNB00/sizeof(int);      /* max word length of s0 */
  Ull   NBNB00xNE11d4 = NBNB00*ne11/sizeof(int); /* max word length of w0 */
  Ull   MNBNB00_MNE0  = (0LL- NBNB00)<<32|((0LL-(Ull)ne0)&0xffffffffLL);
  Ull   NBNB00_NE0    = (     NBNB00)<<32|((    (Ull)ne0)&0xffffffffLL);
  Ull   MBS           = (0LL-(Ull)bs)<<32|((0LL-(Ull)0LL)&0xffffffffLL);
  Ull   BS            = (    (Ull)bs)<<32|((    (Ull)0LL)&0xffffffffLL);

#undef  NCHIP
#define NCHIP 1

  for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
    const uint8_t * restrict sd   = (const uint8_t *)((char *)src0->data + (ir*nb01)); /* nb01: 3200B/ir¡ú, 12800B/ir¡ü */
    const uint8_t * restrict wd   = (const uint8_t *)((char *)params->wdata);
    const uint8_t * restrict sdp1 = sd   + sizeof(float);
    const uint8_t * restrict wdp1 = wd   + sizeof(float);
    const uint8_t * restrict sdp3 = sdp1 + sizeof(Ull);
    const uint8_t * restrict wdp3 = wdp1 + sizeof(Ull);
    const uint8_t * restrict sdp5 = sdp3 + sizeof(Ull);
    const uint8_t * restrict wdp5 = wdp3 + sizeof(Ull);
    float         *       dst_col = (float *)        ((char *)dst->data  + (ir*nb0));  /* nb0:  4B                      */

#define sx(s,p) ((int8_t)((s)>>(p*4)&15)-8)

//EMAX5A begin mul_mat_q4_0_f32 mapdist=0
/**/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
 /*2*/for (INIT1=1,LOOP1=ne11,rofs=MNBNB00_MNE0; LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
        float sum = 0.0;
   /*1*/for (INIT0=1,LOOP0=nb,cofs=MBS; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD,    &cofs, INIT0?cofs:cofs,  EXP_H3210, BS,      EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,    &rofs, rofs,  EXP_H3210, INIT0?NBNB00_NE0:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,    &iofs, rofs,  EXP_H3210, cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
          exe(OP_ADD,    &oofs, rofs,  EXP_H3210, cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

	  mop(OP_LDWR,   1, &BR[2][0][1],  sd,    cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      
	  mop(OP_LDR,    1, &BR[2][2][1],  sdp3,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load (no use)   */
	  mop(OP_LDR,    1, &BR[2][2][0],  sdp1,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load 64bit low  */
	  mop(OP_LDR,    1, &BR[2][3][1],  sdp5,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load (no use)   */
	  mop(OP_LDR,    1, &BR[2][3][0],  sdp3,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load 64bit high */

	  mop(OP_LDWR,   1, &BR[3][0][1],  wd,    iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); 
	  mop(OP_LDR,    1, &BR[3][2][1],  wdp3,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load (no use)   */
	  mop(OP_LDR,    1, &BR[3][2][0],  wdp1,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load 64bit low  */
	  mop(OP_LDR,    1, &BR[3][3][1],  wdp5,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load (no use)   */
	  mop(OP_LDR,    1, &BR[3][3][0],  wdp3,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load 64bit high */

          float  sf = *(float*)&BR[2][0][1]; /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
          Ull    s0 = BR[2][2][0];           /* unaligned 64bit low  */
          Ull    s1 = BR[2][3][0];           /* unaligned 64bit high */
          float  wf = *(float*)&BR[3][0][1]; /* min:160*20=3200B¡ú, max:640*20=12800B¡ü *//* rofs NBNB00(nb*nb00):3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
          Ull    w0 = BR[3][2][0];           /* unaligned 64bit low  */
          Ull    w1 = BR[3][3][0];           /* unaligned 64bit high */

	  /* sf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML sd  = sf * s0[i]    */
	  /* s0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ s1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  */
	  /* wf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML wd  = wf * w0[i]    */
	  /* w0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ w1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /* ¦²+ + + + + + + + + + + + + + + +    + + + + + + + + + + + + + + + +  *//* FMA sum = sum + sd * wd */
          sum += (sf*sx(s0, 0))*(wf*sx(w0, 0))  /* FML  FML      */
	      +  (sf*sx(s0, 1))*(wf*sx(w0, 1))  /* FML  FML  FMA */
              +  (sf*sx(s0, 2))*(wf*sx(w0, 2))  /* FML  FML  FMA */
	      +  (sf*sx(s0, 3))*(wf*sx(w0, 3))  /* FML  FML  FMA */
              +  (sf*sx(s0, 4))*(wf*sx(w0, 4))  /* FML  FML  FMA */
	      +  (sf*sx(s0, 5))*(wf*sx(w0, 5))  /* FML  FML  FMA */
              +  (sf*sx(s0, 6))*(wf*sx(w0, 6))  /* FML  FML  FMA */
	      +  (sf*sx(s0, 7))*(wf*sx(w0, 7))  /* FML  FML  FMA */
              +  (sf*sx(s0, 8))*(wf*sx(w0, 8))  /* FML  FML  FMA */
	      +  (sf*sx(s0, 9))*(wf*sx(w0, 9))  /* FML  FML  FMA */
              +  (sf*sx(s0,10))*(wf*sx(w0,10))  /* FML  FML  FMA */
	      +  (sf*sx(s0,11))*(wf*sx(w0,11))  /* FML  FML  FMA */
              +  (sf*sx(s0,12))*(wf*sx(w0,12))  /* FML  FML  FMA */
	      +  (sf*sx(s0,13))*(wf*sx(w0,13))  /* FML  FML  FMA */
              +  (sf*sx(s0,14))*(wf*sx(w0,14))  /* FML  FML  FMA */
	      +  (sf*sx(s0,15))*(wf*sx(w0,15))  /* FML  FML  FMA */
              +  (sf*sx(s1, 0))*(wf*sx(w1, 0))  /* FML  FML  FMA */
	      +  (sf*sx(s1, 1))*(wf*sx(w1, 1))  /* FML  FML  FMA */
              +  (sf*sx(s1, 2))*(wf*sx(w1, 2))  /* FML  FML  FMA */
	      +  (sf*sx(s1, 3))*(wf*sx(w1, 3))  /* FML  FML  FMA */
              +  (sf*sx(s1, 4))*(wf*sx(w1, 4))  /* FML  FML  FMA */
	      +  (sf*sx(s1, 5))*(wf*sx(w1, 5))  /* FML  FML  FMA */
              +  (sf*sx(s1, 6))*(wf*sx(w1, 6))  /* FML  FML  FMA */
	      +  (sf*sx(s1, 7))*(wf*sx(w1, 7))  /* FML  FML  FMA */
              +  (sf*sx(s1, 8))*(wf*sx(w1, 8))  /* FML  FML  FMA */
	      +  (sf*sx(s1, 9))*(wf*sx(w1, 9))  /* FML  FML  FMA */
              +  (sf*sx(s1,10))*(wf*sx(w1,10))  /* FML  FML  FMA */
	      +  (sf*sx(s1,11))*(wf*sx(w1,11))  /* FML  FML  FMA */
              +  (sf*sx(s1,12))*(wf*sx(w1,12))  /* FML  FML  FMA */
	      +  (sf*sx(s1,13))*(wf*sx(w1,13))  /* FML  FML  FMA */
              +  (sf*sx(s1,14))*(wf*sx(w1,14))  /* FML  FML  FMA */
	      +  (sf*sx(s1,15))*(wf*sx(w1,15)); /* FML  FML  FMA */
        }                                       /*           FMA */
        dst_col[oofs] = sum; /* icËè¤Ë, ne0:5120W, 20480W, 50288WÈô¤Ó¢£ ºÇ³°ir¤Ç4BËè¤Ë¥¹¥È¥¢ */
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm

#elif defined(EMAX7_STEP2)
  if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
    exit(1);
  }
  if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
    exit(1);
  }
  /* output: Hi there, how are you doing? I am Open Assistant and here to help... */
  /*                   <|BEGIN>  50278  12092  2  0  50281  12764  627  13  849  403  368  2509  32 ... <END|> */
  /* output: Hi there! <|BEGIN>  50278  12092  2  0  50281  12764  627   2                              <END|> */
  Ull   CHIP, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   NBNB00        = (Ull)(nb*nb00);
  Ull   NBNB00d4      = NBNB00/sizeof(int);      /* max word length of s0 */
  Ull   NBNB00xNE11d4 = NBNB00*ne11/sizeof(int); /* max word length of w0 */
  Ull   MNBNB00_MNE0  = (0LL- NBNB00)<<32|((0LL-(Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   NBNB00_NE0    = (     NBNB00)<<32|((    (Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   MBS           = (0LL-(Ull)bs)<<32|((0LL-(Ull)0LL)&0xffffffffLL);
  Ull   BS            = (    (Ull)bs)<<32|((    (Ull)0LL)&0xffffffffLL);
  Ull   NE01NE11      = ne01*ne11;

#undef  NCHIP
#define NCHIP 1

  /* clear destinaton */
  int i;
  float *dst_col = (float *)((char *)dst->data);
  for (int i = 0; i < NE01NE11; i++) /* ir:nr=ne01, rofs=ne0Èô¤Ó*ne11 */
    ((float *)(char *)dst->data)[i] = 0;

  for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
    const uint8_t * restrict sd   = (const uint8_t *)((char *)src0->data + (ir*nb01)); /* nb01: 3200B/ir¡ú, 12800B/ir¡ü */
    const uint8_t * restrict wd   = (const uint8_t *)((char *)params->wdata);
    const uint8_t * restrict sdp1 = sd   + sizeof(float);
    const uint8_t * restrict wdp1 = wd   + sizeof(float);
    const uint8_t * restrict sdp3 = sdp1 + sizeof(Ull);
    const uint8_t * restrict wdp3 = wdp1 + sizeof(Ull);
    const uint8_t * restrict sdp5 = sdp3 + sizeof(Ull);
    const uint8_t * restrict wdp5 = wdp3 + sizeof(Ull);
    float         *       dst_col = (float *)        ((char *)dst->data  + (ir*nb0));  /* nb0:  4B                      */
    float convi4f32[16] = {-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

#define mul_mat_core0(r, c, d0, d1, d2, d3) \
	  exe(OP_NOP,      &r0,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000000f0000000fLL,  OP_SLL,  2LL);\
	  exe(OP_NOP,      &r1,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x000000f0000000f0LL,  OP_SRL,  2LL);\
	  exe(OP_NOP,      &r2,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000f0000000f00LL,  OP_SRL,  6LL);\
	  exe(OP_NOP,      &r3,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000f0000000f000LL,  OP_SRL, 10LL);\
	  mop(OP_LDWR,  1, &r4,  convi4f32,   r0, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r8,  convi4f32,   r0, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r5,  convi4f32,   r1, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r9,  convi4f32,   r1, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r6,  convi4f32,   r2, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r10, convi4f32,   r2, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r7,  convi4f32,   r3, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r11, convi4f32,   r3, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  exe(OP_CMOV,     &r12, 0x0000000100000000LL, EXP_H3210, r5,  EXP_H1010, r4,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r13, 0x0000000100000000LL, EXP_H3210, r7,  EXP_H1010, r6,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r14, 0x0000000100000000LL, EXP_H3210, r9,  EXP_H1010, r8,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r15, 0x0000000100000000LL, EXP_H3210, r11, EXP_H1010, r10, EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d0, BR[r][0][1], EXP_H1010, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d1, BR[r][0][1], EXP_H1010, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d2, BR[r][0][1], EXP_H1010, r14, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d3, BR[r][0][1], EXP_H1010, r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define mul_mat_core1(r, c, d0, d1, d2, d3) \
	  exe(OP_NOP,      &r0,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x000f0000000f0000LL,  OP_SRL, 14LL);\
	  exe(OP_NOP,      &r1,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00f0000000f00000LL,  OP_SRL, 18LL);\
	  exe(OP_NOP,      &r2,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0f0000000f000000LL,  OP_SRL, 22LL);\
	  exe(OP_NOP,      &r3,  BR[r][c][0], EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xf0000000f0000000LL,  OP_SRL, 26LL);\
	  mop(OP_LDWR,  1, &r4,  convi4f32,   r0, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r8,  convi4f32,   r0, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r5,  convi4f32,   r1, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r9,  convi4f32,   r1, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r6,  convi4f32,   r2, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r10, convi4f32,   r2, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r7,  convi4f32,   r3, MSK_B0,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r11, convi4f32,   r3, MSK_B4,     convi4f32, 16,  0, 0, (Ull)NULL, 16);\
	  exe(OP_CMOV,     &r12, 0x0000000100000000LL, EXP_H3210, r5,  EXP_H1010, r4,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r13, 0x0000000100000000LL, EXP_H3210, r7,  EXP_H1010, r6,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r14, 0x0000000100000000LL, EXP_H3210, r9,  EXP_H1010, r8,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r15, 0x0000000100000000LL, EXP_H3210, r11, EXP_H1010, r10, EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d0, BR[r][0][1], EXP_H1010, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d1, BR[r][0][1], EXP_H1010, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d2, BR[r][0][1], EXP_H1010, r14, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d3, BR[r][0][1], EXP_H1010, r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

//EMAX5A begin mul_mat_q4_0_f32 mapdist=0
/**/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
 /*2*/for (INIT1=1,LOOP1=ne11,rofs=MNBNB00_MNE0; LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
   /*1*/for (INIT0=1,LOOP0=nb,cofs=MBS; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD,      &cofs,  INIT0?cofs:cofs,     EXP_H3210, BS,      EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,      &rofs,  rofs,   EXP_H3210,   INIT0?NBNB00_NE0:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,      &iofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
          exe(OP_ADD,      &oofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

	  mop(OP_LDWR,  1, &BR[2][0][1],   sd,    cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);                                      /* stage #2 */
	  mop(OP_LDR,   1, &BR[2][2][1],   sdp3,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load (no use)   */ /* stage #2 */
	  mop(OP_LDR,   1, &BR[2][2][0],   sdp1,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load 64bit low  */ /* stage #2 */
	  mop(OP_LDR,   1, &BR[2][3][1],   sdp5,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load (no use)   */ /* stage #2 */
	  mop(OP_LDR,   1, &BR[2][3][0],   sdp3,  cofs, MSK_W1, (Ull)sd, NBNB00d4,      0, 0, (Ull)NULL, NBNB00d4);      /* unaligned load 64bit high */ /* stage #2 */

	  mop(OP_LDWR,  1, &BR[3][0][1],   wd,    iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4);                                 /* stage #3 */
	  mop(OP_LDR,   1, &BR[3][2][1],   wdp3,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load (no use)   */ /* stage #3 */
	  mop(OP_LDR,   1, &BR[3][2][0],   wdp1,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load 64bit low  */ /* stage #3 */
	  mop(OP_LDR,   1, &BR[3][3][1],   wdp5,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load (no use)   */ /* stage #3 */
	  mop(OP_LDR,   1, &BR[3][3][0],   wdp3,  iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4); /* unaligned load 64bit high */ /* stage #3 */
	  exe(OP_NOP,      &AR[3][0], 0LL, EXP_H3210,   0LL,                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#4 (dummy to set target location) */

	  /* sf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML sd  = sf * s0[i]    */
	  /* s0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ s1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  */
	  /* wf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML wd  = wf * w0[i]    */
	  /* w0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ w1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /* ¦²+ + + + + + + + + + + + + + + +    + + + + + + + + + + + + + + + +  *//* FMA sum = sum + sd * wd */
          /*float  sf = *(float*)&BR[2][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
          /*Ull    s0 = BR[2][2][0];           *//* unaligned 64bit low  */
          /*Ull    s1 = BR[2][3][0];           *//* unaligned 64bit high */
          /*float  wf = *(float*)&BR[3][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü *//* rofs NBNB00(nb*nb00):3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
          /*Ull    w0 = BR[3][2][0];           *//* unaligned 64bit low  */
          /*Ull    w1 = BR[3][3][0];           *//* unaligned 64bit high */

	  /* s0 low */
	  mul_mat_core0(2, 2, r16, r17, r18, r19); /* stage #4[4],#5[8],#6[4],#7[4] */
	  /* w0 low */
	  mul_mat_core0(3, 2, r20, r21, r22, r23); /* stage #8[4],#9[8],#10[4],#11[4] */
	  /* FMA0low */
	  exe(OP_FML,      &r24, r16,         EXP_H3210, r20, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r25, r17,         EXP_H3210, r21, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r26, r18,         EXP_H3210, r22, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r27, r19,         EXP_H3210, r23, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */

	  /* s0 high */
	  mul_mat_core1(2, 2, r16, r17, r18, r19); /* stage #13[4],#14[8],#15[4],#16[4] */
	  /* w0 high */
	  mul_mat_core1(3, 2, r20, r21, r22, r23); /* stage #17 XXX */
	  /* FMA0high */
	  exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  /* s1 low */
	  mul_mat_core0(2, 3, r16, r17, r18, r19);
	  /* w1 low */
	  mul_mat_core0(3, 3, r20, r21, r22, r23);
	  /* FMA1low */
	  exe(OP_FMA,      &r24, r28,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r25, r29,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r26, r30,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r27, r31,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  /* s1 high */
	  mul_mat_core1(2, 3, r16, r17, r18, r19);
	  /* w1 high */
	  mul_mat_core1(3, 3, r20, r21, r22, r23);
	  /* FMA1high */
	  exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  /* FAD tree */
	  exe(OP_FAD,      &r3,  r28,         EXP_H3210, r29, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FAD,      &r4,  r30,         EXP_H3210, r31, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  exe(OP_FAD,      &r2,  r3,          EXP_H3210, r4,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  exe(OP_FAD,      &r1,  r2,          EXP_H3232, r2,  EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  mop(OP_LDWR,  1, &r0,  dst_col,     oofs,      MSK_W0, i_m0C[LANE], NE01NE11,  0, 1,   (Ull)NULL,   NE01NE11);
	  exe(OP_FAD,      &r0,  INIT0?r0:r0, EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  mop(OP_STWR,  1, &r0,  oofs,        dst_col,   MSK_D0, i_m0C[LANE], NE01NE11,  0, 1,   (Ull)NULL,   NE01NE11);
        }
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm

#elif defined(EMAX7_STEP3)
  if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
    exit(1);
  }
  if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
    exit(1);
  }
  /* output: Hi there, how are you doing? I am Open Assistant and here to help... */
  /*                    <|BEGIN>  50278  12092  2  0  50281  12764  627  13  849  403  368  2509  32 ... <END|> */
  /* output: Hi there!  <|BEGIN>  50278  12092  2  0  50281  12764  627   2                              <END|> */
  /* output: Hey there! <|BEGIN>  50278  12092  2  0  50281   8262  627   2                              <END|> */
  Ull   CHIP, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   NBNB00        = (Ull)(nb*nb00);
  Ull   NBNB00d4      = NBNB00/sizeof(int);      /* max word length of s0 */
  Ull   NBNB00xNE11d4 = NBNB00*ne11/sizeof(int); /* max word length of w0 */
  Ull   MNBNB00_MNE0  = (0LL- NBNB00)<<32|((0LL-(Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   NBNB00_NE0    = (     NBNB00)<<32|((    (Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   MBS           = (0LL-(Ull)bs)<<32|((0LL-(Ull)0LL)&0xffffffffLL);
  Ull   BS            = (    (Ull)bs)<<32|((    (Ull)0LL)&0xffffffffLL);
  Ull   NE01NE11      = ne01*ne11;

#undef  NCHIP
#define NCHIP 1

  /* clear destinaton */
  int i;
  float *dst_col = (float *)((char *)dst->data);
  for (int i = 0; i < NE01NE11; i++) /* ir:nr=ne01, rofs=ne0Èô¤Ó*ne11 */
    ((float *)(char *)dst->data)[i] = 0;

  for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
    const uint8_t * restrict sd   = (const uint8_t *)((char *)src0->data + (ir*nb01)); /* nb01: 3200B/ir¡ú, 12800B/ir¡ü */
    const uint8_t * restrict wd   = (const uint8_t *)((char *)params->wdata);
    const uint8_t * restrict sdp[4];
    const uint8_t * restrict wdp[4];
    sdp[0] = sd   + sizeof(float)*1;
    wdp[0] = wd   + sizeof(float)*1;
    sdp[1] = sd   + sizeof(float)*2;
    wdp[1] = wd   + sizeof(float)*2;
    sdp[2] = sd   + sizeof(float)*3;
    wdp[2] = wd   + sizeof(float)*3;
    sdp[3] = sd   + sizeof(float)*4;
    wdp[3] = wd   + sizeof(float)*4;
    float         *       dst_col = (float *)        ((char *)dst->data  + (ir*nb0));  /* nb0:  4B                      */
    float convi4f32[16] = {-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

#define mul_mat_cores(r, c, d0, d1, d2, d3) \
	  mop(OP_LDWR,  1, &BR[r][0][1], sd,           cofs, MSK_W1,   (Ull)sd, NBNB00d4, 0, 0, (Ull)NULL, NBNB00d4);\
	  mop(OP_LDWR,  1, &BR[r][2][1], sdp[c],       cofs, MSK_W1,   (Ull)sd, NBNB00d4, 0, 0, (Ull)NULL, NBNB00d4);\
	  exe(OP_NOP,      &r0,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000000f000fLL,  OP_SLL,  2LL);\
	  exe(OP_NOP,      &r1,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000000000f000f0LL,  OP_SRL,  2LL);\
	  exe(OP_NOP,      &r2,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x000000000f000f00LL,  OP_SRL,  6LL);\
	  exe(OP_NOP,      &r3,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000f000f000LL,  OP_SRL, 10LL);\
	  mop(OP_LDWR,  1, &r4,          convi4f32,    r0,   MSK_B0,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r8,          convi4f32,    r0,   MSK_B2,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r5,          convi4f32,    r1,   MSK_B0,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r9,          convi4f32,    r1,   MSK_B2,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r6,          convi4f32,    r2,   MSK_B0,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r10,         convi4f32,    r2,   MSK_B2,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r7,          convi4f32,    r3,   MSK_B0,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r11,         convi4f32,    r3,   MSK_B2,   convi4f32, 16,     0, 0, (Ull)NULL, 16);\
	  exe(OP_CMOV,     &r12, 0x0000000100000000LL, EXP_H3210, r5,  EXP_H1010, r4,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r13, 0x0000000100000000LL, EXP_H3210, r7,  EXP_H1010, r6,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r14, 0x0000000100000000LL, EXP_H3210, r9,  EXP_H1010, r8,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r15, 0x0000000100000000LL, EXP_H3210, r11, EXP_H1010, r10, EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d0,          BR[r][0][1],  EXP_H1010, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d1,          BR[r][0][1],  EXP_H1010, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d2,          BR[r][0][1],  EXP_H1010, r14, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d3,          BR[r][0][1],  EXP_H1010, r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define mul_mat_corew(r, c, d0, d1, d2, d3) \
	  mop(OP_LDWR,  1, &BR[r][0][1], wd,           iofs, MSK_W1,   (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4);\
	  mop(OP_LDWR,  1, &BR[r][2][1], wdp[c],       iofs, MSK_W1,   (Ull)wd, NBNB00xNE11d4, 0, 0, (Ull)NULL, NBNB00xNE11d4);\
	  exe(OP_NOP,      &r0,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000000f000fLL,  OP_SLL,  2LL);\
	  exe(OP_NOP,      &r1,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000000000f000f0LL,  OP_SRL,  2LL);\
	  exe(OP_NOP,      &r2,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x000000000f000f00LL,  OP_SRL,  6LL);\
	  exe(OP_NOP,      &r3,          BR[r][2][1],  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000f000f000LL,  OP_SRL, 10LL);\
	  mop(OP_LDWR,  1, &r4,          convi4f32,    r0,   MSK_B0,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r8,          convi4f32,    r0,   MSK_B2,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r5,          convi4f32,    r1,   MSK_B0,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r9,          convi4f32,    r1,   MSK_B2,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r6,          convi4f32,    r2,   MSK_B0,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r10,         convi4f32,    r2,   MSK_B2,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r7,          convi4f32,    r3,   MSK_B0,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  mop(OP_LDWR,  1, &r11,         convi4f32,    r3,   MSK_B2,   convi4f32, 16,          0, 0, (Ull)NULL, 16);\
	  exe(OP_CMOV,     &r12, 0x0000000100000000LL, EXP_H3210, r5,  EXP_H1010, r4,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r13, 0x0000000100000000LL, EXP_H3210, r7,  EXP_H1010, r6,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r14, 0x0000000100000000LL, EXP_H3210, r9,  EXP_H1010, r8,  EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_CMOV,     &r15, 0x0000000100000000LL, EXP_H3210, r11, EXP_H1010, r10, EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d0,          BR[r][0][1],  EXP_H1010, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d1,          BR[r][0][1],  EXP_H1010, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d2,          BR[r][0][1],  EXP_H1010, r14, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML,      &d3,          BR[r][0][1],  EXP_H1010, r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

//EMAX5A begin mul_mat_q4_0_f32 mapdist=0
/**/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
 /*2*/for (INIT1=1,LOOP1=ne11,rofs=MNBNB00_MNE0; LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
   /*1*/for (INIT0=1,LOOP0=nb,cofs=MBS; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD,      &cofs,  INIT0?cofs:cofs,     EXP_H3210, BS,      EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,      &rofs,  rofs,   EXP_H3210,   INIT0?NBNB00_NE0:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
          exe(OP_ADD,      &iofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
          exe(OP_ADD,      &oofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

	  /* sf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML sd  = sf * s0[i]    */
	  /* s0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ s1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  */
	  /* wf¢¢                                                                  */
	  /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML wd  = wf * w0[i]    */
	  /* w0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ w1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	  /* ¦²+ + + + + + + + + + + + + + + +    + + + + + + + + + + + + + + + +  *//* FMA sum = sum + sd * wd */
          /*float  sf = *(float*)&BR[2][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
          /*Ull    s0 = BR[2][2][0];           *//* unaligned 64bit low  */
          /*Ull    s1 = BR[2][3][0];           *//* unaligned 64bit high */
          /*float  wf = *(float*)&BR[3][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü *//* rofs NBNB00(nb*nb00):3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
          /*Ull    w0 = BR[3][2][0];           *//* unaligned 64bit low  */
          /*Ull    w1 = BR[3][3][0];           *//* unaligned 64bit high */

	  /* 1/4                                                                                                               */
	  /* LDWR(sf:f32)   LDWR(s0:i4x8)                                    W W                    cofs iofs           4  #2  */
	  /* EXTRACT-4bit         x w4                                       |   BB  BB  BB  BB     cofs iofs           7  #3  */
	  /* LDWR base+4bit       x w8                                       |   W W W W W W W W    cofs iofs 00010000 12  #4  */
	  /* CMOV concat 8->4     x d4                                       |   WW  WW  WW  WW     cofs iofs           7  #5  */
	  /* FMUL sf(f32)*s0(f32) x w8                                       ~~~ 16  17  18  19     cofs iofs           6  #6  */
	  /*                                                   16  17  18  19                                                  */
	  /* LDWR(wf:f32)   LDWR(w0:i4x8)                      |   |   |   | W W                    cofs iofs           8  #7  */
	  /* EXTRACT-4bit         x w4                         |   |   |   | |   BB  BB  BB  BB     cofs iofs          11  #8  */
	  /* LDWR base+4bit       x w8                         |   |   |   | |   W W W W W W W W    cofs iofs 00010000 16  #9  */
	  /* CMOV concat 8->4     x d4                         |   |   |   | |   WW  WW  WW  WW     cofs iofs          11  #10 */
	  /* FMUL wf(f32)*s0(w32) x w8                         |   |   |   | ~~~ 20  21  22  23     cofs iofs          10  #11 */
	  /* FMUL sx(f32)*wx(w32) x w8         24  25  26  27                                       cofs iofs           6  #12 */
          /*                                   |   |   |   |                                                                   */
	  /* 2/4                               |   |   |   |                                                                   */
	  /* LDWR(sf:f32)   LDWR(s1:i4x8)      |   |   |   |                 W W                    cofs iofs           8  #13 */
	  /* EXTRACT-4bit         x w4         |   |   |   |                 |   BB  BB  BB  BB     cofs iofs          11  #14 */
	  /* LDWR base+4bit       x w8         |   |   |   |                 |   W W W W W W W W    cofs iofs 00010000 16  #15 */
	  /* CMOV concat 8->4     x d4         |   |   |   |                 |   WW  WW  WW  WW     cofs iofs          11  #16 */
	  /* FMUL sf(f32)*s1(f32) x w8         |   |   |   |                 ~~~ 16  17  18  19     cofs iofs          10  #17 */
          /*                                   |   |   |   |   16  17  18  19                                                  */
	  /* LDWR(wf:f32)   LDWR(w1:i4x8)      |   |   |   |   |   |   |   | W W                    cofs iofs          12  #18 */
	  /* EXTRACT-4bit         x w4         |   |   |   |   |   |   |   | |   BB  BB  BB  BB     cofs iofs          15  #19 */
	  /* LDWR base+4bit       x w8         |   |   |   |   |   |   |   | |   W W W W W W W W    cofs iofs 00010000 20* #20 */
	  /* CMOV concat 8->4     x d4         |   |   |   |   |   |   |   | |   WW  WW  WW  WW     cofs iofs          15  #21 */
	  /* FMUL wf(f32)*s1(w32) x w8         |   |   |   |   |   |   |   | ~~~ 20  21  22  23     cofs iofs          14  #22 */
	  /* FMUL sx(f32)*wx(w32) x w8         28  29  30  31                                       cofs iofs           6  #23 */

	  /*exe(OP_NOP,      &AR[3][0], 0LL, EXP_H3210,   0LL,                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); *//* stage#4 (dummy to set target location) */

	  mul_mat_cores(2, 0, r16, r17, r18, r19); /* stage #2-#6  */
	  mul_mat_corew(7, 0, r20, r21, r22, r23); /* stage #7-#11 */
	  exe(OP_FML,      &r24, r16,         EXP_H3210, r20, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r25, r17,         EXP_H3210, r21, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r26, r18,         EXP_H3210, r22, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FML,      &r27, r19,         EXP_H3210, r23, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */

	  mul_mat_cores(13, 1, r16, r17, r18, r19); /* stage #13-#17 */
	  mul_mat_corew(18, 1, r20, r21, r22, r23); /* stage #18-#22 */
	  exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 XXX r27 has no path */

	  mul_mat_cores(24, 2, r16, r17, r18, r19); /* stage #24-#28 */
	  mul_mat_corew(29, 2, r20, r21, r22, r23); /* stage #29-#33 */
	  exe(OP_FMA,      &r24, r28,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA,      &r25, r29,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA,      &r26, r30,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA,      &r27, r31,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */

	  mul_mat_cores(35, 3, r16, r17, r18, r19); /* stage #35-#39 */
	  mul_mat_corew(40, 3, r20, r21, r22, r23); /* stage #40-#44 */
	  exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */

	  /* FAD tree */
	  exe(OP_FAD,      &r3,  r28,         EXP_H3210, r29, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  exe(OP_FAD,      &r4,  r30,         EXP_H3210, r31, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  exe(OP_FAD,      &r2,  r3,          EXP_H3210, r4,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  exe(OP_FAD,      &r1,  r2,          EXP_H3232, r2,  EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

	  mop(OP_LDWR,  1, &r0,  dst_col,     oofs,      MSK_W0, i_m0C[LANE], NE01NE11,  0, 1,   (Ull)NULL,   NE01NE11);
	  exe(OP_FAD,      &r0,  INIT0?r0:r0, EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	  mop(OP_STWR,  1, &r0,  oofs,        dst_col,   MSK_D0, i_m0C[LANE], NE01NE11,  0, 1,   (Ull)NULL,   NE01NE11);
        }
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm

#elif defined(EMAX7_STEP4)
  if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
    exit(1);
  }
  if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
    printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
    exit(1);
  }
  /* output: Hi there, how are you doing? I am Open Assistant and here to help... */
  /*                    <|BEGIN>  50278  12092  2  0  50281  12764  627  13  849  403  368  2509  32 ... <END|> */
  /* output: Hi there!  <|BEGIN>  50278  12092  2  0  50281  12764  627   2                              <END|> */
  /* output: Hey there! <|BEGIN>  50278  12092  2  0  50281   8262  627   2                              <END|> */
  Ull   CHIP, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   NRNB01d4      = nr*nb01/sizeof(Uint);    /* 50288*3200B/4:40230400 (160MB) max total words of sd */
  Ull   NBNB00        = (Ull)(nb*nb00);          /* 160 * 20B                                            */
  Ull   NBNB00d4      = NBNB00/sizeof(int);      /* 160 * 20B /4  :     800  (3KB) max LMM words of sd   */
  Ull   NBNB00xNE11d4 = NBNB00*ne11/sizeof(int); /* 160 * 20B /4*5:    4000 (16KB) max LMM words of wd   */
  Ull   MNBNB00_MNE0  = (0LL- NBNB00)<<32|((0LL-(Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   NBNB00_NE0    = (     NBNB00)<<32|((    (Ull)ne0*sizeof(int))&0xffffffffLL);
  Ull   MBS           = (0LL-(Ull)bs)<<32|((0LL-(Ull)0LL)&0xffffffffLL);
  Ull   BS            = (    (Ull)bs)<<32|((    (Ull)0LL)&0xffffffffLL);
  Ull   NE01NE11      = ne01*ne11;               /* 50288 * 5    :  251440 (1MB)   max LMM words of dst_col */
  Ull   Force         = 1; /* force wdat load to LMM */

static int nrnb01d4;
static int nbnb00d4;
static int nbnb00xne11d4;
static int ne01ne11;
static int updated;
static int check_lmm;
static int check_lmm_ovf;
static int check_lmm_fit;

#if 0
  if (nrnb01d4      < NRNB01d4)      { nrnb01d4      = NRNB01d4;      updated = 1;}
  if (nbnb00d4      < NBNB00d4)      { nbnb00d4      = NBNB00d4;      updated = 1;}
  if (nbnb00xne11d4 < NBNB00xNE11d4) { nbnb00xne11d4 = NBNB00xNE11d4; updated = 1;}
  if (ne01ne11      < NE01NE11)      { ne01ne11      = NE01NE11;      updated = 1;}
  if (updated) { printf("max sdat_cpyin=%d sdat_lmmwords=%d wdat_lmmwords=%d dst_lmmwords=%d\n", (int)nrnb01d4, (int)nbnb00d4, (int)nbnb00xne11d4, (int)ne01ne11); updated = 0;}
  if ((check_lmm++ & 0xff) == 0) { printf("lmm_ovf=%d lmm_fit=%d\n", check_lmm_ovf, check_lmm_fit);}
  /* 499 @   sdat_cpyin= 4096000 sdat_lmmwords= 800 wdat_lmmwords=  800 dst_lmmwords=  5120 */
  /* 287 @   sdat_cpyin= 4096000 sdat_lmmwords= 800 wdat_lmmwords= 4000 dst_lmmwords= 25600 */
  /* 125 @   sdat_cpyin=16384000 sdat_lmmwords= 800 wdat_lmmwords=  800 dst_lmmwords= 20480 */
  /* 125 @   sdat_cpyin=16384000 sdat_lmmwords=3200 wdat_lmmwords= 3200 dst_lmmwords=  5120 */
  /*  72 @   sdat_cpyin=16384000 sdat_lmmwords= 800 wdat_lmmwords= 4000 dst_lmmwords=102400 */
  /*  72 @   sdat_cpyin=16384000 sdat_lmmwords=3200 wdat_lmmwords=16000 dst_lmmwords= 25600 */
  /*   3 @   sdat_cpyin=40230400 sdat_lmmwords= 800 wdat_lmmwords=  800 dst_lmmwords= 50288 */
  /*   2 @   sdat_cpyin=40230400 sdat_lmmwords= 800 wdat_lmmwords= 4000 dst_lmmwords=251440 */
  /*     max sdat_cpyin=40230400 sdat_lmmwords=3200 wdat_lmmwords=16000 dst_lmmwords=251440 */
  /*                             SDATA:12,800B      WDATA:64,000B       DST:1,005,760B      */
  /* embd.size()=0 embd_inp.size()=5 params.n_predict=100 */
#endif

  /* check LMM_SIZE */
  if (NBNB00d4 > LMM_SIZE/sizeof(Uint) || NBNB00xNE11d4 > LMM_SIZE/sizeof(Uint) || NE01NE11 > LMM_SIZE/sizeof(Uint)) {
    check_lmm_ovf++;
    for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
      const uint8_t * restrict sd = (const uint8_t *)((char *)src0->data + (ir*nb01));              /* nb01:     3200B/ir¡ú, 12800B/ir¡ü */
      float         *     dst_col = (float *)        ((char *) dst->data + (ir*nb0));               /* nb0:      4B                      */
      for (int ic = 0; ic < ne11; ic++) { /* 1,5,8,9 */
	const uint8_t * restrict wd = (const uint8_t *)((char *)params->wdata + ic*nb*nb00); /* nb*nb00:3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
	float sumf = 0.0;
	/* src0->data   sd[float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b] */
	/* param->wdata wd[float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b][float,4b,4b,...4b] */
	for (int i = 0; i < nb; i++) { /* 160, 640 */
	  const float         * sdf32 = (const float *) (sd + i*bs);                 /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
	  const float         * wdf32 = (const float *) (wd + i*bs);                 /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
	  const uint8_t * restrict s0 = (const uint8_t*)(sd + i*bs + sizeof(float)); /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
	  const uint8_t * restrict w0 = (const uint8_t*)(wd + i*bs + sizeof(float)); /* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
	  /*          0  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12 12 13 13 14 15 15 15 16 16 17 17 18 18 19 19 */
	  /* sd:  float,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b */
	  /* sdata  ^sdf32 ^s0                                                                                             */
	  /*            -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */
	  /*            lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi */
	  /* wd:  float,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b,4b */
	  /* wdata  ^wdf32 ^w0                                                                                             */
	  /*            -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */
	  /*            lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi lo hi */
	  /*             *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * */
	  /*      sumf ¦² +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +   */
	  for (int j = 0; j < QK/2; j++) { /* 16 */
	    const float slo = *sdf32 * ((int8_t)(s0[j] & 0xf) - 8); const float shi = *sdf32 * ((int8_t)(s0[j] >> 4)  - 8);
	    const float wlo = *wdf32 * ((int8_t)(w0[j] & 0xf) - 8); const float whi = *wdf32 * ((int8_t)(w0[j] >> 4)  - 8);
	    sumf += slo*wlo + shi*whi;
	  }
	}
	dst_col[ic*ne0] = sumf; /* icËè¤Ë, ne0:5120W, 20480W, 50288WÈô¤Ó¢£ ºÇ³°ir¤Ç4BËè¤Ë¥¹¥È¥¢ */
      }
    }
  }
  else { /* IMAX */
#undef  NCHIP
#define NCHIP 1
    check_lmm_fit++;
    int tmp;
    monitor_time_start(THREAD, IMAX_CPYIN);
    xmax_cpyin(3, i_m0A[LANE], &tmp, src0->data,    1, 1, 1, NRNB01d4,     1);
    xmax_cpyin(3, i_m0B[LANE], &tmp, params->wdata, 1, 1, 1, NBNB00xNE11d4,1);
    xmax_bzero(   i_m0C[LANE], NE01NE11);
    monitor_time_end(THREAD, IMAX_CPYIN);

    /* sdata n_elements(ne01:50288|20480|5120) * elem_size(nb01:3200|12800)  */
    /*      5120*12800B=65,536,000B          ir_loopÆâ:sd¤Ïnb01(nb*bs)Ã±°Ì++ */
    /*                                ¢¢20B*nb(160|640) ¢¢20B*nb(160|640) .. */
    /*                       12800B         ne11(1|5|8|9)²óLOOP1Æâ:ÀèÆ¬¸ÇÄê  */
    /*                                        ¢¢¢¢¢¢¢¢¢¢20B ¢¢¢¢¢¢¢¢¢¢20B .. */
    /*                       12800B       nb(160|640)²óLOOP0Æâ:bs(20B)Ã±°Ì++ */
    /*                                        ¢¢¢¢¢¢¢¢¢¢20B ¢¢¢¢¢¢¢¢¢¢20B .. */

    /* wdata nb(160|640) * nb00(20) * ne11(1|5|8|9)                          */
    /*                       64000B ne11(1|5|8|9)²óLOOP1Æâ:nb01(nb*bs)Ã±°Ì++ */
    /*                                ¢¢20B*nb(160|640) ¢¢20B*nb(160|640) .. */
    /*                       12800B       nb(160|640)²óLOOP0Æâ:bs(20B)Ã±°Ì++ */
    /*                                        ¢¢¢¢¢¢¢¢¢¢20B ¢¢¢¢¢¢¢¢¢¢20B .. */

    /* dst   50288 * 5  = 251440                                             */
    /*                                                     ir_loopÆâ:sd¤Ï4++ */
    /*                                       ¢¢4B ---------------- ¢¢4B ..   */
    /*                                         ¢¢4B ---------------- ¢¢4B .. */
    /*             50288B ne11(1|5|8|9)²óLOOP1Æâ:ne01(50288|20480|5120B)Èô¤Ó */
    /*                                         ¢¢4B ---------------- ¢¢4B .. */
    /*                                           nb(160|640)²óLOOP0Æâ:Æ±°ÌÃÖ */
    /*                                                                  ¢¢4B */

    for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288¢£ */
      const uint8_t * restrict sd   = (const uint8_t *)((char *)i_m0A[LANE]+(ir*nb01));
      const uint8_t * restrict wd   = (const uint8_t *)((char *)i_m0B[LANE]);
      const uint8_t * restrict sdp[4];
      const uint8_t * restrict wdp[4];
      sdp[0] = sd   + sizeof(float)*1;
      wdp[0] = wd   + sizeof(float)*1;
      sdp[1] = sd   + sizeof(float)*2;
      wdp[1] = wd   + sizeof(float)*2;
      sdp[2] = sd   + sizeof(float)*3;
      wdp[2] = wd   + sizeof(float)*3;
      sdp[3] = sd   + sizeof(float)*4;
      wdp[3] = wd   + sizeof(float)*4;
      float         *       dst_col = (float *)        ((char *)i_m0C[LANE]+(ir*nb0));  /* nb0: 4B */

#define mul_mat_cores(r, c, d0, d1, d2, d3) \
	  mop(OP_LDWR,  1, &BR[r][0][1], sd,           cofs, MSK_W1,  (Ull)sd, NBNB00d4,  0, 0, (Ull)NULL, NBNB00d4);\
	  mop(OP_LDWR,  1, &BR[r][2][1], sdp[c],       cofs, MSK_W1,  (Ull)sd, NBNB00d4,  0, 0, (Ull)NULL, NBNB00d4);\
	  exe(OP_FML3,     &d0,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d1,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d2,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d3,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL)

#define mul_mat_corew(r, c, d0, d1, d2, d3, Force) \
	  mop(OP_LDWR,  1, &BR[r][0][1], wd,           iofs, MSK_W1,  (Ull)wd, NBNB00xNE11d4, 0, Force, (Ull)NULL, NBNB00xNE11d4);\
	  mop(OP_LDWR,  1, &BR[r][2][1], wdp[c],       iofs, MSK_W1,  (Ull)wd, NBNB00xNE11d4, 0, Force, (Ull)NULL, NBNB00xNE11d4);\
	  exe(OP_FML3,     &d0,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d1,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d2,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL);\
	  exe(OP_FML3,     &d3,          BR[r][0][1],  EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL)

//EMAX5A begin mul_mat_q4_0_f32 mapdist=0
 /*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
   /*2*/for (INIT1=1,LOOP1=ne11,rofs=MNBNB00_MNE0; LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
     /*1*/for (INIT0=1,LOOP0=nb,cofs=MBS; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,      &cofs,  INIT0?cofs:cofs,     EXP_H3210, BS,      EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,      &rofs,  rofs,   EXP_H3210,   INIT0?NBNB00_NE0:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,      &iofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD,      &oofs,  rofs,   EXP_H3210,   cofs,               EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

	    /* sf¢¢                                                                  */
	    /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML sd  = sf * s0[i]    */
	    /* s0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ s1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	    /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  */
	    /* wf¢¢                                                                  */
	    /*   * * * * * * * * * * * * * * * *    * * * * * * * * * * * * * * * *  *//* FML wd  = wf * w0[i]    */
	    /* w0¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ w1¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢ *//*     f32   f32  i4       */
	    /* ¦²+ + + + + + + + + + + + + + + +    + + + + + + + + + + + + + + + +  *//* FMA sum = sum + sd * wd */
	    /*float  sf = *(float*)&BR[2][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü */
	    /*Ull    s0 = BR[2][2][0];           *//* unaligned 64bit low  */
	    /*Ull    s1 = BR[2][3][0];           *//* unaligned 64bit high */
	    /*float  wf = *(float*)&BR[3][0][1]; *//* min:160*20=3200B¡ú, max:640*20=12800B¡ü *//* rofs NBNB00(nb*nb00):3200B(x9=28800B))¡ú, 12800B(x9=115200B)¡ü */
	    /*Ull    w0 = BR[3][2][0];           *//* unaligned 64bit low  */
	    /*Ull    w1 = BR[3][3][0];           *//* unaligned 64bit high */

	    /* 1/4                                                                                                               */
	    /* LDWR(sf:f32)   LDWR(s0:i4x8)                                    W W                    cofs iofs           4  #2  */
	    /* FMUL sf(f32)*s0(f32) x w8                                       ~~~ 16  17  18  19     cofs iofs           6  #3  */
	    /*                                                   16  17  18  19                                                  */
	    /* LDWR(wf:f32)   LDWR(w0:i4x8)                      |   |   |   | W W                    cofs iofs           8  #4  */
	    /* FMUL wf(f32)*s0(w32) x w8                         |   |   |   | ~~~ 20  21  22  23     cofs iofs          10  #5  */
	    /* FMUL sx(f32)*wx(w32) x w8         24  25  26  27                                       cofs iofs           6  #6  */
	    /*                                   |   |   |   |                                                                   */
	    /* 2/4                               |   |   |   |                                                                   */
	    /* LDWR(sf:f32)   LDWR(s1:i4x8)      |   |   |   |                 W W                    cofs iofs           8  #7  */
	    /* FMUL sf(f32)*s1(f32) x w8         |   |   |   |                 ~~~ 16  17  18  19     cofs iofs          10  #8  */
	    /*                                   |   |   |   |   16  17  18  19                                                  */
	    /* LDWR(wf:f32)   LDWR(w1:i4x8)      |   |   |   |   |   |   |   | W W                    cofs iofs          12  #9  */
	    /* FMUL wf(f32)*s1(w32) x w8         |   |   |   |   |   |   |   | ~~~ 20  21  22  23     cofs iofs          14  #10 */
	    /* FMUL sx(f32)*wx(w32) x w8         28  29  30  31                                       cofs iofs           6  #11 */

	    mul_mat_cores(2,  0, r16, r17, r18, r19);        /* stage #2-#3  */
	    mul_mat_corew(4,  0, r20, r21, r22, r23, Force); /* stage #4-#5 */
	    exe(OP_FML,      &r24, r16,         EXP_H3210, r20, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	    exe(OP_FML,      &r25, r17,         EXP_H3210, r21, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	    exe(OP_FML,      &r26, r18,         EXP_H3210, r22, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	    exe(OP_FML,      &r27, r19,         EXP_H3210, r23, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */

	    mul_mat_cores(7,  1, r16, r17, r18, r19);        /* stage #7-#8 */
	    mul_mat_corew(9,  1, r20, r21, r22, r23, Force); /* stage #9-#10 */
	    exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	    exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	    exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	    exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */

	    mul_mat_cores(12, 2, r16, r17, r18, r19);        /* stage #12-#13 */
	    mul_mat_corew(14, 2, r20, r21, r22, r23, Force); /* stage #14-#15 */
	    exe(OP_FMA,      &r24, r28,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	    exe(OP_FMA,      &r25, r29,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	    exe(OP_FMA,      &r26, r30,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	    exe(OP_FMA,      &r27, r31,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */

	    mul_mat_cores(17, 3, r16, r17, r18, r19);        /* stage #17-#18 */
	    mul_mat_corew(19, 3, r20, r21, r22, r23, Force); /* stage #19-#20 */
	    exe(OP_FMA,      &r28, r24,         EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	    exe(OP_FMA,      &r29, r25,         EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	    exe(OP_FMA,      &r30, r26,         EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	    exe(OP_FMA,      &r31, r27,         EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */

	    /* FAD tree */
	    exe(OP_FAD,      &r3,  r28,         EXP_H3210, r29, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	    exe(OP_FAD,      &r4,  r30,         EXP_H3210, r31, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */

	    exe(OP_FAD,      &r2,  r3,          EXP_H3210, r4,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */

	    exe(OP_FAD,      &r1,  r2,          EXP_H3232, r2,  EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */

	    exe(OP_NOP,      &AR[25][0], 0LL,   EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 (dummy to set target location) */
	    mop(OP_LDWR,  1, &r0,  dst_col,     oofs,      MSK_W0, i_m0C[LANE], NE01NE11,  0, Force, (Ull)NULL, NE01NE11); /* stage#25 */
	    exe(OP_FAD,      &r0,  INIT0?r0:r0, EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
	    mop(OP_STWR,  1, &r0,  oofs,        dst_col,   MSK_D0, i_m0C[LANE], NE01NE11,  0, Force, (Ull)NULL, NE01NE11);
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT);
    xmax_cpyout(2, dst->data, 1, 1, i_m0C[LANE], NE01NE11, 1, 1);
    monitor_time_end(THREAD, IMAX_CPYOUT);
  }
#endif
}
