
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
#include "ggml.h"
#include "monitor.h"
#include "./emax7.h"
#define NO_EMAX7LIB_BODY
#include "./emax7lib.c"

#if 1
int convf32tof8(Uchar*, float);
int convf8tof32(float*, Uchar);
int softf8(Uchar*, Uchar, Uchar, Uchar);
int convf32tos8(Uchar*, float);
int convs8tof32(float*, Uchar);
int convf32tos16(Ushort*, float);
int convs16tof32(float*, Ushort);
int convs16tos8(Uchar*, Ushort, int);
int softs8(Ushort*, Ushort, Uchar, Uchar);
int convf32tou7(Uchar*, float);
int convf32tou8(Uchar*, float);
int convu7tof32(float*, Uchar);
int convu8tof32(float*, Uchar);
int bitcountLL(Ull);
int softu64(int, Ull*, Ull*, Ull*, Ull, Ull, Ull, Ull);
#endif

void x11_vector_clear(), x11_vector_add(), x11_vector_update();

extern int      enable_x11;
extern int      CNN_DEPTH; /* default 1 */
extern int      FC_DEPTH;  /* default 1 */
extern int      VECWIN;

extern Uint    *i_m0A; /* for sgemm00 on ZYNQ_PL */
extern Uint    *i_m0B; /* for sgemm00 on ZYNQ_PL */
extern Uint    *i_m0C; /* for sgemm00 on ZYNQ_PL */
extern int     i_m0A_max_size;
extern int     i_m0B_max_size;
extern int     i_m0C_max_size;

/*          _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A              _A    _A                         */
/*          │  │  │  │  │  │  │  │  │  │  │  │  │  │              ││││                         */
/*          ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ  ξ スパイン     ξ    ξ                         */
/*          〓  〓  〓  〓  〓  〓  〓  〓  〓  〓  〓  〓  〓  〓 BLT-CAP      〓〓〓〓                         */
/*          └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴|＞─         ┴┴┴┴|＞─                    */

/*                      ┌───────────────────────────┐前後発火⇒結合強度↑により学習 */
/*                      └○───────○───────○───────nout─┘  A          ─oubatch         */
/*                      └○───                  ───○───────nout─┘  |          ─oubatch         */
/*                      └○───────○───────○───────nout─┘  |          ─oubatch         */
/* 100                  └○───                  ───○───────nout─┘ n[*][depth] ─oubatch  100    */
/* batch  V1─nhidden[0]─○───────○───────○───────nout─┘  |          ─oubatch  batch  */
/*        V1─nhidden[0]─○───────○───────○───────          |                            */
/*        V1─nhidden[0]─○───────○───────○───────          V                            */
/*                        <--------------------FC_DEPTH---------------------->                                   */
/*                nflat[0]-nout[0] nflat[1]-nout[1]       nflat[FC_DEPTH-1]-nout[FC_DEPTH-1]                     */
/*                      Wh2o[0]         Wh2o[1]                     Wh2o[FC_DEPTH-1]                             */
/*            12x12x19*200-200          200-200                          40-10                                   */

/* float2D {int nstrides; int stride_size; float *data;}                                                                    */
/* float2D CNNet->FC-nflat  [0][batch, isize, data] in    (spike)                                  ●同一outからコピー      */
/* float2D CNNet->FC-Wh2o   [0][isize, osize, data] weight(複数本の逓倍含む) -100:抑制 〜 100:発火 ●結合強度は確率的に変動 */
/* float2D CNNet->FC-g_Wh2o [0][isize, osize, data] 未使用                                                                  */
/* float2D CNNet->FC-nout   [0][batch, osize, data] out   (spike)                                                           */
/* float2D CNNet->FC-noutbak[0][batch, osize, data] 膜電位(時間とともに減衰,th以下で再発火)                                 */
/* float2D CNNet->FC-obias  [0][1,     osize, data] 未使用 ●前段ニューロン(in)発火+直後に後段ニューロン(out)発火⇒結合強化 */
/* float2D CNNet->FC-g_obias[0][1,     osize, data] 未使用 ●前段ニューロン(in)発火+直後に後段ニューロン(out)不発⇒結合弱体 */

//UNARY8_FC   ... Unary化
//SINT8_FC    ... Signed_Int化
//NMORPHIC_FC ... 積和のままneuro-morphic化
//DIGITAL_FC  ... loop交換・変形したコード
//ORIGINAL_FC ... 元のrsimコード

#define   UNARY8_FC
//#define SINT8_FC
//#define NMORPHIC_FC
//#define DIGITAL_FC
//#define ORIGINAL_FC
