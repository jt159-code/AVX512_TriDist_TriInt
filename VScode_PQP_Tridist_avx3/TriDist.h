/*************************************************************************\
  Copyright 1999 The University of North Carolina at Chapel Hill.
  All Rights Reserved.
\**************************************************************************/

#ifndef PQP_TRIDIST_H
#define PQP_TRIDIST_H

#include "PQP_Compile.h"

// 单个三角形对的距离计算 (保持原接口兼容)
PQP_REAL
TriDist(PQP_REAL p[3], PQP_REAL q[3],
        const PQP_REAL s[3][3], const PQP_REAL t[3][3]);

// ========== 批量处理接口：同时计算16对三角形 ==========
// 输入: s_batch[16][3][3], t_batch[16][3][3] - 16对三角形
// 输出: p_batch[16][3], q_batch[16][3] - 16对最近点
// 返回: dist[16] - 16个距离值
void TriDistBatch16(
    PQP_REAL p_batch[16][3],
    PQP_REAL q_batch[16][3],
    PQP_REAL dist[16],
    const PQP_REAL s_batch[16][3][3],
    const PQP_REAL t_batch[16][3][3]);

// 为了方便处理非16倍数的数据，提供8对版本
void TriDistBatch8(
    PQP_REAL p_batch[8][3],
    PQP_REAL q_batch[8][3],
    PQP_REAL dist[8],
    const PQP_REAL s_batch[8][3][3],
    const PQP_REAL t_batch[8][3][3]);

#endif