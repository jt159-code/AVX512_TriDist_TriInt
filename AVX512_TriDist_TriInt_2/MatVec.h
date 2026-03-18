//=============================================================================
// MatVec.h - 精简版本，仅保留 TriDist 所需的向量运算函数
// 包含基本的向量操作：复制、加减、点积、叉积、数乘、距离平方等。
// 数据类型 PQP_REAL 由 PQP_Compile.h 定义（通常为 double 或 float）。
//=============================================================================

#ifndef PQP_MATVEC_H
#define PQP_MATVEC_H

#include <math.h>
#include <stdio.h>
#include "PQP_Compile.h"

// 简单的绝对值宏（仅用于非 GNU 平台）
#ifndef gnu
#define myfabs(x) ((x < 0) ? -x : x)
#endif

// 向量复制：Vr = V
inline void VcV(PQP_REAL Vr[3], const PQP_REAL V[3])
{
    Vr[0] = V[0];
    Vr[1] = V[1];
    Vr[2] = V[2];
}

// 向量减法：Vr = V1 - V2
inline void VmV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
    Vr[0] = V1[0] - V2[0];
    Vr[1] = V1[1] - V2[1];
    Vr[2] = V1[2] - V2[2];
}

// 向量加法：Vr = V1 + V2
inline void VpV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
    Vr[0] = V1[0] + V2[0];
    Vr[1] = V1[1] + V2[1];
    Vr[2] = V1[2] + V2[2];
}

// 向量加标量乘向量：Vr = V1 + V2 * s
inline void VpVxS(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3], PQP_REAL s)
{
    Vr[0] = V1[0] + V2[0] * s;
    Vr[1] = V1[1] + V2[1] * s;
    Vr[2] = V1[2] + V2[2] * s;
}

// 标量乘向量：Vr = V * s
inline void VxS(PQP_REAL Vr[3], const PQP_REAL V[3], PQP_REAL s)
{
    Vr[0] = V[0] * s;
    Vr[1] = V[1] * s;
    Vr[2] = V[2] * s;
}

// 向量点积：返回 V1 · V2
inline PQP_REAL VdotV(const PQP_REAL V1[3], const PQP_REAL V2[3])
{
    return V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2];
}

// 向量叉积：Vr = V1 × V2
inline void VcrossV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
    Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
    Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
    Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
}

// 两点间距离的平方：|V1 - V2|²
inline PQP_REAL VdistV2(const PQP_REAL V1[3], const PQP_REAL V2[3])
{
    PQP_REAL dx = V1[0] - V2[0];
    PQP_REAL dy = V1[1] - V2[1];
    PQP_REAL dz = V1[2] - V2[2];
    return dx * dx + dy * dy + dz * dz;
}

#endif // PQP_MATVEC_H