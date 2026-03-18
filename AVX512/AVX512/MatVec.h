﻿//=============================================================================
// MatVec.h - AVX512加速版本
// 包含基本的向量操作：复制、加减、点积、叉积、数乘、距离平方等。
// 数据类型 PQP_REAL 由 PQP_Compile.h 定义（通常为 double）。
//=============================================================================

#ifndef PQP_MATVEC_H
#define PQP_MATVEC_H

#include <math.h>
#include <stdio.h>
#include "PQP_Compile.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// 简单的绝对值宏（仅用于非 GNU 平台）
#ifndef gnu
#define myfabs(x) ((x < 0) ? -x : x)
#endif

//=============================================================================
// 向量复制：Vr = V
// 使用 AVX512 掩码加载/存储，仅操作前3个元素
//=============================================================================
inline void VcV(PQP_REAL Vr[3], const PQP_REAL V[3])
{
#ifdef __AVX512F__
    __m512d v = _mm512_maskz_loadu_pd(0x07, V); // 加载3个双精度，其余置0
    _mm512_mask_storeu_pd(Vr, 0x07, v);         // 存储3个双精度
#else
    Vr[0] = V[0];
    Vr[1] = V[1];
    Vr[2] = V[2];
#endif
}

//=============================================================================
// 向量减法：Vr = V1 - V2
// 使用 AVX512 向量减法，掩码操作3个元素
//=============================================================================
inline void VmV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
#ifdef __AVX512F__
    __m512d a = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d b = _mm512_maskz_loadu_pd(0x07, V2);
    __m512d res = _mm512_sub_pd(a, b);
    _mm512_mask_storeu_pd(Vr, 0x07, res);
#else
    Vr[0] = V1[0] - V2[0];
    Vr[1] = V1[1] - V2[1];
    Vr[2] = V1[2] - V2[2];
#endif
}

//=============================================================================
// 向量加法：Vr = V1 + V2
//=============================================================================
inline void VpV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
#ifdef __AVX512F__
    __m512d a = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d b = _mm512_maskz_loadu_pd(0x07, V2);
    __m512d res = _mm512_add_pd(a, b);
    _mm512_mask_storeu_pd(Vr, 0x07, res);
#else
    Vr[0] = V1[0] + V2[0];
    Vr[1] = V1[1] + V2[1];
    Vr[2] = V1[2] + V2[2];
#endif
}

//=============================================================================
// 向量加标量乘向量：Vr = V1 + V2 * s
// 先广播标量s，再乘加
//=============================================================================
inline void VpVxS(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3], PQP_REAL s)
{
#ifdef __AVX512F__
    __m512d a = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d b = _mm512_maskz_loadu_pd(0x07, V2);
    __m512d scalar = _mm512_set1_pd(s);
    __m512d prod = _mm512_mul_pd(b, scalar);
    __m512d res = _mm512_add_pd(a, prod);
    _mm512_mask_storeu_pd(Vr, 0x07, res);
#else
    Vr[0] = V1[0] + V2[0] * s;
    Vr[1] = V1[1] + V2[1] * s;
    Vr[2] = V1[2] + V2[2] * s;
#endif
}

//=============================================================================
// 标量乘向量：Vr = V * s
//=============================================================================
inline void VxS(PQP_REAL Vr[3], const PQP_REAL V[3], PQP_REAL s)
{
#ifdef __AVX512F__
    __m512d v = _mm512_maskz_loadu_pd(0x07, V);
    __m512d scalar = _mm512_set1_pd(s);
    __m512d res = _mm512_mul_pd(v, scalar);
    _mm512_mask_storeu_pd(Vr, 0x07, res);
#else
    Vr[0] = V[0] * s;
    Vr[1] = V[1] * s;
    Vr[2] = V[2] * s;
#endif
}

//=============================================================================
// 向量点积：返回 V1 · V2
// 使用 reduce 累加，只累加低3个元素
//=============================================================================
inline PQP_REAL VdotV(const PQP_REAL V1[3], const PQP_REAL V2[3])
{
#ifdef __AVX512F__
    __m512d a = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d b = _mm512_maskz_loadu_pd(0x07, V2);
    __m512d prod = _mm512_mul_pd(a, b);
    // 水平累加低3个双精度（高5个为0）
    __m512d sum1 = _mm512_add_pd(prod, _mm512_permutex_pd(prod, _MM_PERM_AAAA)); // 与自身交换加
    // 更简单：直接使用 reduce 函数
    return _mm512_reduce_add_pd(prod); // 注意：reduce会累加所有8个元素，但高5个为0，结果正确
#else
    return V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2];
#endif
}

//=============================================================================
// 向量叉积：Vr = V1 × V2
// 叉积公式：(a1,a2,a3) × (b1,b2,b3) = (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
// 使用 AVX512 并行计算三个分量
//=============================================================================
inline void VcrossV(PQP_REAL Vr[3], const PQP_REAL V1[3], const PQP_REAL V2[3])
{
#ifdef __AVX512F__
    // 加载 V1 和 V2
    __m512d v1 = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d v2 = _mm512_maskz_loadu_pd(0x07, V2);

    // 为了计算叉积，我们需要将向量分量重新排列
    // 定义掩码：选择需要的分量
    // 使用 _mm512_permutex_pd 进行 lane 内 shuffle（每个128位通道内）
    // 但 v1 和 v2 是连续的 8 个双精度，我们只关心低3个。
    // 方法：将低3个分量复制到高5个位置，然后 shuffle
    // 更简单：手动计算三个分量，但使用向量指令并行
    __m512d a2 = _mm512_maskz_permutex_pd(0x07, v1, _MM_PERM_BBBB); // 提取 v1[1] (a2) 到所有 lane
    __m512d a3 = _mm512_maskz_permutex_pd(0x07, v1, _MM_PERM_CCCC); // v1[2] (a3)
    __m512d a1 = _mm512_maskz_permutex_pd(0x07, v1, _MM_PERM_AAAA); // v1[0] (a1)

    __m512d b2 = _mm512_maskz_permutex_pd(0x07, v2, _MM_PERM_BBBB);
    __m512d b3 = _mm512_maskz_permutex_pd(0x07, v2, _MM_PERM_CCCC);
    __m512d b1 = _mm512_maskz_permutex_pd(0x07, v2, _MM_PERM_AAAA);

    // 计算分量
    __m512d c1 = _mm512_sub_pd(_mm512_mul_pd(a2, b3), _mm512_mul_pd(a3, b2));
    __m512d c2 = _mm512_sub_pd(_mm512_mul_pd(a3, b1), _mm512_mul_pd(a1, b3));
    __m512d c3 = _mm512_sub_pd(_mm512_mul_pd(a1, b2), _mm512_mul_pd(a2, b1));

    // 合并结果：c1, c2, c3 分别存储在三个寄存器的低64位，需要压缩存储
    // 可以使用 shuffle 将三个结果合并到一个寄存器
    // 但存储时，我们只需要三个标量，可以直接提取低64位存储
    // 为了简单，我们使用标量提取：
    Vr[0] = _mm512_cvtsd_f64(c1);
    Vr[1] = _mm512_cvtsd_f64(c2);
    Vr[2] = _mm512_cvtsd_f64(c3);
#else
    Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
    Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
    Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
#endif
}

//=============================================================================
// 两点间距离的平方：|V1 - V2|²
// 利用 VdotV 实现
//=============================================================================
inline PQP_REAL VdistV2(const PQP_REAL V1[3], const PQP_REAL V2[3])
{
#ifdef __AVX512F__
    __m512d a = _mm512_maskz_loadu_pd(0x07, V1);
    __m512d b = _mm512_maskz_loadu_pd(0x07, V2);
    __m512d diff = _mm512_sub_pd(a, b);
    __m512d sq = _mm512_mul_pd(diff, diff);
    return _mm512_reduce_add_pd(sq);
#else
    PQP_REAL dx = V1[0] - V2[0];
    PQP_REAL dy = V1[1] - V2[1];
    PQP_REAL dz = V1[2] - V2[2];
    return dx * dx + dy * dy + dz * dz;
#endif
}

#endif // PQP_MATVEC_H