//--------------------------------------------------------------------------
// 文件: TriDist.cpp (AVX512加速版)
// 包含 SegPoints() 和 TriDist() 函数，用于计算两个三角形之间的最近点对。
//--------------------------------------------------------------------------

#include "MatVec.h"
#ifdef _WIN32
#include <float.h>
#define isnan _isnan
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

//--------------------------------------------------------------------------
// SegPoints() 保持不变（分支密集，不适合向量化）
//--------------------------------------------------------------------------
void
SegPoints(PQP_REAL VEC[3],
    PQP_REAL X[3], PQP_REAL Y[3],
    const PQP_REAL P[3], const PQP_REAL A[3],
    const PQP_REAL Q[3], const PQP_REAL B[3])
{
    // ... 原代码不变 ...
    PQP_REAL T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
    PQP_REAL TMP[3];

    VmV(T, Q, P);                     // T = Q - P
    A_dot_A = VdotV(A, A);            // A·A
    B_dot_B = VdotV(B, B);            // B·B
    A_dot_B = VdotV(A, B);            // A·B
    A_dot_T = VdotV(A, T);            // A·T
    B_dot_T = VdotV(B, T);            // B·T

    PQP_REAL t, u;
    PQP_REAL denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;
    t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;
    if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;
    u = (t * A_dot_B - B_dot_T) / B_dot_B;
    if ((u <= 0) || isnan(u)) {
        VcV(Y, Q);
        t = A_dot_T / A_dot_A;
        if ((t <= 0) || isnan(t)) {
            VcV(X, P);
            VmV(VEC, Q, P);
        }
        else if (t >= 1) {
            VpV(X, P, A);
            VmV(VEC, Q, X);
        }
        else {
            VpVxS(X, P, A, t);
            VcrossV(TMP, T, A);
            VcrossV(VEC, A, TMP);
        }
    }
    else if (u >= 1) {
        VpV(Y, Q, B);
        t = (A_dot_B + A_dot_T) / A_dot_A;
        if ((t <= 0) || isnan(t)) {
            VcV(X, P);
            VmV(VEC, Y, P);
        }
        else if (t >= 1) {
            VpV(X, P, A);
            VmV(VEC, Y, X);
        }
        else {
            VpVxS(X, P, A, t);
            VmV(T, Y, P);
            VcrossV(TMP, T, A);
            VcrossV(VEC, A, TMP);
        }
    }
    else {
        VpVxS(Y, Q, B, u);
        if ((t <= 0) || isnan(t)) {
            VcV(X, P);
            VcrossV(TMP, T, B);
            VcrossV(VEC, B, TMP);
        }
        else if (t >= 1) {
            VpV(X, P, A);
            VmV(T, Q, X);
            VcrossV(TMP, T, B);
            VcrossV(VEC, B, TMP);
        }
        else {
            VpVxS(X, P, A, t);
            VcrossV(VEC, A, B);
            if (VdotV(VEC, T) < 0) {
                VxS(VEC, VEC, -1);
            }
        }
    }
}

//--------------------------------------------------------------------------
// TriDist() 向量化版本
//--------------------------------------------------------------------------
PQP_REAL
TriDist(PQP_REAL P[3], PQP_REAL Q[3],
    const PQP_REAL S[3][3], const PQP_REAL T[3][3])
{
    // 计算 6 条边的方向向量
    PQP_REAL Sv[3][3], Tv[3][3];
    PQP_REAL VEC[3];

#ifdef __AVX512F__
    // 使用 AVX512 同时计算三条边向量（每个边向量三个分量）
    // 将三角形 S 的顶点坐标视为三个数组（x, y, z），但输入是 AoS，需要 gather
    // 此处简化：手动加载并使用 AVX512 并行减法
    // 分别处理 x, y, z 分量
    __m512d sx = _mm512_set_pd(S[2][0], S[1][0], S[0][0], 0, 0, 0, 0, 0); // 低3个元素有效
    __m512d sy = _mm512_set_pd(S[2][1], S[1][1], S[0][1], 0, 0, 0, 0, 0);
    __m512d sz = _mm512_set_pd(S[2][2], S[1][2], S[0][2], 0, 0, 0, 0, 0);

    // 计算边向量：Sv[0] = S[1]-S[0], Sv[1] = S[2]-S[1], Sv[2] = S[0]-S[2]
    // 使用 permute 获取所需的顺序
    __m512d s0 = _mm512_set_pd(S[0][0], S[0][1], S[0][2], 0, 0, 0, 0, 0); // 实际只需要分量，上面已分开
    // 更清晰：分别处理每个分量
    // 对于 x 分量：计算三个差值
    __m512d sx0 = _mm512_set1_pd(S[0][0]); // 广播 S[0][0]
    __m512d sx1 = _mm512_set1_pd(S[1][0]);
    __m512d sx2 = _mm512_set1_pd(S[2][0]);
    __m512d svx0 = _mm512_sub_pd(sx1, sx0); // S[1][0]-S[0][0]
    __m512d svx1 = _mm512_sub_pd(sx2, sx1); // S[2][0]-S[1][0]
    __m512d svx2 = _mm512_sub_pd(sx0, sx2); // S[0][0]-S[2][0]
    // 存储结果
    _mm512_mask_storeu_pd(&Sv[0][0], 0x01, svx0); // 只存储低64位
    _mm512_mask_storeu_pd(&Sv[1][0], 0x01, svx1);
    _mm512_mask_storeu_pd(&Sv[2][0], 0x01, svx2);

    // 同样处理 y, z 分量
    __m512d sy0 = _mm512_set1_pd(S[0][1]);
    __m512d sy1 = _mm512_set1_pd(S[1][1]);
    __m512d sy2 = _mm512_set1_pd(S[2][1]);
    __m512d svy0 = _mm512_sub_pd(sy1, sy0);
    __m512d svy1 = _mm512_sub_pd(sy2, sy1);
    __m512d svy2 = _mm512_sub_pd(sy0, sy2);
    _mm512_mask_storeu_pd(&Sv[0][1], 0x01, svy0);
    _mm512_mask_storeu_pd(&Sv[1][1], 0x01, svy1);
    _mm512_mask_storeu_pd(&Sv[2][1], 0x01, svy2);

    __m512d sz0 = _mm512_set1_pd(S[0][2]);
    __m512d sz1 = _mm512_set1_pd(S[1][2]);
    __m512d sz2 = _mm512_set1_pd(S[2][2]);
    __m512d svz0 = _mm512_sub_pd(sz1, sz0);
    __m512d svz1 = _mm512_sub_pd(sz2, sz1);
    __m512d svz2 = _mm512_sub_pd(sz0, sz2);
    _mm512_mask_storeu_pd(&Sv[0][2], 0x01, svz0);
    _mm512_mask_storeu_pd(&Sv[1][2], 0x01, svz1);
    _mm512_mask_storeu_pd(&Sv[2][2], 0x01, svz2);

    // 对三角形 T 做同样操作
    __m512d tx0 = _mm512_set1_pd(T[0][0]);
    __m512d tx1 = _mm512_set1_pd(T[1][0]);
    __m512d tx2 = _mm512_set1_pd(T[2][0]);
    __m512d tvx0 = _mm512_sub_pd(tx1, tx0);
    __m512d tvx1 = _mm512_sub_pd(tx2, tx1);
    __m512d tvx2 = _mm512_sub_pd(tx0, tx2);
    _mm512_mask_storeu_pd(&Tv[0][0], 0x01, tvx0);
    _mm512_mask_storeu_pd(&Tv[1][0], 0x01, tvx1);
    _mm512_mask_storeu_pd(&Tv[2][0], 0x01, tvx2);

    __m512d ty0 = _mm512_set1_pd(T[0][1]);
    __m512d ty1 = _mm512_set1_pd(T[1][1]);
    __m512d ty2 = _mm512_set1_pd(T[2][1]);
    __m512d tvy0 = _mm512_sub_pd(ty1, ty0);
    __m512d tvy1 = _mm512_sub_pd(ty2, ty1);
    __m512d tvy2 = _mm512_sub_pd(ty0, ty2);
    _mm512_mask_storeu_pd(&Tv[0][1], 0x01, tvy0);
    _mm512_mask_storeu_pd(&Tv[1][1], 0x01, tvy1);
    _mm512_mask_storeu_pd(&Tv[2][1], 0x01, tvy2);

    __m512d tz0 = _mm512_set1_pd(T[0][2]);
    __m512d tz1 = _mm512_set1_pd(T[1][2]);
    __m512d tz2 = _mm512_set1_pd(T[2][2]);
    __m512d tvz0 = _mm512_sub_pd(tz1, tz0);
    __m512d tvz1 = _mm512_sub_pd(tz2, tz1);
    __m512d tvz2 = _mm512_sub_pd(tz0, tz2);
    _mm512_mask_storeu_pd(&Tv[0][2], 0x01, tvz0);
    _mm512_mask_storeu_pd(&Tv[1][2], 0x01, tvz1);
    _mm512_mask_storeu_pd(&Tv[2][2], 0x01, tvz2);
#else
    // 标量版本
    VmV(Sv[0], S[1], S[0]);
    VmV(Sv[1], S[2], S[1]);
    VmV(Sv[2], S[0], S[2]);

    VmV(Tv[0], T[1], T[0]);
    VmV(Tv[1], T[2], T[1]);
    VmV(Tv[2], T[0], T[2]);
#endif

    PQP_REAL V[3];
    PQP_REAL Z[3];
    PQP_REAL minP[3], minQ[3], mindd;
    int shown_disjoint = 0;

    mindd = VdistV2(S[0], T[0]) + 1;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            VmV(V, Q, P);
            PQP_REAL dd = VdotV(V, V);

            if (dd <= mindd)
            {
                VcV(minP, P);
                VcV(minQ, Q);
                mindd = dd;

                VmV(Z, S[(i + 2) % 3], P);
                PQP_REAL a = VdotV(Z, VEC);
                VmV(Z, T[(j + 2) % 3], Q);
                PQP_REAL b = VdotV(Z, VEC);

                if ((a <= 0) && (b >= 0)) return sqrt(dd);

                PQP_REAL p = VdotV(V, VEC);

                if (a < 0) a = 0;
                if (b > 0) b = 0;
                if ((p - a + b) > 0) shown_disjoint = 1;
            }
        }
    }

    // 检查情况1：一个顶点到另一个三角形平面的投影
    PQP_REAL Sn[3], Snl;
    VcrossV(Sn, Sv[0], Sv[1]);
    Snl = VdotV(Sn, Sn);

    if (Snl > 1e-15)
    {
        PQP_REAL Tp[3];

#ifdef __AVX512F__
        // 使用 AVX512 同时计算三个顶点到平面 S 的有符号距离
        __m512d sn = _mm512_set_pd(Sn[2], Sn[1], Sn[0], 0, 0, 0, 0, 0);
        __m512d s0 = _mm512_set_pd(S[0][2], S[0][1], S[0][0], 0, 0, 0, 0, 0);

        // 加载 T 的三个顶点坐标（分别处理 x,y,z 分量，但这里用点积）
        // 简化：使用 VdotV 标量，也可以向量化，但为了清晰暂时保持标量
        // 实际上可以一次性计算三个点积：将 T 的三个顶点坐标组成 SoA，然后与 Sn 点积
        // 这里由于数据量小，不一定能体现优势，故保留标量
        VmV(V, S[0], T[0]);
        Tp[0] = VdotV(V, Sn);
        VmV(V, S[0], T[1]);
        Tp[1] = VdotV(V, Sn);
        VmV(V, S[0], T[2]);
        Tp[2] = VdotV(V, Sn);
#else
        VmV(V, S[0], T[0]);
        Tp[0] = VdotV(V, Sn);
        VmV(V, S[0], T[1]);
        Tp[1] = VdotV(V, Sn);
        VmV(V, S[0], T[2]);
        Tp[2] = VdotV(V, Sn);
#endif

        int point = -1;
        if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
        {
            if (Tp[0] < Tp[1]) point = 0; else point = 1;
            if (Tp[2] < Tp[point]) point = 2;
        }
        else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
        {
            if (Tp[0] > Tp[1]) point = 0; else point = 1;
            if (Tp[2] > Tp[point]) point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;
            VmV(V, T[point], S[0]);
            VcrossV(Z, Sn, Sv[0]);
            if (VdotV(V, Z) > 0)
            {
                VmV(V, T[point], S[1]);
                VcrossV(Z, Sn, Sv[1]);
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, T[point], S[2]);
                    VcrossV(Z, Sn, Sv[2]);
                    if (VdotV(V, Z) > 0)
                    {
                        VpVxS(P, T[point], Sn, Tp[point] / Snl);
                        VcV(Q, T[point]);
                        return sqrt(VdistV2(P, Q));
                    }
                }
            }
        }
    }

    PQP_REAL Tn[3], Tnl;
    VcrossV(Tn, Tv[0], Tv[1]);
    Tnl = VdotV(Tn, Tn);

    if (Tnl > 1e-15)
    {
        PQP_REAL Sp[3];

        VmV(V, T[0], S[0]);
        Sp[0] = VdotV(V, Tn);
        VmV(V, T[0], S[1]);
        Sp[1] = VdotV(V, Tn);
        VmV(V, T[0], S[2]);
        Sp[2] = VdotV(V, Tn);

        int point = -1;
        if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
        {
            if (Sp[0] < Sp[1]) point = 0; else point = 1;
            if (Sp[2] < Sp[point]) point = 2;
        }
        else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
        {
            if (Sp[0] > Sp[1]) point = 0; else point = 1;
            if (Sp[2] > Sp[point]) point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;
            VmV(V, S[point], T[0]);
            VcrossV(Z, Tn, Tv[0]);
            if (VdotV(V, Z) > 0)
            {
                VmV(V, S[point], T[1]);
                VcrossV(Z, Tn, Tv[1]);
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, S[point], T[2]);
                    VcrossV(Z, Tn, Tv[2]);
                    if (VdotV(V, Z) > 0)
                    {
                        VcV(P, S[point]);
                        VpVxS(Q, S[point], Tn, Sp[point] / Tnl);
                        return sqrt(VdistV2(P, Q));
                    }
                }
            }
        }
    }

    if (shown_disjoint)
    {
        VcV(P, minP);
        VcV(Q, minQ);
        return sqrt(mindd);
    }
    else return 0;
}