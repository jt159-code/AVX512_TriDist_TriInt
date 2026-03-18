//--------------------------------------------------------------------------
// 文件: TriDist.cpp (纯C++优化版 - 批处理 n=16 + SoA)
// 描述: 同时处理16对三角形的距离计算，使用Structure of Arrays内存布局
//        完全使用标准C++，不依赖任何SIMD指令集
//        16对大小正好填满AVX512寄存器，为未来向量化做准备
//--------------------------------------------------------------------------

#include "MatVec.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>

// ========== SoA数据结构 (Structure of Arrays) - 16对版本 ==========
// 用于存储16对三角形的所有数据，内存连续便于缓存优化
// 16个float正好填满一个AVX512寄存器（512位）
struct TriangleBatch16
{
    // 所有x坐标连续存储 (16个一组)
    float S0_x[16], S0_y[16], S0_z[16]; // 三角形S的顶点0
    float S1_x[16], S1_y[16], S1_z[16]; // 三角形S的顶点1
    float S2_x[16], S2_y[16], S2_z[16]; // 三角形S的顶点2

    float T0_x[16], T0_y[16], T0_z[16]; // 三角形T的顶点0
    float T1_x[16], T1_y[16], T1_z[16]; // 三角形T的顶点1
    float T2_x[16], T2_y[16], T2_z[16]; // 三角形T的顶点2

    // 预计算的边向量
    float Sv0_x[16], Sv0_y[16], Sv0_z[16]; // S[1]-S[0]
    float Sv1_x[16], Sv1_y[16], Sv1_z[16]; // S[2]-S[1]
    float Sv2_x[16], Sv2_y[16], Sv2_z[16]; // S[0]-S[2]

    float Tv0_x[16], Tv0_y[16], Tv0_z[16]; // T[1]-T[0]
    float Tv1_x[16], Tv1_y[16], Tv1_z[16]; // T[2]-T[1]
    float Tv2_x[16], Tv2_y[16], Tv2_z[16]; // T[0]-T[2]

    // 法向量
    float Sn_x[16], Sn_y[16], Sn_z[16]; // S的法向量
    float Tn_x[16], Tn_y[16], Tn_z[16]; // T的法向量

    // 法向量长度的平方
    float Snl2[16];
    float Tnl2[16];

    // 输出结果
    float P_x[16], P_y[16], P_z[16]; // 最近点P
    float Q_x[16], Q_y[16], Q_z[16]; // 最近点Q
    float dist[16];                  // 距离
};

// ========== 边对计算结果存储 - 16对版本 ==========
struct EdgePairResults16
{
    // 9对边的计算结果 (每对边16个结果)
    float P_x[9][16], P_y[9][16], P_z[9][16];
    float Q_x[9][16], Q_y[9][16], Q_z[9][16];
    float dist2[9][16];                             // 距离平方
    float vec_x[9][16], vec_y[9][16], vec_z[9][16]; // 从P指向Q的向量

    // 几何测试结果 (用于证明分离)
    float a[9][16], b[9][16], p[9][16];
    int shown_disjoint[16]; // 是否已证明分离

    // 构造函数
    EdgePairResults16()
    {
        memset(this, 0, sizeof(*this));
        for (int i = 0; i < 16; i++)
        {
            shown_disjoint[i] = 0;
        }
    }
};

// ========== 辅助函数：批量点积 (16个并行) ==========
static inline void dotprod_batch16(
    float *C, // 输出: 16个点积结果
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    for (int i = 0; i < 16; i++)
    {
        C[i] = A_x[i] * B_x[i] + A_y[i] * B_y[i] + A_z[i] * B_z[i];
    }
}

// ========== 辅助函数：批量向量减法 (16个并行) ==========
static inline void vec_sub_batch16(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    for (int i = 0; i < 16; i++)
    {
        V_x[i] = A_x[i] - B_x[i];
        V_y[i] = A_y[i] - B_y[i];
        V_z[i] = A_z[i] - B_z[i];
    }
}

// ========== 辅助函数：批量向量加法 (16个并行) ==========
static inline void vec_add_batch16(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    for (int i = 0; i < 16; i++)
    {
        V_x[i] = A_x[i] + B_x[i];
        V_y[i] = A_y[i] + B_y[i];
        V_z[i] = A_z[i] + B_z[i];
    }
}

// ========== 辅助函数：批量数乘 (16个并行) ==========
static inline void vec_scale_batch16(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    float s)
{
    for (int i = 0; i < 16; i++)
    {
        V_x[i] = A_x[i] * s;
        V_y[i] = A_y[i] * s;
        V_z[i] = A_z[i] * s;
    }
}

// ========== 辅助函数：批量叉积 (16个并行) ==========
static inline void crossprod_batch16(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    for (int i = 0; i < 16; i++)
    {
        V_x[i] = A_y[i] * B_z[i] - A_z[i] * B_y[i];
        V_y[i] = A_z[i] * B_x[i] - A_x[i] * B_z[i];
        V_z[i] = A_x[i] * B_y[i] - A_y[i] * B_x[i];
    }
}

// ========== 辅助函数：批量距离平方 (16个并行) ==========
static inline void dist2_batch16(
    float *D2,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    for (int i = 0; i < 16; i++)
    {
        float dx = A_x[i] - B_x[i];
        float dy = A_y[i] - B_y[i];
        float dz = A_z[i] - B_z[i];
        D2[i] = dx * dx + dy * dy + dz * dz;
    }
}

// ========== 批量计算所有边向量 (16对) ==========
static void compute_edge_vectors_batch16(TriangleBatch16 &batch)
{
    // Sv0 = S1 - S0
    vec_sub_batch16(batch.Sv0_x, batch.Sv0_y, batch.Sv0_z,
                    batch.S1_x, batch.S1_y, batch.S1_z,
                    batch.S0_x, batch.S0_y, batch.S0_z);

    // Sv1 = S2 - S1
    vec_sub_batch16(batch.Sv1_x, batch.Sv1_y, batch.Sv1_z,
                    batch.S2_x, batch.S2_y, batch.S2_z,
                    batch.S1_x, batch.S1_y, batch.S1_z);

    // Sv2 = S0 - S2
    vec_sub_batch16(batch.Sv2_x, batch.Sv2_y, batch.Sv2_z,
                    batch.S0_x, batch.S0_y, batch.S0_z,
                    batch.S2_x, batch.S2_y, batch.S2_z);

    // Tv0 = T1 - T0
    vec_sub_batch16(batch.Tv0_x, batch.Tv0_y, batch.Tv0_z,
                    batch.T1_x, batch.T1_y, batch.T1_z,
                    batch.T0_x, batch.T0_y, batch.T0_z);

    // Tv1 = T2 - T1
    vec_sub_batch16(batch.Tv1_x, batch.Tv1_y, batch.Tv1_z,
                    batch.T2_x, batch.T2_y, batch.T2_z,
                    batch.T1_x, batch.T1_y, batch.T1_z);

    // Tv2 = T0 - T2
    vec_sub_batch16(batch.Tv2_x, batch.Tv2_y, batch.Tv2_z,
                    batch.T0_x, batch.T0_y, batch.T0_z,
                    batch.T2_x, batch.T2_y, batch.T2_z);

    // 计算法向量: Sn = Sv0 × Sv1
    crossprod_batch16(batch.Sn_x, batch.Sn_y, batch.Sn_z,
                      batch.Sv0_x, batch.Sv0_y, batch.Sv0_z,
                      batch.Sv1_x, batch.Sv1_y, batch.Sv1_z);

    // 计算法向量长度平方
    for (int i = 0; i < 16; i++)
    {
        batch.Snl2[i] = batch.Sn_x[i] * batch.Sn_x[i] +
                        batch.Sn_y[i] * batch.Sn_y[i] +
                        batch.Sn_z[i] * batch.Sn_z[i];
    }

    // 计算法向量: Tn = Tv0 × Tv1
    crossprod_batch16(batch.Tn_x, batch.Tn_y, batch.Tn_z,
                      batch.Tv0_x, batch.Tv0_y, batch.Tv0_z,
                      batch.Tv1_x, batch.Tv1_y, batch.Tv1_z);

    // 计算法向量长度平方
    for (int i = 0; i < 16; i++)
    {
        batch.Tnl2[i] = batch.Tn_x[i] * batch.Tn_x[i] +
                        batch.Tn_y[i] * batch.Tn_y[i] +
                        batch.Tn_z[i] * batch.Tn_z[i];
    }
}

// ========== SegPoints函数的批量版本 (16对) ==========
static void SegPointsBatch16(
    float *VEC_x, float *VEC_y, float *VEC_z,
    float *X_x, float *X_y, float *X_z,
    float *Y_x, float *Y_y, float *Y_z,
    const float *P_x, const float *P_y, const float *P_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *Q_x, const float *Q_y, const float *Q_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    // 计算 T = Q - P
    float T_x[16], T_y[16], T_z[16];
    vec_sub_batch16(T_x, T_y, T_z, Q_x, Q_y, Q_z, P_x, P_y, P_z);

    // 计算点积
    float A_dot_A[16], B_dot_B[16], A_dot_B[16], A_dot_T[16], B_dot_T[16];
    dotprod_batch16(A_dot_A, A_x, A_y, A_z, A_x, A_y, A_z);
    dotprod_batch16(B_dot_B, B_x, B_y, B_z, B_x, B_y, B_z);
    dotprod_batch16(A_dot_B, A_x, A_y, A_z, B_x, B_y, B_z);
    dotprod_batch16(A_dot_T, A_x, A_y, A_z, T_x, T_y, T_z);
    dotprod_batch16(B_dot_T, B_x, B_y, B_z, T_x, T_y, T_z);

    // 计算分母
    float denom[16];
    for (int i = 0; i < 16; i++)
    {
        denom[i] = A_dot_A[i] * B_dot_B[i] - A_dot_B[i] * A_dot_B[i];
    }

    // 对每个三角形对独立处理（因为分支不同）
    for (int i = 0; i < 16; i++)
    {
        float t, u;

        // 处理退化情况：线段退化或平行
        if (fabs(denom[i]) < 1e-15f)
        {
            // 线段平行或退化，需要比较端点组合
            // 找到两条线段上最近的端点对
            float best_dist2 = FLT_MAX;
            float best_t = 0, best_u = 0;

            // 检查4个端点组合: (P, Q), (P+A, Q), (P, Q+B), (P+A, Q+B)
            for (int ti = 0; ti <= 1; ti++)
            {
                for (int ui = 0; ui <= 1; ui++)
                {
                    float px = P_x[i] + A_x[i] * ti;
                    float py = P_y[i] + A_y[i] * ti;
                    float pz = P_z[i] + A_z[i] * ti;
                    float qx = Q_x[i] + B_x[i] * ui;
                    float qy = Q_y[i] + B_y[i] * ui;
                    float qz = Q_z[i] + B_z[i] * ui;
                    float d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz);
                    if (d2 < best_dist2)
                    {
                        best_dist2 = d2;
                        best_t = (float)ti;
                        best_u = (float)ui;
                    }
                }
            }
            t = best_t;
            u = best_u;
        }
        else
        {
            t = (A_dot_T[i] * B_dot_B[i] - B_dot_T[i] * A_dot_B[i]) / denom[i];
            u = (t * A_dot_B[i] - B_dot_T[i]) / B_dot_B[i];
        }

        // 约束参数到[0,1]范围内，约束后需重新计算另一个参数
        float t_clamped = t;
        float u_clamped = u;

        if (t_clamped < 0)
        {
            t_clamped = 0;
            // 约束t后重新计算u
            if (B_dot_B[i] > 1e-15f)
                u_clamped = -B_dot_T[i] / B_dot_B[i];
            else
                u_clamped = 0;
        }
        else if (t_clamped > 1)
        {
            t_clamped = 1;
            // 约束t后重新计算u
            if (B_dot_B[i] > 1e-15f)
                u_clamped = (A_dot_B[i] - B_dot_T[i]) / B_dot_B[i];
            else
                u_clamped = 0;
        }

        // 再次约束u到[0,1]
        if (u_clamped < 0)
        {
            u_clamped = 0;
            // 约束u后重新计算t
            if (A_dot_A[i] > 1e-15f)
                t_clamped = A_dot_T[i] / A_dot_A[i];
            else
                t_clamped = 0;
            // 再次约束t
            if (t_clamped < 0)
                t_clamped = 0;
            if (t_clamped > 1)
                t_clamped = 1;
        }
        else if (u_clamped > 1)
        {
            u_clamped = 1;
            // 约束u后重新计算t
            if (A_dot_A[i] > 1e-15f)
                t_clamped = (A_dot_T[i] + A_dot_B[i]) / A_dot_A[i];
            else
                t_clamped = 0;
            // 再次约束t
            if (t_clamped < 0)
                t_clamped = 0;
            if (t_clamped > 1)
                t_clamped = 1;
        }

        // 计算最近点
        X_x[i] = P_x[i] + A_x[i] * t_clamped;
        X_y[i] = P_y[i] + A_y[i] * t_clamped;
        X_z[i] = P_z[i] + A_z[i] * t_clamped;

        Y_x[i] = Q_x[i] + B_x[i] * u_clamped;
        Y_y[i] = Q_y[i] + B_y[i] * u_clamped;
        Y_z[i] = Q_z[i] + B_z[i] * u_clamped;

        // 计算向量 VEC = Y - X
        VEC_x[i] = Y_x[i] - X_x[i];
        VEC_y[i] = Y_y[i] - X_y[i];
        VEC_z[i] = Y_z[i] - X_z[i];
    }
}

// ========== 批量计算一对边的最近点 (16对) ==========
static void compute_edge_pair_batch16(
    EdgePairResults16 &results,
    int edge_idx, // 0-8 对应9对边
    const TriangleBatch16 &batch,
    int s_edge, // S三角形的边索引 0,1,2
    int t_edge) // T三角形的边索引 0,1,2
{
    // 根据边索引获取对应的原点和方向向量
    const float *P_x, *P_y, *P_z; // 边的原点
    const float *A_x, *A_y, *A_z; // 边的方向向量
    const float *Q_x, *Q_y, *Q_z;
    const float *B_x, *B_y, *B_z;

    // 选择S三角形的边
    switch (s_edge)
    {
    case 0:
        P_x = batch.S0_x;
        P_y = batch.S0_y;
        P_z = batch.S0_z;
        A_x = batch.Sv0_x;
        A_y = batch.Sv0_y;
        A_z = batch.Sv0_z;
        break;
    case 1:
        P_x = batch.S1_x;
        P_y = batch.S1_y;
        P_z = batch.S1_z;
        A_x = batch.Sv1_x;
        A_y = batch.Sv1_y;
        A_z = batch.Sv1_z;
        break;
    case 2:
        P_x = batch.S2_x;
        P_y = batch.S2_y;
        P_z = batch.S2_z;
        A_x = batch.Sv2_x;
        A_y = batch.Sv2_y;
        A_z = batch.Sv2_z;
        break;
    default:
        return;
    }

    // 选择T三角形的边
    switch (t_edge)
    {
    case 0:
        Q_x = batch.T0_x;
        Q_y = batch.T0_y;
        Q_z = batch.T0_z;
        B_x = batch.Tv0_x;
        B_y = batch.Tv0_y;
        B_z = batch.Tv0_z;
        break;
    case 1:
        Q_x = batch.T1_x;
        Q_y = batch.T1_y;
        Q_z = batch.T1_z;
        B_x = batch.Tv1_x;
        B_y = batch.Tv1_y;
        B_z = batch.Tv1_z;
        break;
    case 2:
        Q_x = batch.T2_x;
        Q_y = batch.T2_y;
        Q_z = batch.T2_z;
        B_x = batch.Tv2_x;
        B_y = batch.Tv2_y;
        B_z = batch.Tv2_z;
        break;
    default:
        return;
    }

    // 调用SegPoints批量版本
    SegPointsBatch16(
        results.vec_x[edge_idx], results.vec_y[edge_idx], results.vec_z[edge_idx],
        results.P_x[edge_idx], results.P_y[edge_idx], results.P_z[edge_idx],
        results.Q_x[edge_idx], results.Q_y[edge_idx], results.Q_z[edge_idx],
        P_x, P_y, P_z, A_x, A_y, A_z, Q_x, Q_y, Q_z, B_x, B_y, B_z);

    // 计算距离平方
    for (int i = 0; i < 16; i++)
    {
        float dx = results.vec_x[edge_idx][i];
        float dy = results.vec_y[edge_idx][i];
        float dz = results.vec_z[edge_idx][i];
        results.dist2[edge_idx][i] = dx * dx + dy * dy + dz * dz;
    }

    // 计算几何测试参数
    for (int i = 0; i < 16; i++)
    {
        // 计算第三个顶点
        int s_third = (s_edge + 2) % 3;
        int t_third = (t_edge + 2) % 3;

        float Z_S_x, Z_S_y, Z_S_z;
        float Z_T_x, Z_T_y, Z_T_z;

        switch (s_third)
        {
        case 0:
            Z_S_x = batch.S0_x[i];
            Z_S_y = batch.S0_y[i];
            Z_S_z = batch.S0_z[i];
            break;
        case 1:
            Z_S_x = batch.S1_x[i];
            Z_S_y = batch.S1_y[i];
            Z_S_z = batch.S1_z[i];
            break;
        case 2:
            Z_S_x = batch.S2_x[i];
            Z_S_y = batch.S2_y[i];
            Z_S_z = batch.S2_z[i];
            break;
        }

        switch (t_third)
        {
        case 0:
            Z_T_x = batch.T0_x[i];
            Z_T_y = batch.T0_y[i];
            Z_T_z = batch.T0_z[i];
            break;
        case 1:
            Z_T_x = batch.T1_x[i];
            Z_T_y = batch.T1_y[i];
            Z_T_z = batch.T1_z[i];
            break;
        case 2:
            Z_T_x = batch.T2_x[i];
            Z_T_y = batch.T2_y[i];
            Z_T_z = batch.T2_z[i];
            break;
        }

        // Z_S = third_vertex - P
        float ZS_x = Z_S_x - P_x[i];
        float ZS_y = Z_S_y - P_y[i];
        float ZS_z = Z_S_z - P_z[i];

        // Z_T = third_vertex - Q
        float ZT_x = Z_T_x - Q_x[i];
        float ZT_y = Z_T_y - Q_y[i];
        float ZT_z = Z_T_z - Q_z[i];

        // a = Z_S · VEC
        results.a[edge_idx][i] = ZS_x * results.vec_x[edge_idx][i] +
                                 ZS_y * results.vec_y[edge_idx][i] +
                                 ZS_z * results.vec_z[edge_idx][i];

        // b = Z_T · VEC
        results.b[edge_idx][i] = ZT_x * results.vec_x[edge_idx][i] +
                                 ZT_y * results.vec_y[edge_idx][i] +
                                 ZT_z * results.vec_z[edge_idx][i];

        // p = V · VEC (其中 V = Q - P)
        float V_x = Q_x[i] - P_x[i];
        float V_y = Q_y[i] - P_y[i];
        float V_z = Q_z[i] - P_z[i];

        results.p[edge_idx][i] = V_x * results.vec_x[edge_idx][i] +
                                 V_y * results.vec_y[edge_idx][i] +
                                 V_z * results.vec_z[edge_idx][i];

        // 检查是否可证明分离
        float a_clamped = results.a[edge_idx][i];
        float b_clamped = results.b[edge_idx][i];
        if (a_clamped < 0)
            a_clamped = 0;
        if (b_clamped > 0)
            b_clamped = 0;

        if (results.p[edge_idx][i] - a_clamped + b_clamped > 0)
        {
            results.shown_disjoint[i] = 1;
        }
    }
}

// ========== 批量选择最优最近点 (16对) ==========
static void select_best_points_batch16(
    const EdgePairResults16 &results,
    float P_x[16], float P_y[16], float P_z[16],
    float Q_x[16], float Q_y[16], float Q_z[16],
    float dist[16],
    int shown_disjoint[16])
{
    // 初始化每个三角形对的最优值
    for (int i = 0; i < 16; i++)
    {
        dist[i] = FLT_MAX;
        shown_disjoint[i] = results.shown_disjoint[i];
    }

    // 遍历9对边，选择距离最小的
    for (int edge = 0; edge < 9; edge++)
    {
        for (int i = 0; i < 16; i++)
        {
            if (results.dist2[edge][i] < dist[i])
            {
                dist[i] = results.dist2[edge][i];
                P_x[i] = results.P_x[edge][i];
                P_y[i] = results.P_y[edge][i];
                P_z[i] = results.P_z[edge][i];
                Q_x[i] = results.Q_x[edge][i];
                Q_y[i] = results.Q_y[edge][i];
                Q_z[i] = results.Q_z[edge][i];
            }
        }
    }

    // 距离平方转距离
    for (int i = 0; i < 16; i++)
    {
        dist[i] = sqrtf(dist[i]);
    }
}

// ========== 顶点投影测试 (16对) ==========
static void check_vertex_projection_batch16(
    TriangleBatch16 &batch,
    float P_x[16], float P_y[16], float P_z[16],
    float Q_x[16], float Q_y[16], float Q_z[16],
    float dist[16],
    const int shown_disjoint[16])
{
    // 检查三角形S的法向量方向 - T的顶点投影到S面
    for (int i = 0; i < 16; i++)
    {
        // 如果已证明分离，跳过顶点投影测试
        if (shown_disjoint[i])
            continue;

        if (batch.Snl2[i] <= 1e-15f)
            continue; // 退化三角形跳过

        // 计算T的三个顶点在S法向上的投影
        float Tp[3];

        // T0
        float V0_x = batch.S0_x[i] - batch.T0_x[i];
        float V0_y = batch.S0_y[i] - batch.T0_y[i];
        float V0_z = batch.S0_z[i] - batch.T0_z[i];
        Tp[0] = V0_x * batch.Sn_x[i] + V0_y * batch.Sn_y[i] + V0_z * batch.Sn_z[i];

        // T1
        float V1_x = batch.S0_x[i] - batch.T1_x[i];
        float V1_y = batch.S0_y[i] - batch.T1_y[i];
        float V1_z = batch.S0_z[i] - batch.T1_z[i];
        Tp[1] = V1_x * batch.Sn_x[i] + V1_y * batch.Sn_y[i] + V1_z * batch.Sn_z[i];

        // T2
        float V2_x = batch.S0_x[i] - batch.T2_x[i];
        float V2_y = batch.S0_y[i] - batch.T2_y[i];
        float V2_z = batch.S0_z[i] - batch.T2_z[i];
        Tp[2] = V2_x * batch.Sn_x[i] + V2_y * batch.Sn_y[i] + V2_z * batch.Sn_z[i];

        // 如果所有投影同号，找到绝对值最小的顶点
        if ((Tp[0] > 0 && Tp[1] > 0 && Tp[2] > 0) ||
            (Tp[0] < 0 && Tp[1] < 0 && Tp[2] < 0))
        {

            int point = 0;
            if (fabs(Tp[1]) < fabs(Tp[point]))
                point = 1;
            if (fabs(Tp[2]) < fabs(Tp[point]))
                point = 2;

            // 计算投影点
            float t = Tp[point] / batch.Snl2[i];

            // 获取顶点坐标
            float Vtx_x, Vtx_y, Vtx_z;
            switch (point)
            {
            case 0:
                Vtx_x = batch.T0_x[i];
                Vtx_y = batch.T0_y[i];
                Vtx_z = batch.T0_z[i];
                break;
            case 1:
                Vtx_x = batch.T1_x[i];
                Vtx_y = batch.T1_y[i];
                Vtx_z = batch.T1_z[i];
                break;
            case 2:
                Vtx_x = batch.T2_x[i];
                Vtx_y = batch.T2_y[i];
                Vtx_z = batch.T2_z[i];
                break;
            default:
                Vtx_x = batch.T0_x[i];
                Vtx_y = batch.T0_y[i];
                Vtx_z = batch.T0_z[i];
                break;
            }

            // 计算新的P和Q点
            float new_P_x = Vtx_x;
            float new_P_y = Vtx_y;
            float new_P_z = Vtx_z;

            float new_Q_x = new_P_x + batch.Sn_x[i] * t;
            float new_Q_y = new_P_y + batch.Sn_y[i] * t;
            float new_Q_z = new_P_z + batch.Sn_z[i] * t;

            float dx = new_Q_x - new_P_x;
            float dy = new_Q_y - new_P_y;
            float dz = new_Q_z - new_P_z;
            float new_dist = sqrtf(dx * dx + dy * dy + dz * dz);

            // 只在新距离更小时更新结果
            if (new_dist < dist[i])
            {
                P_x[i] = new_P_x;
                P_y[i] = new_P_y;
                P_z[i] = new_P_z;
                Q_x[i] = new_Q_x;
                Q_y[i] = new_Q_y;
                Q_z[i] = new_Q_z;
                dist[i] = new_dist;
            }
        }
    }

    // 检查三角形T的法向量方向 - S的顶点投影到T面
    for (int i = 0; i < 16; i++)
    {
        // 如果已证明分离，跳过顶点投影测试
        if (shown_disjoint[i])
            continue;

        if (batch.Tnl2[i] <= 1e-15f)
            continue;

        // 计算S的三个顶点在T法向上的投影
        float Sp[3];

        // S0
        float V0_x = batch.T0_x[i] - batch.S0_x[i];
        float V0_y = batch.T0_y[i] - batch.S0_y[i];
        float V0_z = batch.T0_z[i] - batch.S0_z[i];
        Sp[0] = V0_x * batch.Tn_x[i] + V0_y * batch.Tn_y[i] + V0_z * batch.Tn_z[i];

        // S1
        float V1_x = batch.T0_x[i] - batch.S1_x[i];
        float V1_y = batch.T0_y[i] - batch.S1_y[i];
        float V1_z = batch.T0_z[i] - batch.S1_z[i];
        Sp[1] = V1_x * batch.Tn_x[i] + V1_y * batch.Tn_y[i] + V1_z * batch.Tn_z[i];

        // S2
        float V2_x = batch.T0_x[i] - batch.S2_x[i];
        float V2_y = batch.T0_y[i] - batch.S2_y[i];
        float V2_z = batch.T0_z[i] - batch.S2_z[i];
        Sp[2] = V2_x * batch.Tn_x[i] + V2_y * batch.Tn_y[i] + V2_z * batch.Tn_z[i];

        if ((Sp[0] > 0 && Sp[1] > 0 && Sp[2] > 0) ||
            (Sp[0] < 0 && Sp[1] < 0 && Sp[2] < 0))
        {

            int point = 0;
            if (fabs(Sp[1]) < fabs(Sp[point]))
                point = 1;
            if (fabs(Sp[2]) < fabs(Sp[point]))
                point = 2;

            float t = Sp[point] / batch.Tnl2[i];

            float Vtx_x, Vtx_y, Vtx_z;
            switch (point)
            {
            case 0:
                Vtx_x = batch.S0_x[i];
                Vtx_y = batch.S0_y[i];
                Vtx_z = batch.S0_z[i];
                break;
            case 1:
                Vtx_x = batch.S1_x[i];
                Vtx_y = batch.S1_y[i];
                Vtx_z = batch.S1_z[i];
                break;
            case 2:
                Vtx_x = batch.S2_x[i];
                Vtx_y = batch.S2_y[i];
                Vtx_z = batch.S2_z[i];
                break;
            default:
                Vtx_x = batch.S0_x[i];
                Vtx_y = batch.S0_y[i];
                Vtx_z = batch.S0_z[i];
                break;
            }

            // 计算新的Q和P点 (注意顺序：Q在T面上，P是S的顶点)
            float new_Q_x = Vtx_x;
            float new_Q_y = Vtx_y;
            float new_Q_z = Vtx_z;

            float new_P_x = new_Q_x + batch.Tn_x[i] * t;
            float new_P_y = new_Q_y + batch.Tn_y[i] * t;
            float new_P_z = new_Q_z + batch.Tn_z[i] * t;

            float dx = new_Q_x - new_P_x;
            float dy = new_Q_y - new_P_y;
            float dz = new_Q_z - new_P_z;
            float new_dist = sqrtf(dx * dx + dy * dy + dz * dz);

            // 只在新距离更小时更新结果
            if (new_dist < dist[i])
            {
                P_x[i] = new_P_x;
                P_y[i] = new_P_y;
                P_z[i] = new_P_z;
                Q_x[i] = new_Q_x;
                Q_y[i] = new_Q_y;
                Q_z[i] = new_Q_z;
                dist[i] = new_dist;
            }
        }
    }
}

// ========== 批量三角形距离计算主函数 (16对) ==========
void TriDistBatch16(
    PQP_REAL p_batch[16][3],
    PQP_REAL q_batch[16][3],
    PQP_REAL dist[16],
    const PQP_REAL s_batch[16][3][3],
    const PQP_REAL t_batch[16][3][3])
{
    // 转换输入数据到SoA格式
    TriangleBatch16 batch;

    // 填充batch数据
    for (int i = 0; i < 16; i++)
    {
        // S三角形顶点
        batch.S0_x[i] = s_batch[i][0][0];
        batch.S0_y[i] = s_batch[i][0][1];
        batch.S0_z[i] = s_batch[i][0][2];

        batch.S1_x[i] = s_batch[i][1][0];
        batch.S1_y[i] = s_batch[i][1][1];
        batch.S1_z[i] = s_batch[i][1][2];

        batch.S2_x[i] = s_batch[i][2][0];
        batch.S2_y[i] = s_batch[i][2][1];
        batch.S2_z[i] = s_batch[i][2][2];

        // T三角形顶点
        batch.T0_x[i] = t_batch[i][0][0];
        batch.T0_y[i] = t_batch[i][0][1];
        batch.T0_z[i] = t_batch[i][0][2];

        batch.T1_x[i] = t_batch[i][1][0];
        batch.T1_y[i] = t_batch[i][1][1];
        batch.T1_z[i] = t_batch[i][1][2];

        batch.T2_x[i] = t_batch[i][2][0];
        batch.T2_y[i] = t_batch[i][2][1];
        batch.T2_z[i] = t_batch[i][2][2];
    }

    // 计算所有边向量和法向量
    compute_edge_vectors_batch16(batch);

    // 存储9对边的结果
    EdgePairResults16 results;

    // 计算所有9对边的最近点
    compute_edge_pair_batch16(results, 0, batch, 0, 0);
    compute_edge_pair_batch16(results, 1, batch, 0, 1);
    compute_edge_pair_batch16(results, 2, batch, 0, 2);
    compute_edge_pair_batch16(results, 3, batch, 1, 0);
    compute_edge_pair_batch16(results, 4, batch, 1, 1);
    compute_edge_pair_batch16(results, 5, batch, 1, 2);
    compute_edge_pair_batch16(results, 6, batch, 2, 0);
    compute_edge_pair_batch16(results, 7, batch, 2, 1);
    compute_edge_pair_batch16(results, 8, batch, 2, 2);

    // 选择每个三角形对的最优最近点
    float P_x[16], P_y[16], P_z[16];
    float Q_x[16], Q_y[16], Q_z[16];
    int shown_disjoint[16];
    select_best_points_batch16(results, P_x, P_y, P_z, Q_x, Q_y, Q_z, dist, shown_disjoint);

    // 检查顶点投影情况（处理顶点-面最近点）
    check_vertex_projection_batch16(batch, P_x, P_y, P_z, Q_x, Q_y, Q_z, dist, shown_disjoint);

    // 转换回AoS格式输出
    for (int i = 0; i < 16; i++)
    {
        p_batch[i][0] = P_x[i];
        p_batch[i][1] = P_y[i];
        p_batch[i][2] = P_z[i];

        q_batch[i][0] = Q_x[i];
        q_batch[i][1] = Q_y[i];
        q_batch[i][2] = Q_z[i];
    }
}

// ========== 8对版本 (用于处理非16倍数的情况) ==========
void TriDistBatch8(
    PQP_REAL p_batch[8][3],
    PQP_REAL q_batch[8][3],
    PQP_REAL dist[8],
    const PQP_REAL s_batch[8][3][3],
    const PQP_REAL t_batch[8][3][3])
{
    // 转换为16对格式，只填充前8个
    PQP_REAL s_batch16[16][3][3];
    PQP_REAL t_batch16[16][3][3];
    PQP_REAL p_batch16[16][3];
    PQP_REAL q_batch16[16][3];
    PQP_REAL dist16[16];

    // 复制前8对
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                s_batch16[i][j][k] = s_batch[i][j][k];
                t_batch16[i][j][k] = t_batch[i][j][k];
            }
        }
    }

    // 后8对填0（不会用到）
    for (int i = 8; i < 16; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                s_batch16[i][j][k] = 0;
                t_batch16[i][j][k] = 0;
            }
        }
    }

    // 调用16对版本
    TriDistBatch16(p_batch16, q_batch16, dist16, s_batch16, t_batch16);

    // 复制前8个结果
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            p_batch[i][j] = p_batch16[i][j];
            q_batch[i][j] = q_batch16[i][j];
        }
        dist[i] = dist16[i];
    }
}

// ========== 单个三角形对计算（保持原接口）==========
// 独立的标量实现，避免batch转换开销
PQP_REAL TriDist(PQP_REAL P[3], PQP_REAL Q[3],
                 const PQP_REAL S[3][3], const PQP_REAL T[3][3])
{
    // 计算边向量
    PQP_REAL Sv[3][3], Tv[3][3];
    for (int i = 0; i < 3; i++)
    {
        Sv[0][i] = S[1][i] - S[0][i]; // S[1]-S[0]
        Sv[1][i] = S[2][i] - S[1][i]; // S[2]-S[1]
        Sv[2][i] = S[0][i] - S[2][i]; // S[0]-S[2]
        Tv[0][i] = T[1][i] - T[0][i]; // T[1]-T[0]
        Tv[1][i] = T[2][i] - T[1][i]; // T[2]-T[1]
        Tv[2][i] = T[0][i] - T[2][i]; // T[0]-T[2]
    }

    // 计算法向量
    PQP_REAL Sn[3], Tn[3];
    Sn[0] = Sv[0][1] * Sv[1][2] - Sv[0][2] * Sv[1][1];
    Sn[1] = Sv[0][2] * Sv[1][0] - Sv[0][0] * Sv[1][2];
    Sn[2] = Sv[0][0] * Sv[1][1] - Sv[0][1] * Sv[1][0];

    Tn[0] = Tv[0][1] * Tv[1][2] - Tv[0][2] * Tv[1][1];
    Tn[1] = Tv[0][2] * Tv[1][0] - Tv[0][0] * Tv[1][2];
    Tn[2] = Tv[0][0] * Tv[1][1] - Tv[0][1] * Tv[1][0];

    // 法向量长度平方
    PQP_REAL Snl2 = Sn[0] * Sn[0] + Sn[1] * Sn[1] + Sn[2] * Sn[2];
    PQP_REAL Tnl2 = Tn[0] * Tn[0] + Tn[1] * Tn[1] + Tn[2] * Tn[2];

    // 初始化结果
    PQP_REAL min_dist = FLT_MAX;
    int shown_disjoint = 0;

    // ========== 内部函数：计算两条线段的最近点 ==========
    auto SegPoints = [&](const PQP_REAL *SegPt, const PQP_REAL *SegVec,
                         const PQP_REAL *TriPt, const PQP_REAL *TriVec,
                         PQP_REAL *X, PQP_REAL *Y) -> PQP_REAL
    {
        PQP_REAL T[3], V[3];
        for (int i = 0; i < 3; i++)
        {
            T[i] = TriPt[i] - SegPt[i];
            V[i] = Y[i] - X[i];
        }

        PQP_REAL A_dot_A = SegVec[0] * SegVec[0] + SegVec[1] * SegVec[1] + SegVec[2] * SegVec[2];
        PQP_REAL B_dot_B = TriVec[0] * TriVec[0] + TriVec[1] * TriVec[1] + TriVec[2] * TriVec[2];
        PQP_REAL A_dot_B = SegVec[0] * TriVec[0] + SegVec[1] * TriVec[1] + SegVec[2] * TriVec[2];
        PQP_REAL A_dot_T = SegVec[0] * T[0] + SegVec[1] * T[1] + SegVec[2] * T[2];
        PQP_REAL B_dot_T = TriVec[0] * T[0] + TriVec[1] * T[1] + TriVec[2] * T[2];

        PQP_REAL denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;
        PQP_REAL t, u;

        if (fabs(denom) < 1e-15f)
        {
            // 线段平行或退化，比较端点组合
            PQP_REAL best_dist2 = FLT_MAX;
            PQP_REAL best_t = 0, best_u = 0;

            for (int ti = 0; ti <= 1; ti++)
            {
                for (int ui = 0; ui <= 1; ui++)
                {
                    PQP_REAL px = SegPt[0] + SegVec[0] * ti;
                    PQP_REAL py = SegPt[1] + SegVec[1] * ti;
                    PQP_REAL pz = SegPt[2] + SegVec[2] * ti;
                    PQP_REAL qx = TriPt[0] + TriVec[0] * ui;
                    PQP_REAL qy = TriPt[1] + TriVec[1] * ui;
                    PQP_REAL qz = TriPt[2] + TriVec[2] * ui;
                    PQP_REAL d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz);
                    if (d2 < best_dist2)
                    {
                        best_dist2 = d2;
                        best_t = (PQP_REAL)ti;
                        best_u = (PQP_REAL)ui;
                    }
                }
            }
            t = best_t;
            u = best_u;
        }
        else
        {
            t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;
            u = (t * A_dot_B - B_dot_T) / B_dot_B;
        }

        // 约束参数
        if (t < 0)
        {
            t = 0;
            if (B_dot_B > 1e-15f)
                u = -B_dot_T / B_dot_B;
            else
                u = 0;
        }
        else if (t > 1)
        {
            t = 1;
            if (B_dot_B > 1e-15f)
                u = (A_dot_B - B_dot_T) / B_dot_B;
            else
                u = 0;
        }

        if (u < 0)
        {
            u = 0;
            if (A_dot_A > 1e-15f)
                t = A_dot_T / A_dot_A;
            else
                t = 0;
            if (t < 0)
                t = 0;
            if (t > 1)
                t = 1;
        }
        else if (u > 1)
        {
            u = 1;
            if (A_dot_A > 1e-15f)
                t = (A_dot_T + A_dot_B) / A_dot_A;
            else
                t = 0;
            if (t < 0)
                t = 0;
            if (t > 1)
                t = 1;
        }

        // 计算最近点
        for (int i = 0; i < 3; i++)
        {
            X[i] = SegPt[i] + SegVec[i] * t;
            Y[i] = TriPt[i] + TriVec[i] * u;
        }

        PQP_REAL dx = Y[0] - X[0];
        PQP_REAL dy = Y[1] - X[1];
        PQP_REAL dz = Y[2] - X[2];
        return dx * dx + dy * dy + dz * dz;
    };

    // ========== 计算9对边的最近点 ==========
    for (int s_edge = 0; s_edge < 3; s_edge++)
    {
        for (int t_edge = 0; t_edge < 3; t_edge++)
        {
            PQP_REAL X[3], Y[3];
            PQP_REAL dist2 = SegPoints(S[s_edge], Sv[s_edge], T[t_edge], Tv[t_edge], X, Y);

            if (dist2 < min_dist * min_dist)
            {
                min_dist = sqrtf(dist2);
                for (int i = 0; i < 3; i++)
                {
                    P[i] = X[i];
                    Q[i] = Y[i];
                }
            }

            // 检查分离条件
            if (!shown_disjoint)
            {
                // 计算第三个顶点
                int s_third = (s_edge + 2) % 3;
                int t_third = (t_edge + 2) % 3;

                PQP_REAL ZS[3], ZT[3], VEC[3];
                for (int i = 0; i < 3; i++)
                {
                    ZS[i] = S[s_third][i] - S[s_edge][i];
                    ZT[i] = T[t_third][i] - T[t_edge][i];
                    VEC[i] = Y[i] - X[i];
                }

                PQP_REAL a = ZS[0] * VEC[0] + ZS[1] * VEC[1] + ZS[2] * VEC[2];
                PQP_REAL b = ZT[0] * VEC[0] + ZT[1] * VEC[1] + ZT[2] * VEC[2];
                PQP_REAL V_dot_VEC = (T[t_edge][0] - S[s_edge][0]) * VEC[0] +
                                     (T[t_edge][1] - S[s_edge][1]) * VEC[1] +
                                     (T[t_edge][2] - S[s_edge][2]) * VEC[2];

                PQP_REAL a_clamped = a < 0 ? 0 : a;
                PQP_REAL b_clamped = b > 0 ? 0 : b;

                if (V_dot_VEC - a_clamped + b_clamped > 0)
                {
                    shown_disjoint = 1;
                }
            }
        }
    }

    // ========== 顶点投影测试 ==========
    // 只在未证明分离时进行
    if (!shown_disjoint)
    {
        // 检查T顶点在S法向的投影
        if (Snl2 > 1e-15f)
        {
            PQP_REAL Tp[3];
            for (int i = 0; i < 3; i++)
            {
                PQP_REAL V[3];
                for (int j = 0; j < 3; j++)
                    V[j] = S[0][j] - T[i][j];
                Tp[i] = V[0] * Sn[0] + V[1] * Sn[1] + V[2] * Sn[2];
            }

            if ((Tp[0] > 0 && Tp[1] > 0 && Tp[2] > 0) ||
                (Tp[0] < 0 && Tp[1] < 0 && Tp[2] < 0))
            {
                int point = 0;
                if (fabs(Tp[1]) < fabs(Tp[point]))
                    point = 1;
                if (fabs(Tp[2]) < fabs(Tp[point]))
                    point = 2;

                PQP_REAL t = Tp[point] / Snl2;
                PQP_REAL new_P[3], new_Q[3];

                for (int i = 0; i < 3; i++)
                {
                    new_P[i] = T[point][i];
                    new_Q[i] = new_P[i] + Sn[i] * t;
                }

                PQP_REAL dx = new_Q[0] - new_P[0];
                PQP_REAL dy = new_Q[1] - new_P[1];
                PQP_REAL dz = new_Q[2] - new_P[2];
                PQP_REAL new_dist = sqrtf(dx * dx + dy * dy + dz * dz);

                if (new_dist < min_dist)
                {
                    min_dist = new_dist;
                    for (int i = 0; i < 3; i++)
                    {
                        P[i] = new_P[i];
                        Q[i] = new_Q[i];
                    }
                }
            }
        }

        // 检查S顶点在T法向的投影
        if (Tnl2 > 1e-15f)
        {
            PQP_REAL Sp[3];
            for (int i = 0; i < 3; i++)
            {
                PQP_REAL V[3];
                for (int j = 0; j < 3; j++)
                    V[j] = T[0][j] - S[i][j];
                Sp[i] = V[0] * Tn[0] + V[1] * Tn[1] + V[2] * Tn[2];
            }

            if ((Sp[0] > 0 && Sp[1] > 0 && Sp[2] > 0) ||
                (Sp[0] < 0 && Sp[1] < 0 && Sp[2] < 0))
            {
                int point = 0;
                if (fabs(Sp[1]) < fabs(Sp[point]))
                    point = 1;
                if (fabs(Sp[2]) < fabs(Sp[point]))
                    point = 2;

                PQP_REAL t = Sp[point] / Tnl2;
                PQP_REAL new_P[3], new_Q[3];

                for (int i = 0; i < 3; i++)
                {
                    new_Q[i] = S[point][i];
                    new_P[i] = new_Q[i] + Tn[i] * t;
                }

                PQP_REAL dx = new_Q[0] - new_P[0];
                PQP_REAL dy = new_Q[1] - new_P[1];
                PQP_REAL dz = new_Q[2] - new_P[2];
                PQP_REAL new_dist = sqrtf(dx * dx + dy * dy + dz * dz);

                if (new_dist < min_dist)
                {
                    min_dist = new_dist;
                    for (int i = 0; i < 3; i++)
                    {
                        P[i] = new_P[i];
                        Q[i] = new_Q[i];
                    }
                }
            }
        }
    }

    return min_dist;
}