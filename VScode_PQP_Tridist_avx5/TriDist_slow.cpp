//--------------------------------------------------------------------------
// 文件: TriDist.cpp (AVX512优化版 - 批处理 n=16 + SoA) - 修复版本
// 描述: 同时处理16对三角形的距离计算，使用AVX512指令集加速
//--------------------------------------------------------------------------

#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <immintrin.h>

typedef float PQP_REAL;

// ========== SoA数据结构 ==========
struct TriangleBatch16
{
    float S0_x[16], S0_y[16], S0_z[16];
    float S1_x[16], S1_y[16], S1_z[16];
    float S2_x[16], S2_y[16], S2_z[16];

    float T0_x[16], T0_y[16], T0_z[16];
    float T1_x[16], T1_y[16], T1_z[16];
    float T2_x[16], T2_y[16], T2_z[16];

    float Sv0_x[16], Sv0_y[16], Sv0_z[16];
    float Sv1_x[16], Sv1_y[16], Sv1_z[16];
    float Sv2_x[16], Sv2_y[16], Sv2_z[16];

    float Tv0_x[16], Tv0_y[16], Tv0_z[16];
    float Tv1_x[16], Tv1_y[16], Tv1_z[16];
    float Tv2_x[16], Tv2_y[16], Tv2_z[16];

    float Sn_x[16], Sn_y[16], Sn_z[16];
    float Tn_x[16], Tn_y[16], Tn_z[16];

    float Snl2[16];
    float Tnl2[16];

    float P_x[16], P_y[16], P_z[16];
    float Q_x[16], Q_y[16], Q_z[16];
    float dist[16];
};

struct EdgePairResults16
{
    float P_x[9][16], P_y[9][16], P_z[9][16];
    float Q_x[9][16], Q_y[9][16], Q_z[9][16];
    float dist2[9][16];
    float vec_x[9][16], vec_y[9][16], vec_z[9][16];

    float a[9][16], b[9][16], p[9][16];
    int shown_disjoint[16];

    EdgePairResults16()
    {
        memset(this, 0, sizeof(*this));
    }
};

// ========== AVX512辅助函数 ==========
static inline __m512 load_ps(const float *ptr)
{
    return _mm512_loadu_ps(ptr);
}

static inline void store_ps(float *ptr, __m512 val)
{
    _mm512_storeu_ps(ptr, val);
}

// 批量点积
static inline void dotprod_batch16_avx512(
    float *C,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    __m512 ax = load_ps(A_x);
    __m512 ay = load_ps(A_y);
    __m512 az = load_ps(A_z);

    __m512 bx = load_ps(B_x);
    __m512 by = load_ps(B_y);
    __m512 bz = load_ps(B_z);

    __m512 dot = _mm512_fmadd_ps(ax, bx, _mm512_mul_ps(ay, by));
    dot = _mm512_fmadd_ps(az, bz, dot);

    store_ps(C, dot);
}

// 批量向量减法
static inline void vec_sub_batch16_avx512(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    store_ps(V_x, _mm512_sub_ps(load_ps(A_x), load_ps(B_x)));
    store_ps(V_y, _mm512_sub_ps(load_ps(A_y), load_ps(B_y)));
    store_ps(V_z, _mm512_sub_ps(load_ps(A_z), load_ps(B_z)));
}

// 批量向量加法
static inline void vec_add_batch16_avx512(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    store_ps(V_x, _mm512_add_ps(load_ps(A_x), load_ps(B_x)));
    store_ps(V_y, _mm512_add_ps(load_ps(A_y), load_ps(B_y)));
    store_ps(V_z, _mm512_add_ps(load_ps(A_z), load_ps(B_z)));
}

// 批量数乘
static inline void vec_scale_batch16_avx512(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    float s)
{
    __m512 scale = _mm512_set1_ps(s);
    store_ps(V_x, _mm512_mul_ps(load_ps(A_x), scale));
    store_ps(V_y, _mm512_mul_ps(load_ps(A_y), scale));
    store_ps(V_z, _mm512_mul_ps(load_ps(A_z), scale));
}

// 批量叉积
static inline void crossprod_batch16_avx512(
    float *V_x, float *V_y, float *V_z,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    __m512 ax = load_ps(A_x);
    __m512 ay = load_ps(A_y);
    __m512 az = load_ps(A_z);
    __m512 bx = load_ps(B_x);
    __m512 by = load_ps(B_y);
    __m512 bz = load_ps(B_z);

    store_ps(V_x, _mm512_fmsub_ps(ay, bz, _mm512_mul_ps(az, by)));
    store_ps(V_y, _mm512_fmsub_ps(az, bx, _mm512_mul_ps(ax, bz)));
    store_ps(V_z, _mm512_fmsub_ps(ax, by, _mm512_mul_ps(ay, bx)));
}

// 批量距离平方
static inline void dist2_batch16_avx512(
    float *D2,
    const float *A_x, const float *A_y, const float *A_z,
    const float *B_x, const float *B_y, const float *B_z)
{
    __m512 dx = _mm512_sub_ps(load_ps(A_x), load_ps(B_x));
    __m512 dy = _mm512_sub_ps(load_ps(A_y), load_ps(B_y));
    __m512 dz = _mm512_sub_ps(load_ps(A_z), load_ps(B_z));

    __m512 d2 = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dz, dz)));
    store_ps(D2, d2);
}

// 批量向量长度平方
static inline void length2_batch16_avx512(
    float *L2,
    const float *V_x, const float *V_y, const float *V_z)
{
    __m512 vx = load_ps(V_x);
    __m512 vy = load_ps(V_y);
    __m512 vz = load_ps(V_z);

    __m512 l2 = _mm512_fmadd_ps(vx, vx, _mm512_fmadd_ps(vy, vy, _mm512_mul_ps(vz, vz)));
    store_ps(L2, l2);
}

// ========== 计算边向量和法向量 ==========
static void compute_edge_vectors_batch16(TriangleBatch16 &batch)
{
    vec_sub_batch16_avx512(batch.Sv0_x, batch.Sv0_y, batch.Sv0_z,
                           batch.S1_x, batch.S1_y, batch.S1_z,
                           batch.S0_x, batch.S0_y, batch.S0_z);

    vec_sub_batch16_avx512(batch.Sv1_x, batch.Sv1_y, batch.Sv1_z,
                           batch.S2_x, batch.S2_y, batch.S2_z,
                           batch.S1_x, batch.S1_y, batch.S1_z);

    vec_sub_batch16_avx512(batch.Sv2_x, batch.Sv2_y, batch.Sv2_z,
                           batch.S0_x, batch.S0_y, batch.S0_z,
                           batch.S2_x, batch.S2_y, batch.S2_z);

    vec_sub_batch16_avx512(batch.Tv0_x, batch.Tv0_y, batch.Tv0_z,
                           batch.T1_x, batch.T1_y, batch.T1_z,
                           batch.T0_x, batch.T0_y, batch.T0_z);

    vec_sub_batch16_avx512(batch.Tv1_x, batch.Tv1_y, batch.Tv1_z,
                           batch.T2_x, batch.T2_y, batch.T2_z,
                           batch.T1_x, batch.T1_y, batch.T1_z);

    vec_sub_batch16_avx512(batch.Tv2_x, batch.Tv2_y, batch.Tv2_z,
                           batch.T0_x, batch.T0_y, batch.T0_z,
                           batch.T2_x, batch.T2_y, batch.T2_z);

    crossprod_batch16_avx512(batch.Sn_x, batch.Sn_y, batch.Sn_z,
                             batch.Sv0_x, batch.Sv0_y, batch.Sv0_z,
                             batch.Sv1_x, batch.Sv1_y, batch.Sv1_z);

    length2_batch16_avx512(batch.Snl2, batch.Sn_x, batch.Sn_y, batch.Sn_z);

    crossprod_batch16_avx512(batch.Tn_x, batch.Tn_y, batch.Tn_z,
                             batch.Tv0_x, batch.Tv0_y, batch.Tv0_z,
                             batch.Tv1_x, batch.Tv1_y, batch.Tv1_z);

    length2_batch16_avx512(batch.Tnl2, batch.Tn_x, batch.Tn_y, batch.Tn_z);
}

// ========== 计算线段最近点 ==========
static void SegPointsBatch16_avx512(
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
    vec_sub_batch16_avx512(T_x, T_y, T_z, Q_x, Q_y, Q_z, P_x, P_y, P_z);

    // 计算点积
    float A_dot_A[16], B_dot_B[16], A_dot_B[16], A_dot_T[16], B_dot_T[16];
    dotprod_batch16_avx512(A_dot_A, A_x, A_y, A_z, A_x, A_y, A_z);
    dotprod_batch16_avx512(B_dot_B, B_x, B_y, B_z, B_x, B_y, B_z);
    dotprod_batch16_avx512(A_dot_B, A_x, A_y, A_z, B_x, B_y, B_z);
    dotprod_batch16_avx512(A_dot_T, A_x, A_y, A_z, T_x, T_y, T_z);
    dotprod_batch16_avx512(B_dot_T, B_x, B_y, B_z, T_x, T_y, T_z);

    // 计算分母
    float denom[16];
    {
        __m512 a_dot_a = load_ps(A_dot_A);
        __m512 b_dot_b = load_ps(B_dot_B);
        __m512 a_dot_b = load_ps(A_dot_B);
        __m512 den = _mm512_fmsub_ps(a_dot_a, b_dot_b, _mm512_mul_ps(a_dot_b, a_dot_b));
        store_ps(denom, den);
    }

    // 加载所有数据到向量寄存器
    __m512 p_x = load_ps(P_x);
    __m512 p_y = load_ps(P_y);
    __m512 p_z = load_ps(P_z);
    __m512 a_x_v = load_ps(A_x);
    __m512 a_y_v = load_ps(A_y);
    __m512 a_z_v = load_ps(A_z);
    __m512 q_x = load_ps(Q_x);
    __m512 q_y = load_ps(Q_y);
    __m512 q_z = load_ps(Q_z);
    __m512 b_x_v = load_ps(B_x);
    __m512 b_y_v = load_ps(B_y);
    __m512 b_z_v = load_ps(B_z);

    __m512 a_dot_a = load_ps(A_dot_A);
    __m512 b_dot_b = load_ps(B_dot_B);
    __m512 a_dot_b = load_ps(A_dot_B);
    __m512 a_dot_t = load_ps(A_dot_T);
    __m512 b_dot_t = load_ps(B_dot_T);
    __m512 den = load_ps(denom);

    __m512 zero = _mm512_setzero_ps();
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 eps = _mm512_set1_ps(1e-15f);

    // 检测退化情况
    __m512 den_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(den), _mm512_set1_epi32(0x7FFFFFFF)));
    __mmask16 degen_mask = _mm512_cmp_ps_mask(den_abs, eps, _CMP_LT_OS);

    // 计算正常情况的t和u
    __m512 t_num = _mm512_fmsub_ps(a_dot_t, b_dot_b, _mm512_mul_ps(b_dot_t, a_dot_b));
    __m512 t_normal = _mm512_div_ps(t_num, den);
    __m512 u_num = _mm512_fmsub_ps(t_normal, a_dot_b, b_dot_t);
    __m512 u_normal = _mm512_div_ps(u_num, b_dot_b);

    // 初始化t和u
    __m512 t = t_normal;
    __m512 u = u_normal;

    // 处理退化情况（使用标量回退，因为退化情况很少）
    if (degen_mask != 0)
    {
        float t_vals[16], u_vals[16];
        _mm512_storeu_ps(t_vals, t_normal);
        _mm512_storeu_ps(u_vals, u_normal);

        for (int i = 0; i < 16; i++)
        {
            if ((degen_mask >> i) & 1)
            {
                // 退化情况：比较4个端点
                float best_dist2 = FLT_MAX;
                float best_t = 0, best_u = 0;

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
                t_vals[i] = best_t;
                u_vals[i] = best_u;
            }
            else
            {
                // 非退化情况：先约束 t
                float t = t_vals[i];
                float u = u_vals[i];

                // 约束 t 到 [0, 1]
                if (t < 0)
                {
                    t = 0;
                    if (fabs(B_dot_B[i]) > 1e-15f)
                        u = -B_dot_T[i] / B_dot_B[i];
                }
                else if (t > 1)
                {
                    t = 1;
                    if (fabs(B_dot_B[i]) > 1e-15f)
                        u = (A_dot_B[i] - B_dot_T[i]) / B_dot_B[i];
                }

                // 约束 u 到 [0, 1]
                if (u < 0)
                {
                    u = 0;
                    if (fabs(A_dot_A[i]) > 1e-15f)
                        t = A_dot_T[i] / A_dot_A[i];
                    if (t < 0)
                        t = 0;
                    if (t > 1)
                        t = 1;
                }
                else if (u > 1)
                {
                    u = 1;
                    if (fabs(A_dot_A[i]) > 1e-15f)
                        t = (A_dot_T[i] + A_dot_B[i]) / A_dot_A[i];
                    if (t < 0)
                        t = 0;
                    if (t > 1)
                        t = 1;
                }

                t_vals[i] = t;
                u_vals[i] = u;
            }
        }

        t = load_ps(t_vals);
        u = load_ps(u_vals);
    }
    else
    {
        // 非退化情况：批量处理边界约束
        // 约束参数（完整版本，与标量逻辑一致）
        // 首先处理 t 的边界情况
        __mmask16 t_lower = _mm512_cmp_ps_mask(t, zero, _CMP_LT_OS);
        __mmask16 t_upper = _mm512_cmp_ps_mask(t, one, _CMP_GT_OS);

        // 当 t < 0 时，t = 0，重新计算 u
        __m512 b_dot_b_safe = _mm512_mask_mov_ps(b_dot_b,
                                                 _mm512_cmp_ps_mask(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(b_dot_b), _mm512_set1_epi32(0x7FFFFFFF))), eps, _CMP_LT_OS),
                                                 _mm512_set1_ps(1.0f)); // 避免除以零
        __m512 u_new_t0 = _mm512_div_ps(_mm512_sub_ps(zero, b_dot_t), b_dot_b_safe);
        u = _mm512_mask_mov_ps(u, t_lower, u_new_t0);
        t = _mm512_mask_mov_ps(t, t_lower, zero);

        // 当 t > 1 时，t = 1，重新计算 u
        __m512 u_new_t1 = _mm512_div_ps(_mm512_sub_ps(a_dot_b, b_dot_t), b_dot_b_safe);
        u = _mm512_mask_mov_ps(u, t_upper, u_new_t1);
        t = _mm512_mask_mov_ps(t, t_upper, one);

        // 然后处理 u 的边界情况
        __mmask16 u_lower = _mm512_cmp_ps_mask(u, zero, _CMP_LT_OS);
        __mmask16 u_upper = _mm512_cmp_ps_mask(u, one, _CMP_GT_OS);

        // 当 u < 0 时，u = 0，重新计算 t
        __m512 a_dot_a_safe = _mm512_mask_mov_ps(a_dot_a,
                                                 _mm512_cmp_ps_mask(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a_dot_a), _mm512_set1_epi32(0x7FFFFFFF))), eps, _CMP_LT_OS),
                                                 _mm512_set1_ps(1.0f)); // 避免除以零
        __m512 t_new_u0 = _mm512_div_ps(a_dot_t, a_dot_a_safe);
        t_new_u0 = _mm512_min_ps(_mm512_max_ps(t_new_u0, zero), one);
        t = _mm512_mask_mov_ps(t, u_lower, t_new_u0);
        u = _mm512_mask_mov_ps(u, u_lower, zero);

        // 当 u > 1 时，u = 1，重新计算 t
        __m512 t_new_u1 = _mm512_div_ps(_mm512_add_ps(a_dot_t, a_dot_b), a_dot_a_safe);
        t_new_u1 = _mm512_min_ps(_mm512_max_ps(t_new_u1, zero), one);
        t = _mm512_mask_mov_ps(t, u_upper, t_new_u1);
        u = _mm512_mask_mov_ps(u, u_upper, one);
    }

    // 最终截断（确保在 [0,1] 范围内）
    __m512 t_clamped = _mm512_min_ps(_mm512_max_ps(t, zero), one);
    __m512 u_clamped = _mm512_min_ps(_mm512_max_ps(u, zero), one);

    // 计算最近点
    __m512 x_x = _mm512_fmadd_ps(a_x_v, t_clamped, p_x);
    __m512 x_y = _mm512_fmadd_ps(a_y_v, t_clamped, p_y);
    __m512 x_z = _mm512_fmadd_ps(a_z_v, t_clamped, p_z);

    __m512 y_x = _mm512_fmadd_ps(b_x_v, u_clamped, q_x);
    __m512 y_y = _mm512_fmadd_ps(b_y_v, u_clamped, q_y);
    __m512 y_z = _mm512_fmadd_ps(b_z_v, u_clamped, q_z);

    __m512 vec_x = _mm512_sub_ps(y_x, x_x);
    __m512 vec_y = _mm512_sub_ps(y_y, x_y);
    __m512 vec_z = _mm512_sub_ps(y_z, x_z);

    // 存储结果
    store_ps(X_x, x_x);
    store_ps(X_y, x_y);
    store_ps(X_z, x_z);
    store_ps(Y_x, y_x);
    store_ps(Y_y, y_y);
    store_ps(Y_z, y_z);
    store_ps(VEC_x, vec_x);
    store_ps(VEC_y, vec_y);
    store_ps(VEC_z, vec_z);
}

// ========== 计算一对边的最近点 ==========
static void compute_edge_pair_batch16_avx512(
    EdgePairResults16 &results,
    int edge_idx,
    const TriangleBatch16 &batch,
    int s_edge, int t_edge)
{
    const float *P_x, *P_y, *P_z;
    const float *A_x, *A_y, *A_z;
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

    SegPointsBatch16_avx512(
        results.vec_x[edge_idx], results.vec_y[edge_idx], results.vec_z[edge_idx],
        results.P_x[edge_idx], results.P_y[edge_idx], results.P_z[edge_idx],
        results.Q_x[edge_idx], results.Q_y[edge_idx], results.Q_z[edge_idx],
        P_x, P_y, P_z, A_x, A_y, A_z, Q_x, Q_y, Q_z, B_x, B_y, B_z);

    dist2_batch16_avx512(results.dist2[edge_idx],
                         results.P_x[edge_idx], results.P_y[edge_idx], results.P_z[edge_idx],
                         results.Q_x[edge_idx], results.Q_y[edge_idx], results.Q_z[edge_idx]);

    // 计算几何测试参数
    int s_third = (s_edge + 2) % 3;
    int t_third = (t_edge + 2) % 3;

    const float *ZS_x, *ZS_y, *ZS_z;
    const float *ZT_x, *ZT_y, *ZT_z;

    switch (s_third)
    {
    case 0:
        ZS_x = batch.S0_x;
        ZS_y = batch.S0_y;
        ZS_z = batch.S0_z;
        break;
    case 1:
        ZS_x = batch.S1_x;
        ZS_y = batch.S1_y;
        ZS_z = batch.S1_z;
        break;
    case 2:
        ZS_x = batch.S2_x;
        ZS_y = batch.S2_y;
        ZS_z = batch.S2_z;
        break;
    }

    switch (t_third)
    {
    case 0:
        ZT_x = batch.T0_x;
        ZT_y = batch.T0_y;
        ZT_z = batch.T0_z;
        break;
    case 1:
        ZT_x = batch.T1_x;
        ZT_y = batch.T1_y;
        ZT_z = batch.T1_z;
        break;
    case 2:
        ZT_x = batch.T2_x;
        ZT_y = batch.T2_y;
        ZT_z = batch.T2_z;
        break;
    }

    float ZS_minus_P_x[16], ZS_minus_P_y[16], ZS_minus_P_z[16];
    vec_sub_batch16_avx512(ZS_minus_P_x, ZS_minus_P_y, ZS_minus_P_z,
                           ZS_x, ZS_y, ZS_z, P_x, P_y, P_z);

    float ZT_minus_Q_x[16], ZT_minus_Q_y[16], ZT_minus_Q_z[16];
    vec_sub_batch16_avx512(ZT_minus_Q_x, ZT_minus_Q_y, ZT_minus_Q_z,
                           ZT_x, ZT_y, ZT_z, Q_x, Q_y, Q_z);

    dotprod_batch16_avx512(results.a[edge_idx],
                           ZS_minus_P_x, ZS_minus_P_y, ZS_minus_P_z,
                           results.vec_x[edge_idx], results.vec_y[edge_idx], results.vec_z[edge_idx]);

    dotprod_batch16_avx512(results.b[edge_idx],
                           ZT_minus_Q_x, ZT_minus_Q_y, ZT_minus_Q_z,
                           results.vec_x[edge_idx], results.vec_y[edge_idx], results.vec_z[edge_idx]);

    float V_x[16], V_y[16], V_z[16];
    vec_sub_batch16_avx512(V_x, V_y, V_z, Q_x, Q_y, Q_z, P_x, P_y, P_z);

    dotprod_batch16_avx512(results.p[edge_idx],
                           V_x, V_y, V_z,
                           results.vec_x[edge_idx], results.vec_y[edge_idx], results.vec_z[edge_idx]);

    // 检查是否可证明分离
    __m512 a_val = load_ps(results.a[edge_idx]);
    __m512 b_val = load_ps(results.b[edge_idx]);
    __m512 p_val = load_ps(results.p[edge_idx]);

    __m512 a_clamped = _mm512_max_ps(a_val, _mm512_setzero_ps());
    __m512 b_clamped = _mm512_min_ps(b_val, _mm512_setzero_ps());

    __m512 cond_val = _mm512_add_ps(_mm512_sub_ps(p_val, a_clamped), b_clamped);
    __mmask16 disjoint_mask = _mm512_cmp_ps_mask(cond_val, _mm512_setzero_ps(), _CMP_GT_OS);

    int mask_int = _mm512_mask2int(disjoint_mask);
    for (int i = 0; i < 16; i++)
    {
        if (mask_int & (1 << i))
            results.shown_disjoint[i] = 1;
    }
}

// ========== 选择最优最近点 ==========
static void select_best_points_batch16_avx512(
    const EdgePairResults16 &results,
    float P_x[16], float P_y[16], float P_z[16],
    float Q_x[16], float Q_y[16], float Q_z[16],
    float dist[16],
    int shown_disjoint[16])
{
    __m512 best_dist = _mm512_set1_ps(FLT_MAX);
    __m512 best_p_x = _mm512_setzero_ps();
    __m512 best_p_y = _mm512_setzero_ps();
    __m512 best_p_z = _mm512_setzero_ps();
    __m512 best_q_x = _mm512_setzero_ps();
    __m512 best_q_y = _mm512_setzero_ps();
    __m512 best_q_z = _mm512_setzero_ps();

    for (int edge = 0; edge < 9; edge++)
    {
        __m512 edge_dist = load_ps(results.dist2[edge]);
        __m512 edge_p_x = load_ps(results.P_x[edge]);
        __m512 edge_p_y = load_ps(results.P_y[edge]);
        __m512 edge_p_z = load_ps(results.P_z[edge]);
        __m512 edge_q_x = load_ps(results.Q_x[edge]);
        __m512 edge_q_y = load_ps(results.Q_y[edge]);
        __m512 edge_q_z = load_ps(results.Q_z[edge]);

        __mmask16 mask = _mm512_cmp_ps_mask(edge_dist, best_dist, _CMP_LT_OS);

        best_dist = _mm512_mask_mov_ps(best_dist, mask, edge_dist);
        best_p_x = _mm512_mask_mov_ps(best_p_x, mask, edge_p_x);
        best_p_y = _mm512_mask_mov_ps(best_p_y, mask, edge_p_y);
        best_p_z = _mm512_mask_mov_ps(best_p_z, mask, edge_p_z);
        best_q_x = _mm512_mask_mov_ps(best_q_x, mask, edge_q_x);
        best_q_y = _mm512_mask_mov_ps(best_q_y, mask, edge_q_y);
        best_q_z = _mm512_mask_mov_ps(best_q_z, mask, edge_q_z);
    }

    // 直接存储开方后的距离
    __m512 sqrt_dist = _mm512_sqrt_ps(best_dist);
    store_ps(dist, sqrt_dist);

    store_ps(P_x, best_p_x);
    store_ps(P_y, best_p_y);
    store_ps(P_z, best_p_z);
    store_ps(Q_x, best_q_x);
    store_ps(Q_y, best_q_y);
    store_ps(Q_z, best_q_z);

    memcpy(shown_disjoint, results.shown_disjoint, sizeof(int) * 16);
}

// ========== 顶点投影测试 ==========
static void check_vertex_projection_batch16_avx512(
    TriangleBatch16 &batch,
    float P_x[16], float P_y[16], float P_z[16],
    float Q_x[16], float Q_y[16], float Q_z[16],
    float dist[16],
    const int shown_disjoint[16])
{
    __m512 zero = _mm512_setzero_ps();
    __m512 eps = _mm512_set1_ps(1e-15f);

    // 加载当前结果
    __m512 cur_p_x = load_ps(P_x);
    __m512 cur_p_y = load_ps(P_y);
    __m512 cur_p_z = load_ps(P_z);
    __m512 cur_q_x = load_ps(Q_x);
    __m512 cur_q_y = load_ps(Q_y);
    __m512 cur_q_z = load_ps(Q_z);
    __m512 cur_dist = load_ps(dist);

    // 创建skip mask
    __m512i skip = _mm512_set_epi32(
        shown_disjoint[15], shown_disjoint[14], shown_disjoint[13], shown_disjoint[12],
        shown_disjoint[11], shown_disjoint[10], shown_disjoint[9], shown_disjoint[8],
        shown_disjoint[7], shown_disjoint[6], shown_disjoint[5], shown_disjoint[4],
        shown_disjoint[3], shown_disjoint[2], shown_disjoint[1], shown_disjoint[0]);
    __mmask16 skip_mask = _mm512_cmp_epi32_mask(skip, _mm512_setzero_si512(), _MM_CMPINT_EQ);

    // ========== S面投影检查 ==========
    {
        __m512 snl2 = load_ps(batch.Snl2);
        __m512 sn_x = load_ps(batch.Sn_x);
        __m512 sn_y = load_ps(batch.Sn_y);
        __m512 sn_z = load_ps(batch.Sn_z);

        __m512 s0_x = load_ps(batch.S0_x);
        __m512 s0_y = load_ps(batch.S0_y);
        __m512 s0_z = load_ps(batch.S0_z);

        __m512 t0_x = load_ps(batch.T0_x);
        __m512 t0_y = load_ps(batch.T0_y);
        __m512 t0_z = load_ps(batch.T0_z);
        __m512 t1_x = load_ps(batch.T1_x);
        __m512 t1_y = load_ps(batch.T1_y);
        __m512 t1_z = load_ps(batch.T1_z);
        __m512 t2_x = load_ps(batch.T2_x);
        __m512 t2_y = load_ps(batch.T2_y);
        __m512 t2_z = load_ps(batch.T2_z);

        // 计算投影
        __m512 v0_x = _mm512_sub_ps(s0_x, t0_x);
        __m512 v0_y = _mm512_sub_ps(s0_y, t0_y);
        __m512 v0_z = _mm512_sub_ps(s0_z, t0_z);
        __m512 tp0 = _mm512_fmadd_ps(v0_x, sn_x, _mm512_fmadd_ps(v0_y, sn_y, _mm512_mul_ps(v0_z, sn_z)));

        __m512 v1_x = _mm512_sub_ps(s0_x, t1_x);
        __m512 v1_y = _mm512_sub_ps(s0_y, t1_y);
        __m512 v1_z = _mm512_sub_ps(s0_z, t1_z);
        __m512 tp1 = _mm512_fmadd_ps(v1_x, sn_x, _mm512_fmadd_ps(v1_y, sn_y, _mm512_mul_ps(v1_z, sn_z)));

        __m512 v2_x = _mm512_sub_ps(s0_x, t2_x);
        __m512 v2_y = _mm512_sub_ps(s0_y, t2_y);
        __m512 v2_z = _mm512_sub_ps(s0_z, t2_z);
        __m512 tp2 = _mm512_fmadd_ps(v2_x, sn_x, _mm512_fmadd_ps(v2_y, sn_y, _mm512_mul_ps(v2_z, sn_z)));

        // 检查同号
        __mmask16 all_pos = _mm512_kand(
            _mm512_cmp_ps_mask(tp0, zero, _CMP_GT_OS),
            _mm512_kand(
                _mm512_cmp_ps_mask(tp1, zero, _CMP_GT_OS),
                _mm512_cmp_ps_mask(tp2, zero, _CMP_GT_OS)));

        __mmask16 all_neg = _mm512_kand(
            _mm512_cmp_ps_mask(tp0, zero, _CMP_LT_OS),
            _mm512_kand(
                _mm512_cmp_ps_mask(tp1, zero, _CMP_LT_OS),
                _mm512_cmp_ps_mask(tp2, zero, _CMP_LT_OS)));

        __mmask16 valid_mask = _mm512_kor(all_pos, all_neg);

        // 排除退化和跳过情况
        __mmask16 degen_mask = _mm512_cmp_ps_mask(snl2, eps, _CMP_LT_OS);
        __mmask16 final_mask = _mm512_kand(valid_mask,
                                           _mm512_kand(_mm512_knot(degen_mask), skip_mask));

        if (final_mask != 0)
        {
            // 找到最小绝对值
            __m512 tp0_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(tp0), _mm512_set1_epi32(0x7FFFFFFF)));
            __m512 tp1_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(tp1), _mm512_set1_epi32(0x7FFFFFFF)));
            __m512 tp2_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(tp2), _mm512_set1_epi32(0x7FFFFFFF)));

            __mmask16 mask01 = _mm512_cmp_ps_mask(tp1_abs, tp0_abs, _CMP_LT_OS);
            __m512 min_abs = _mm512_mask_mov_ps(tp0_abs, mask01, tp1_abs);
            __m512 min_val = _mm512_mask_mov_ps(tp0, mask01, tp1);

            __mmask16 mask02 = _mm512_cmp_ps_mask(tp2_abs, min_abs, _CMP_LT_OS);
            min_abs = _mm512_mask_mov_ps(min_abs, mask02, tp2_abs);
            min_val = _mm512_mask_mov_ps(min_val, mask02, tp2);

            // 选择对应顶点
            __m512 vtx_x = _mm512_mask_mov_ps(t0_x, mask02,
                                              _mm512_mask_mov_ps(t1_x, mask01, t2_x));
            __m512 vtx_y = _mm512_mask_mov_ps(t0_y, mask02,
                                              _mm512_mask_mov_ps(t1_y, mask01, t2_y));
            __m512 vtx_z = _mm512_mask_mov_ps(t0_z, mask02,
                                              _mm512_mask_mov_ps(t1_z, mask01, t2_z));

            __m512 t_val = _mm512_div_ps(min_val, snl2);

            __m512 new_p_x = vtx_x;
            __m512 new_p_y = vtx_y;
            __m512 new_p_z = vtx_z;

            __m512 new_q_x = _mm512_fmadd_ps(sn_x, t_val, new_p_x);
            __m512 new_q_y = _mm512_fmadd_ps(sn_y, t_val, new_p_y);
            __m512 new_q_z = _mm512_fmadd_ps(sn_z, t_val, new_p_z);

            __m512 dx = _mm512_sub_ps(new_q_x, new_p_x);
            __m512 dy = _mm512_sub_ps(new_q_y, new_p_y);
            __m512 dz = _mm512_sub_ps(new_q_z, new_p_z);

            __m512 new_dist = _mm512_sqrt_ps(_mm512_fmadd_ps(dx, dx,
                                                             _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dz, dz))));

            __mmask16 update_mask = _mm512_kand(final_mask,
                                                _mm512_cmp_ps_mask(new_dist, cur_dist, _CMP_LT_OS));

            if (update_mask != 0)
            {
                cur_p_x = _mm512_mask_mov_ps(cur_p_x, update_mask, new_p_x);
                cur_p_y = _mm512_mask_mov_ps(cur_p_y, update_mask, new_p_y);
                cur_p_z = _mm512_mask_mov_ps(cur_p_z, update_mask, new_p_z);
                cur_q_x = _mm512_mask_mov_ps(cur_q_x, update_mask, new_q_x);
                cur_q_y = _mm512_mask_mov_ps(cur_q_y, update_mask, new_q_y);
                cur_q_z = _mm512_mask_mov_ps(cur_q_z, update_mask, new_q_z);
                cur_dist = _mm512_mask_mov_ps(cur_dist, update_mask, new_dist);
            }
        }
    }

    // ========== T面投影检查（对称处理）=========
    {
        __m512 tnl2 = load_ps(batch.Tnl2);
        __m512 tn_x = load_ps(batch.Tn_x);
        __m512 tn_y = load_ps(batch.Tn_y);
        __m512 tn_z = load_ps(batch.Tn_z);

        __m512 t0_x = load_ps(batch.T0_x);
        __m512 t0_y = load_ps(batch.T0_y);
        __m512 t0_z = load_ps(batch.T0_z);

        __m512 s0_x = load_ps(batch.S0_x);
        __m512 s0_y = load_ps(batch.S0_y);
        __m512 s0_z = load_ps(batch.S0_z);
        __m512 s1_x = load_ps(batch.S1_x);
        __m512 s1_y = load_ps(batch.S1_y);
        __m512 s1_z = load_ps(batch.S1_z);
        __m512 s2_x = load_ps(batch.S2_x);
        __m512 s2_y = load_ps(batch.S2_y);
        __m512 s2_z = load_ps(batch.S2_z);

        // 计算S顶点在T法向上的投影
        __m512 v0_x = _mm512_sub_ps(t0_x, s0_x);
        __m512 v0_y = _mm512_sub_ps(t0_y, s0_y);
        __m512 v0_z = _mm512_sub_ps(t0_z, s0_z);
        __m512 sp0 = _mm512_fmadd_ps(v0_x, tn_x, _mm512_fmadd_ps(v0_y, tn_y, _mm512_mul_ps(v0_z, tn_z)));

        __m512 v1_x = _mm512_sub_ps(t0_x, s1_x);
        __m512 v1_y = _mm512_sub_ps(t0_y, s1_y);
        __m512 v1_z = _mm512_sub_ps(t0_z, s1_z);
        __m512 sp1 = _mm512_fmadd_ps(v1_x, tn_x, _mm512_fmadd_ps(v1_y, tn_y, _mm512_mul_ps(v1_z, tn_z)));

        __m512 v2_x = _mm512_sub_ps(t0_x, s2_x);
        __m512 v2_y = _mm512_sub_ps(t0_y, s2_y);
        __m512 v2_z = _mm512_sub_ps(t0_z, s2_z);
        __m512 sp2 = _mm512_fmadd_ps(v2_x, tn_x, _mm512_fmadd_ps(v2_y, tn_y, _mm512_mul_ps(v2_z, tn_z)));

        // 检查同号
        __mmask16 all_pos = _mm512_kand(
            _mm512_cmp_ps_mask(sp0, zero, _CMP_GT_OS),
            _mm512_kand(
                _mm512_cmp_ps_mask(sp1, zero, _CMP_GT_OS),
                _mm512_cmp_ps_mask(sp2, zero, _CMP_GT_OS)));

        __mmask16 all_neg = _mm512_kand(
            _mm512_cmp_ps_mask(sp0, zero, _CMP_LT_OS),
            _mm512_kand(
                _mm512_cmp_ps_mask(sp1, zero, _CMP_LT_OS),
                _mm512_cmp_ps_mask(sp2, zero, _CMP_LT_OS)));

        __mmask16 valid_mask = _mm512_kor(all_pos, all_neg);

        __mmask16 degen_mask = _mm512_cmp_ps_mask(tnl2, eps, _CMP_LT_OS);
        __mmask16 final_mask = _mm512_kand(valid_mask,
                                           _mm512_kand(_mm512_knot(degen_mask), skip_mask));

        if (final_mask != 0)
        {
            // 找到最小绝对值
            __m512 sp0_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(sp0), _mm512_set1_epi32(0x7FFFFFFF)));
            __m512 sp1_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(sp1), _mm512_set1_epi32(0x7FFFFFFF)));
            __m512 sp2_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(sp2), _mm512_set1_epi32(0x7FFFFFFF)));

            __mmask16 mask01 = _mm512_cmp_ps_mask(sp1_abs, sp0_abs, _CMP_LT_OS);
            __m512 min_abs = _mm512_mask_mov_ps(sp0_abs, mask01, sp1_abs);
            __m512 min_val = _mm512_mask_mov_ps(sp0, mask01, sp1);

            __mmask16 mask02 = _mm512_cmp_ps_mask(sp2_abs, min_abs, _CMP_LT_OS);
            min_abs = _mm512_mask_mov_ps(min_abs, mask02, sp2_abs);
            min_val = _mm512_mask_mov_ps(min_val, mask02, sp2);

            // 选择对应顶点
            __m512 vtx_x = _mm512_mask_mov_ps(s0_x, mask02,
                                              _mm512_mask_mov_ps(s1_x, mask01, s2_x));
            __m512 vtx_y = _mm512_mask_mov_ps(s0_y, mask02,
                                              _mm512_mask_mov_ps(s1_y, mask01, s2_y));
            __m512 vtx_z = _mm512_mask_mov_ps(s0_z, mask02,
                                              _mm512_mask_mov_ps(s1_z, mask01, s2_z));

            __m512 t_val = _mm512_div_ps(min_val, tnl2);

            __m512 new_q_x = vtx_x;
            __m512 new_q_y = vtx_y;
            __m512 new_q_z = vtx_z;

            __m512 new_p_x = _mm512_fmadd_ps(tn_x, t_val, new_q_x);
            __m512 new_p_y = _mm512_fmadd_ps(tn_y, t_val, new_q_y);
            __m512 new_p_z = _mm512_fmadd_ps(tn_z, t_val, new_q_z);

            __m512 dx = _mm512_sub_ps(new_q_x, new_p_x);
            __m512 dy = _mm512_sub_ps(new_q_y, new_p_y);
            __m512 dz = _mm512_sub_ps(new_q_z, new_p_z);

            __m512 new_dist = _mm512_sqrt_ps(_mm512_fmadd_ps(dx, dx,
                                                             _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dz, dz))));

            __mmask16 update_mask = _mm512_kand(final_mask,
                                                _mm512_cmp_ps_mask(new_dist, cur_dist, _CMP_LT_OS));

            if (update_mask != 0)
            {
                cur_p_x = _mm512_mask_mov_ps(cur_p_x, update_mask, new_p_x);
                cur_p_y = _mm512_mask_mov_ps(cur_p_y, update_mask, new_p_y);
                cur_p_z = _mm512_mask_mov_ps(cur_p_z, update_mask, new_p_z);
                cur_q_x = _mm512_mask_mov_ps(cur_q_x, update_mask, new_q_x);
                cur_q_y = _mm512_mask_mov_ps(cur_q_y, update_mask, new_q_y);
                cur_q_z = _mm512_mask_mov_ps(cur_q_z, update_mask, new_q_z);
                cur_dist = _mm512_mask_mov_ps(cur_dist, update_mask, new_dist);
            }
        }
    }

    // 存储最终结果
    store_ps(P_x, cur_p_x);
    store_ps(P_y, cur_p_y);
    store_ps(P_z, cur_p_z);
    store_ps(Q_x, cur_q_x);
    store_ps(Q_y, cur_q_y);
    store_ps(Q_z, cur_q_z);
    store_ps(dist, cur_dist);
}

// ========== 主函数 ==========
void TriDistBatch16(
    PQP_REAL p_batch[16][3],
    PQP_REAL q_batch[16][3],
    PQP_REAL dist[16],
    const PQP_REAL s_batch[16][3][3],
    const PQP_REAL t_batch[16][3][3])
{
    TriangleBatch16 batch;

    for (int i = 0; i < 16; i++)
    {
        batch.S0_x[i] = s_batch[i][0][0];
        batch.S0_y[i] = s_batch[i][0][1];
        batch.S0_z[i] = s_batch[i][0][2];

        batch.S1_x[i] = s_batch[i][1][0];
        batch.S1_y[i] = s_batch[i][1][1];
        batch.S1_z[i] = s_batch[i][1][2];

        batch.S2_x[i] = s_batch[i][2][0];
        batch.S2_y[i] = s_batch[i][2][1];
        batch.S2_z[i] = s_batch[i][2][2];

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

    compute_edge_vectors_batch16(batch);

    EdgePairResults16 results;

    compute_edge_pair_batch16_avx512(results, 0, batch, 0, 0);
    compute_edge_pair_batch16_avx512(results, 1, batch, 0, 1);
    compute_edge_pair_batch16_avx512(results, 2, batch, 0, 2);
    compute_edge_pair_batch16_avx512(results, 3, batch, 1, 0);
    compute_edge_pair_batch16_avx512(results, 4, batch, 1, 1);
    compute_edge_pair_batch16_avx512(results, 5, batch, 1, 2);
    compute_edge_pair_batch16_avx512(results, 6, batch, 2, 0);
    compute_edge_pair_batch16_avx512(results, 7, batch, 2, 1);
    compute_edge_pair_batch16_avx512(results, 8, batch, 2, 2);

    float P_x[16], P_y[16], P_z[16];
    float Q_x[16], Q_y[16], Q_z[16];
    int shown_disjoint[16];
    select_best_points_batch16_avx512(results, P_x, P_y, P_z, Q_x, Q_y, Q_z, dist, shown_disjoint);

    check_vertex_projection_batch16_avx512(batch, P_x, P_y, P_z, Q_x, Q_y, Q_z, dist, shown_disjoint);

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

// ========== 8对版本 ==========
void TriDistBatch8(
    PQP_REAL p_batch[8][3],
    PQP_REAL q_batch[8][3],
    PQP_REAL dist[8],
    const PQP_REAL s_batch[8][3][3],
    const PQP_REAL t_batch[8][3][3])
{
    PQP_REAL s_batch16[16][3][3] = {0};
    PQP_REAL t_batch16[16][3][3] = {0};
    PQP_REAL p_batch16[16][3];
    PQP_REAL q_batch16[16][3];
    PQP_REAL dist16[16];

    for (int i = 0; i < 8; i++)
    {
        memcpy(s_batch16[i], s_batch[i], sizeof(PQP_REAL) * 9);
        memcpy(t_batch16[i], t_batch[i], sizeof(PQP_REAL) * 9);
    }

    TriDistBatch16(p_batch16, q_batch16, dist16, s_batch16, t_batch16);

    for (int i = 0; i < 8; i++)
    {
        memcpy(p_batch[i], p_batch16[i], sizeof(PQP_REAL) * 3);
        memcpy(q_batch[i], q_batch16[i], sizeof(PQP_REAL) * 3);
        dist[i] = dist16[i];
    }
}

// ========== 标量版本（完整实现）==========
PQP_REAL TriDist(PQP_REAL P[3], PQP_REAL Q[3],
                 const PQP_REAL S[3][3], const PQP_REAL T[3][3])
{
    // 计算边向量
    PQP_REAL Sv[3][3], Tv[3][3];
    for (int i = 0; i < 3; i++)
    {
        Sv[0][i] = S[1][i] - S[0][i];
        Sv[1][i] = S[2][i] - S[1][i];
        Sv[2][i] = S[0][i] - S[2][i];
        Tv[0][i] = T[1][i] - T[0][i];
        Tv[1][i] = T[2][i] - T[1][i];
        Tv[2][i] = T[0][i] - T[2][i];
    }

    // 计算法向量
    PQP_REAL Sn[3], Tn[3];
    Sn[0] = Sv[0][1] * Sv[1][2] - Sv[0][2] * Sv[1][1];
    Sn[1] = Sv[0][2] * Sv[1][0] - Sv[0][0] * Sv[1][2];
    Sn[2] = Sv[0][0] * Sv[1][1] - Sv[0][1] * Sv[1][0];

    Tn[0] = Tv[0][1] * Tv[1][2] - Tv[0][2] * Tv[1][1];
    Tn[1] = Tv[0][2] * Tv[1][0] - Tv[0][0] * Tv[1][2];
    Tn[2] = Tv[0][0] * Tv[1][1] - Tv[0][1] * Tv[1][0];

    PQP_REAL Snl2 = Sn[0] * Sn[0] + Sn[1] * Sn[1] + Sn[2] * Sn[2];
    PQP_REAL Tnl2 = Tn[0] * Tn[0] + Tn[1] * Tn[1] + Tn[2] * Tn[2];

    PQP_REAL min_dist = FLT_MAX;
    int shown_disjoint = 0;

    // 线段最近点函数
    auto SegPoints = [&](const PQP_REAL *SegPt, const PQP_REAL *SegVec,
                         const PQP_REAL *TriPt, const PQP_REAL *TriVec,
                         PQP_REAL *X, PQP_REAL *Y) -> PQP_REAL
    {
        PQP_REAL Tvec[3];
        for (int i = 0; i < 3; i++)
            Tvec[i] = TriPt[i] - SegPt[i];

        PQP_REAL A_dot_A = SegVec[0] * SegVec[0] + SegVec[1] * SegVec[1] + SegVec[2] * SegVec[2];
        PQP_REAL B_dot_B = TriVec[0] * TriVec[0] + TriVec[1] * TriVec[1] + TriVec[2] * TriVec[2];
        PQP_REAL A_dot_B = SegVec[0] * TriVec[0] + SegVec[1] * TriVec[1] + SegVec[2] * TriVec[2];
        PQP_REAL A_dot_T = SegVec[0] * Tvec[0] + SegVec[1] * Tvec[1] + SegVec[2] * Tvec[2];
        PQP_REAL B_dot_T = TriVec[0] * Tvec[0] + TriVec[1] * Tvec[1] + TriVec[2] * Tvec[2];

        PQP_REAL denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;
        PQP_REAL t, u;

        if (fabs(denom) < 1e-15f)
        {
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

    // 检查9对边
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

            if (!shown_disjoint)
            {
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
                    shown_disjoint = 1;
            }
        }
    }

    // 顶点投影测试
    if (!shown_disjoint)
    {
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