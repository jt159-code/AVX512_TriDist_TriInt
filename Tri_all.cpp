// Tri_avx_full.cpp
// 完全向量化（AVX512）的三角形对批量相交检测
// 编译：需要支持 AVX512F 的编译器，例如：
//   g++ -mavx512f -O3 -o program main_avx_speed.cpp Tri_avx_full.cpp
// 注意：不再依赖 Tri_16.cpp 中的标量实现，所有步骤均已向量化

#include <immintrin.h>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace tdbase
{

    static const float EPSILON = 1e-6f;

    // 64 字节对齐的 SOA 结构，存储 16 对三角形的顶点坐标
    struct alignas(64) TriPairBatch
    {
        float p1x[16], p1y[16], p1z[16];
        float q1x[16], q1y[16], q1z[16];
        float r1x[16], r1y[16], r1z[16];
        float p2x[16], p2y[16], p2z[16];
        float q2x[16], q2y[16], q2z[16];
        float r2x[16], r2y[16], r2z[16];
    };

    // ----------------------------------------------------------------------
    // 辅助函数：根据投影平面掩码从 3D 坐标中选择 2D 坐标 (x,y)
    // 输入：三个分量的向量，以及三个掩码（表示该三角形对应使用哪种投影）
    // 输出：一对 __m512，分别为所选平面的 x 坐标和 y 坐标
    // 投影规则：
    //   plane_x_mask 为真 -> 使用 (y, z)
    //   plane_y_mask 为真 -> 使用 (x, z)
    //   否则使用 (x, y)
    // ----------------------------------------------------------------------
    static inline std::pair<__m512, __m512> select_xy(
        __m512 x, __m512 y, __m512 z,
        __mmask16 plane_x_mask, __mmask16 plane_y_mask)
    {
        // x 坐标选择：
        // plane_x_mask -> y
        // plane_y_mask -> x
        // 否则 -> x
        __m512 x_coord = _mm512_mask_blend_ps(plane_x_mask,
                                              _mm512_mask_blend_ps(plane_y_mask, x, x), y);

        // y 坐标选择：
        // plane_x_mask -> z
        // plane_y_mask -> z
        // 否则 -> y
        __m512 y_coord = _mm512_mask_blend_ps(plane_x_mask,
                                              _mm512_mask_blend_ps(plane_y_mask, y, z), z);

        return {x_coord, y_coord};
    }

    // ----------------------------------------------------------------------
    // 向量化 2D 三角形相交检测（一次处理 16 对）
    // 输入：三角形1和三角形2在 3D 空间中的顶点坐标向量，
    //       以及用于选择投影平面的三个掩码。
    // 输出：一个 16 位掩码，每位为 1 表示对应三角形对在 2D 中相交。
    // ----------------------------------------------------------------------
    static __mmask16 intersect_2d_16(
        __m512 p1x, __m512 p1y, __m512 p1z,
        __m512 q1x, __m512 q1y, __m512 q1z,
        __m512 r1x, __m512 r1y, __m512 r1z,
        __m512 p2x, __m512 p2y, __m512 p2z,
        __m512 q2x, __m512 q2y, __m512 q2z,
        __m512 r2x, __m512 r2y, __m512 r2z,
        __mmask16 plane_x_mask, __mmask16 plane_y_mask, __mmask16 plane_z_mask)
    {
        // 根据平面掩码选择 2D 坐标
        auto [p1x2d, p1y2d] = select_xy(p1x, p1y, p1z, plane_x_mask, plane_y_mask);
        auto [q1x2d, q1y2d] = select_xy(q1x, q1y, q1z, plane_x_mask, plane_y_mask);
        auto [r1x2d, r1y2d] = select_xy(r1x, r1y, r1z, plane_x_mask, plane_y_mask);
        auto [p2x2d, p2y2d] = select_xy(p2x, p2y, p2z, plane_x_mask, plane_y_mask);
        auto [q2x2d, q2y2d] = select_xy(q2x, q2y, q2z, plane_x_mask, plane_y_mask);
        auto [r2x2d, r2y2d] = select_xy(r2x, r2y, r2z, plane_x_mask, plane_y_mask);

        // 最终结果掩码
        __mmask16 result = 0;

        // ========== 1. 顶点包含测试（使用重心坐标法） ==========

        // ---------- 三角形1的顶点是否在三角形2内 ----------
        // 计算三角形2的边向量
        __m512 v0x = _mm512_sub_ps(r2x2d, p2x2d);
        __m512 v0y = _mm512_sub_ps(r2y2d, p2y2d);
        __m512 v1x = _mm512_sub_ps(q2x2d, p2x2d);
        __m512 v1y = _mm512_sub_ps(q2y2d, p2y2d);

        // 点积
        __m512 dot00 = _mm512_fmadd_ps(v0x, v0x, _mm512_mul_ps(v0y, v0y));
        __m512 dot01 = _mm512_fmadd_ps(v0x, v1x, _mm512_mul_ps(v0y, v1y));
        __m512 dot11 = _mm512_fmadd_ps(v1x, v1x, _mm512_mul_ps(v1y, v1y));
        __m512 invDenom = _mm512_div_ps(_mm512_set1_ps(1.0f),
                                        _mm512_fmsub_ps(dot00, dot11, _mm512_mul_ps(dot01, dot01)));

        // 检查一个点是否在三角形2内的 lambda
        auto point_in_tri2 = [&](__m512 px, __m512 py) -> __mmask16
        {
            __m512 v2x = _mm512_sub_ps(px, p2x2d);
            __m512 v2y = _mm512_sub_ps(py, p2y2d);
            __m512 dot02 = _mm512_fmadd_ps(v0x, v2x, _mm512_mul_ps(v0y, v2y));
            __m512 dot12 = _mm512_fmadd_ps(v1x, v2x, _mm512_mul_ps(v1y, v2y));

            __m512 u = _mm512_mul_ps(_mm512_fmsub_ps(dot11, dot02, _mm512_mul_ps(dot01, dot12)), invDenom);
            __m512 v = _mm512_mul_ps(_mm512_fmsub_ps(dot00, dot12, _mm512_mul_ps(dot01, dot02)), invDenom);

            __m512 eps = _mm512_set1_ps(EPSILON);
            __mmask16 in = _mm512_cmp_ps_mask(u, eps, _CMP_GE_OQ) &
                           _mm512_cmp_ps_mask(v, eps, _CMP_GE_OQ) &
                           _mm512_cmp_ps_mask(_mm512_add_ps(u, v),
                                              _mm512_set1_ps(1.0f + EPSILON), _CMP_LE_OQ);

            // 排除退化三角形（面积为零）
            __m512 area2 = _mm512_fmsub_ps(v0x, v1y, _mm512_mul_ps(v0y, v1x));
            __mmask16 degenerate = _mm512_cmp_ps_mask(_mm512_abs_ps(area2),
                                                      _mm512_set1_ps(EPSILON * EPSILON), _CMP_LT_OQ);
            return in & ~degenerate;
        };

        // 检查三角形1的三个顶点
        result |= point_in_tri2(p1x2d, p1y2d);
        result |= point_in_tri2(q1x2d, q1y2d);
        result |= point_in_tri2(r1x2d, r1y2d);

        // ---------- 三角形2的顶点是否在三角形1内 ----------
        __m512 u0x = _mm512_sub_ps(r1x2d, p1x2d);
        __m512 u0y = _mm512_sub_ps(r1y2d, p1y2d);
        __m512 u1x = _mm512_sub_ps(q1x2d, p1x2d);
        __m512 u1y = _mm512_sub_ps(q1y2d, p1y2d);

        __m512 dot00_1 = _mm512_fmadd_ps(u0x, u0x, _mm512_mul_ps(u0y, u0y));
        __m512 dot01_1 = _mm512_fmadd_ps(u0x, u1x, _mm512_mul_ps(u0y, u1y));
        __m512 dot11_1 = _mm512_fmadd_ps(u1x, u1x, _mm512_mul_ps(u1y, u1y));
        __m512 invDenom_1 = _mm512_div_ps(_mm512_set1_ps(1.0f),
                                          _mm512_fmsub_ps(dot00_1, dot11_1, _mm512_mul_ps(dot01_1, dot01_1)));

        auto point_in_tri1 = [&](__m512 px, __m512 py) -> __mmask16
        {
            __m512 v2x = _mm512_sub_ps(px, p1x2d);
            __m512 v2y = _mm512_sub_ps(py, p1y2d);
            __m512 dot02 = _mm512_fmadd_ps(u0x, v2x, _mm512_mul_ps(u0y, v2y));
            __m512 dot12 = _mm512_fmadd_ps(u1x, v2x, _mm512_mul_ps(u1y, v2y));

            __m512 u = _mm512_mul_ps(_mm512_fmsub_ps(dot11_1, dot02, _mm512_mul_ps(dot01_1, dot12)), invDenom_1);
            __m512 v = _mm512_mul_ps(_mm512_fmsub_ps(dot00_1, dot12, _mm512_mul_ps(dot01_1, dot02)), invDenom_1);

            __m512 eps = _mm512_set1_ps(EPSILON);
            __mmask16 in = _mm512_cmp_ps_mask(u, eps, _CMP_GE_OQ) &
                           _mm512_cmp_ps_mask(v, eps, _CMP_GE_OQ) &
                           _mm512_cmp_ps_mask(_mm512_add_ps(u, v),
                                              _mm512_set1_ps(1.0f + EPSILON), _CMP_LE_OQ);

            __m512 area2 = _mm512_fmsub_ps(u0x, u1y, _mm512_mul_ps(u0y, u1x));
            __mmask16 degenerate = _mm512_cmp_ps_mask(_mm512_abs_ps(area2),
                                                      _mm512_set1_ps(EPSILON * EPSILON), _CMP_LT_OQ);
            return in & ~degenerate;
        };

        result |= point_in_tri1(p2x2d, p2y2d);
        result |= point_in_tri1(q2x2d, q2y2d);
        result |= point_in_tri1(r2x2d, r2y2d);

        // ========== 2. 边相交测试（定向法） ==========
        auto orient = [](__m512 ax, __m512 ay, __m512 bx, __m512 by, __m512 cx, __m512 cy) -> __m512
        {
            return _mm512_fmsub_ps(_mm512_sub_ps(ax, cx), _mm512_sub_ps(by, cy),
                                   _mm512_mul_ps(_mm512_sub_ps(ay, cy), _mm512_sub_ps(bx, cx)));
        };

        // 三角形1的三条边（起点）
        __m512 tri1_start_x[3] = {p1x2d, q1x2d, r1x2d};
        __m512 tri1_start_y[3] = {p1y2d, q1y2d, r1y2d};
        // 三角形1的三条边（终点，下一条边）
        __m512 tri1_end_x[3] = {q1x2d, r1x2d, p1x2d};
        __m512 tri1_end_y[3] = {q1y2d, r1y2d, p1y2d};

        __m512 tri2_start_x[3] = {p2x2d, q2x2d, r2x2d};
        __m512 tri2_start_y[3] = {p2y2d, q2y2d, r2y2d};
        __m512 tri2_end_x[3] = {q2x2d, r2x2d, p2x2d};
        __m512 tri2_end_y[3] = {q2y2d, r2y2d, p2y2d};

        __mmask16 edge_intersect = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                __m512 o1 = orient(tri1_start_x[i], tri1_start_y[i],
                                   tri1_end_x[i], tri1_end_y[i],
                                   tri2_start_x[j], tri2_start_y[j]);
                __m512 o2 = orient(tri1_start_x[i], tri1_start_y[i],
                                   tri1_end_x[i], tri1_end_y[i],
                                   tri2_end_x[j], tri2_end_y[j]);
                __m512 o3 = orient(tri2_start_x[j], tri2_start_y[j],
                                   tri2_end_x[j], tri2_end_y[j],
                                   tri1_start_x[i], tri1_start_y[i]);
                __m512 o4 = orient(tri2_start_x[j], tri2_start_y[j],
                                   tri2_end_x[j], tri2_end_y[j],
                                   tri1_end_x[i], tri1_end_y[i]);

                __mmask16 general = _mm512_cmp_ps_mask(_mm512_mul_ps(o1, o2), _mm512_setzero_ps(), _CMP_LT_OQ) &
                                    _mm512_cmp_ps_mask(_mm512_mul_ps(o3, o4), _mm512_setzero_ps(), _CMP_LT_OQ);
                edge_intersect |= general;
            }
        }

        result |= edge_intersect;
        return result;
    }

    // ----------------------------------------------------------------------
    // 批量相交检测（AVX512 完全向量化）
    // 输入：16 对三角形的 SOA 数据
    // 输出：16 个布尔结果
    // ----------------------------------------------------------------------
    bool TriIntBatch(const TriPairBatch &batch, bool results[16])
    {
        // 加载所有顶点坐标
        __m512 p1x = _mm512_load_ps(batch.p1x);
        __m512 p1y = _mm512_load_ps(batch.p1y);
        __m512 p1z = _mm512_load_ps(batch.p1z);
        __m512 q1x = _mm512_load_ps(batch.q1x);
        __m512 q1y = _mm512_load_ps(batch.q1y);
        __m512 q1z = _mm512_load_ps(batch.q1z);
        __m512 r1x = _mm512_load_ps(batch.r1x);
        __m512 r1y = _mm512_load_ps(batch.r1y);
        __m512 r1z = _mm512_load_ps(batch.r1z);
        __m512 p2x = _mm512_load_ps(batch.p2x);
        __m512 p2y = _mm512_load_ps(batch.p2y);
        __m512 p2z = _mm512_load_ps(batch.p2z);
        __m512 q2x = _mm512_load_ps(batch.q2x);
        __m512 q2y = _mm512_load_ps(batch.q2y);
        __m512 q2z = _mm512_load_ps(batch.q2z);
        __m512 r2x = _mm512_load_ps(batch.r2x);
        __m512 r2y = _mm512_load_ps(batch.r2y);
        __m512 r2z = _mm512_load_ps(batch.r2z);

        // 1. 计算三角形1的法线 N1
        __m512 v1x = _mm512_sub_ps(q1x, p1x);
        __m512 v1y = _mm512_sub_ps(q1y, p1y);
        __m512 v1z = _mm512_sub_ps(q1z, p1z);
        __m512 v2x = _mm512_sub_ps(r1x, p1x);
        __m512 v2y = _mm512_sub_ps(r1y, p1y);
        __m512 v2z = _mm512_sub_ps(r1z, p1z);
        __m512 N1x = _mm512_fmsub_ps(v1y, v2z, _mm512_mul_ps(v1z, v2y));
        __m512 N1y = _mm512_fmsub_ps(v1z, v2x, _mm512_mul_ps(v1x, v2z));
        __m512 N1z = _mm512_fmsub_ps(v1x, v2y, _mm512_mul_ps(v1y, v2x));

        // 2. 计算三角形2的法线 N2
        __m512 w1x = _mm512_sub_ps(q2x, p2x);
        __m512 w1y = _mm512_sub_ps(q2y, p2y);
        __m512 w1z = _mm512_sub_ps(q2z, p2z);
        __m512 w2x = _mm512_sub_ps(r2x, p2x);
        __m512 w2y = _mm512_sub_ps(r2y, p2y);
        __m512 w2z = _mm512_sub_ps(r2z, p2z);
        __m512 N2x = _mm512_fmsub_ps(w1y, w2z, _mm512_mul_ps(w1z, w2y));
        __m512 N2y = _mm512_fmsub_ps(w1z, w2x, _mm512_mul_ps(w1x, w2z));
        __m512 N2z = _mm512_fmsub_ps(w1x, w2y, _mm512_mul_ps(w1y, w2x));

        // 3. 三角形1顶点到平面2的距离（使用 r2 作为平面点）
        __m512 dx1 = _mm512_sub_ps(p1x, r2x);
        __m512 dy1 = _mm512_sub_ps(p1y, r2y);
        __m512 dz1 = _mm512_sub_ps(p1z, r2z);
        __m512 dp1 = _mm512_fmadd_ps(dx1, N2x, _mm512_fmadd_ps(dy1, N2y, _mm512_mul_ps(dz1, N2z)));
        __m512 dq1 = _mm512_fmadd_ps(_mm512_sub_ps(q1x, r2x), N2x,
                                     _mm512_fmadd_ps(_mm512_sub_ps(q1y, r2y), N2y,
                                                     _mm512_mul_ps(_mm512_sub_ps(q1z, r2z), N2z)));
        __m512 dr1 = _mm512_fmadd_ps(_mm512_sub_ps(r1x, r2x), N2x,
                                     _mm512_fmadd_ps(_mm512_sub_ps(r1y, r2y), N2y,
                                                     _mm512_mul_ps(_mm512_sub_ps(r1z, r2z), N2z)));

        // 4. 三角形2顶点到平面1的距离（使用 r1 作为平面点）
        __m512 dx2 = _mm512_sub_ps(p2x, r1x);
        __m512 dy2 = _mm512_sub_ps(p2y, r1y);
        __m512 dz2 = _mm512_sub_ps(p2z, r1z);
        __m512 dp2 = _mm512_fmadd_ps(dx2, N1x, _mm512_fmadd_ps(dy2, N1y, _mm512_mul_ps(dz2, N1z)));
        __m512 dq2 = _mm512_fmadd_ps(_mm512_sub_ps(q2x, r1x), N1x,
                                     _mm512_fmadd_ps(_mm512_sub_ps(q2y, r1y), N1y,
                                                     _mm512_mul_ps(_mm512_sub_ps(q2z, r1z), N1z)));
        __m512 dr2 = _mm512_fmadd_ps(_mm512_sub_ps(r2x, r1x), N1x,
                                     _mm512_fmadd_ps(_mm512_sub_ps(r2y, r1y), N1y,
                                                     _mm512_mul_ps(_mm512_sub_ps(r2z, r1z), N1z)));

        // 5. 早期排除：如果三角形完全在另一侧（所有距离同号且非零）
        __m512 zero = _mm512_setzero_ps();
        __mmask16 same_sign1 = (_mm512_cmp_ps_mask(dp1, zero, _CMP_GT_OQ) ==
                                _mm512_cmp_ps_mask(dq1, zero, _CMP_GT_OQ)) &&
                               (_mm512_cmp_ps_mask(dp1, zero, _CMP_GT_OQ) ==
                                _mm512_cmp_ps_mask(dr1, zero, _CMP_GT_OQ));
        __mmask16 any_zero1 = _mm512_cmp_ps_mask(_mm512_abs_ps(dp1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) |
                              _mm512_cmp_ps_mask(_mm512_abs_ps(dq1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) |
                              _mm512_cmp_ps_mask(_mm512_abs_ps(dr1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ);
        __mmask16 early_out1 = same_sign1 & ~any_zero1;

        __mmask16 same_sign2 = (_mm512_cmp_ps_mask(dp2, zero, _CMP_GT_OQ) ==
                                _mm512_cmp_ps_mask(dq2, zero, _CMP_GT_OQ)) &&
                               (_mm512_cmp_ps_mask(dp2, zero, _CMP_GT_OQ) ==
                                _mm512_cmp_ps_mask(dr2, zero, _CMP_GT_OQ));
        __mmask16 any_zero2 = _mm512_cmp_ps_mask(_mm512_abs_ps(dp2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) |
                              _mm512_cmp_ps_mask(_mm512_abs_ps(dq2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) |
                              _mm512_cmp_ps_mask(_mm512_abs_ps(dr2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ);
        __mmask16 early_out2 = same_sign2 & ~any_zero2;

        __mmask16 active_mask = ~(early_out1 | early_out2) & 0xFFFF;

        // 6. 共面检测：所有顶点到另一平面的距离都为零
        __mmask16 zero1_all = (_mm512_cmp_ps_mask(_mm512_abs_ps(dp1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) &
                               _mm512_cmp_ps_mask(_mm512_abs_ps(dq1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) &
                               _mm512_cmp_ps_mask(_mm512_abs_ps(dr1), _mm512_set1_ps(EPSILON), _CMP_LT_OQ));
        __mmask16 zero2_all = (_mm512_cmp_ps_mask(_mm512_abs_ps(dp2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) &
                               _mm512_cmp_ps_mask(_mm512_abs_ps(dq2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ) &
                               _mm512_cmp_ps_mask(_mm512_abs_ps(dr2), _mm512_set1_ps(EPSILON), _CMP_LT_OQ));
        __mmask16 coplanar_mask = zero1_all | zero2_all;

        // 7. 对非共面对进行分离轴测试
        __mmask16 intersect_mask = 0;
        if (active_mask != 0)
        {
            // 三角形1的边向量（已经计算过 v1, v2，还需 v3）
            __m512 e1_0x = v1x, e1_0y = v1y, e1_0z = v1z; // q1-p1
            __m512 e1_1x = _mm512_sub_ps(r1x, q1x);
            __m512 e1_1y = _mm512_sub_ps(r1y, q1y);
            __m512 e1_1z = _mm512_sub_ps(r1z, q1z); // r1-q1
            __m512 e1_2x = _mm512_sub_ps(p1x, r1x);
            __m512 e1_2y = _mm512_sub_ps(p1y, r1y);
            __m512 e1_2z = _mm512_sub_ps(p1z, r1z); // p1-r1

            // 三角形2的边向量
            __m512 e2_0x = w1x, e2_0y = w1y, e2_0z = w1z; // q2-p2
            __m512 e2_1x = _mm512_sub_ps(r2x, q2x);
            __m512 e2_1y = _mm512_sub_ps(r2y, q2y);
            __m512 e2_1z = _mm512_sub_ps(r2z, q2z); // r2-q2
            __m512 e2_2x = _mm512_sub_ps(p2x, r2x);
            __m512 e2_2y = _mm512_sub_ps(p2y, r2y);
            __m512 e2_2z = _mm512_sub_ps(p2z, r2z); // p2-r2

            // 三角形顶点数组
            __m512 tri1_x[3] = {p1x, q1x, r1x};
            __m512 tri1_y[3] = {p1y, q1y, r1y};
            __m512 tri1_z[3] = {p1z, q1z, r1z};
            __m512 tri2_x[3] = {p2x, q2x, r2x};
            __m512 tri2_y[3] = {p2y, q2y, r2y};
            __m512 tri2_z[3] = {p2z, q2z, r2z};

            // 投影区间计算 lambda
            auto project = [&](const __m512 &ax, const __m512 &ay, const __m512 &az,
                               const __m512 *tx, const __m512 *ty, const __m512 *tz)
            {
                __m512 dot0 = _mm512_fmadd_ps(tx[0], ax, _mm512_fmadd_ps(ty[0], ay, _mm512_mul_ps(tz[0], az)));
                __m512 dot1 = _mm512_fmadd_ps(tx[1], ax, _mm512_fmadd_ps(ty[1], ay, _mm512_mul_ps(tz[1], az)));
                __m512 dot2 = _mm512_fmadd_ps(tx[2], ax, _mm512_fmadd_ps(ty[2], ay, _mm512_mul_ps(tz[2], az)));
                __m512 minv = _mm512_min_ps(_mm512_min_ps(dot0, dot1), dot2);
                __m512 maxv = _mm512_max_ps(_mm512_max_ps(dot0, dot1), dot2);
                return std::pair<__m512, __m512>(minv, maxv);
            };

            // 测试单个轴，更新活跃掩码
            auto test_axis = [&](const __m512 &ax, const __m512 &ay, const __m512 &az, __mmask16 &mask)
            {
                if (mask == 0)
                    return;
                auto [min1, max1] = project(ax, ay, az, tri1_x, tri1_y, tri1_z);
                auto [min2, max2] = project(ax, ay, az, tri2_x, tri2_y, tri2_z);
                __m512 eps = _mm512_set1_ps(EPSILON);
                __mmask16 no_overlap = _mm512_cmp_ps_mask(max1, _mm512_sub_ps(min2, eps), _CMP_LT_OQ) |
                                       _mm512_cmp_ps_mask(max2, _mm512_sub_ps(min1, eps), _CMP_LT_OQ);
                mask &= ~no_overlap;
            };

            // 轴1: N1
            test_axis(N1x, N1y, N1z, active_mask);
            // 轴2: N2
            test_axis(N2x, N2y, N2z, active_mask);

            // 轴3-11: 边叉积
            __m512 edge1_x[3] = {e1_0x, e1_1x, e1_2x};
            __m512 edge1_y[3] = {e1_0y, e1_1y, e1_2y};
            __m512 edge1_z[3] = {e1_0z, e1_1z, e1_2z};
            __m512 edge2_x[3] = {e2_0x, e2_1x, e2_2x};
            __m512 edge2_y[3] = {e2_0y, e2_1y, e2_2y};
            __m512 edge2_z[3] = {e2_0z, e2_1z, e2_2z};

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    __m512 ax = _mm512_fmsub_ps(edge1_y[i], edge2_z[j], _mm512_mul_ps(edge1_z[i], edge2_y[j]));
                    __m512 ay = _mm512_fmsub_ps(edge1_z[i], edge2_x[j], _mm512_mul_ps(edge1_x[i], edge2_z[j]));
                    __m512 az = _mm512_fmsub_ps(edge1_x[i], edge2_y[j], _mm512_mul_ps(edge1_y[i], edge2_x[j]));
                    __m512 len2 = _mm512_fmadd_ps(ax, ax, _mm512_fmadd_ps(ay, ay, _mm512_mul_ps(az, az)));
                    __mmask16 nonzero = _mm512_cmp_ps_mask(len2, _mm512_set1_ps(EPSILON * EPSILON), _CMP_GT_OQ);
                    if (nonzero == 0)
                        continue;
                    // 归一化（可选，但可提高数值稳定性）
                    __m512 inv_len = _mm512_rsqrt14_ps(len2);
                    ax = _mm512_mul_ps(ax, inv_len);
                    ay = _mm512_mul_ps(ay, inv_len);
                    az = _mm512_mul_ps(az, inv_len);
                    test_axis(ax, ay, az, active_mask);
                }
            }

            // 通过所有分离轴测试的即为相交
            intersect_mask = active_mask;
        }

        // 8. 处理共面对
        if (coplanar_mask != 0)
        {
            // 为每个共面对确定投影平面（基于法线最大绝对值分量）
            __m512 absN1x = _mm512_abs_ps(N1x);
            __m512 absN1y = _mm512_abs_ps(N1y);
            __m512 absN1z = _mm512_abs_ps(N1z);
            __m512 max_comp = _mm512_max_ps(_mm512_max_ps(absN1x, absN1y), absN1z);

            __mmask16 plane_x = _mm512_cmp_ps_mask(absN1x, max_comp, _CMP_EQ_OQ);
            __mmask16 plane_y = _mm512_cmp_ps_mask(absN1y, max_comp, _CMP_EQ_OQ);
            __mmask16 plane_z = _mm512_cmp_ps_mask(absN1z, max_comp, _CMP_EQ_OQ);
            // 处理多个分量相等的情况：优先选择 x，其次 y，最后 z
            plane_y &= ~plane_x;
            plane_z &= ~(plane_x | plane_y);

            // 调用向量化 2D 相交检测
            __mmask16 coplanar_result = intersect_2d_16(
                p1x, p1y, p1z, q1x, q1y, q1z, r1x, r1y, r1z,
                p2x, p2y, p2z, q2x, q2y, q2z, r2x, r2y, r2z,
                plane_x, plane_y, plane_z);

            // 合并结果
            intersect_mask = (intersect_mask & ~coplanar_mask) | (coplanar_result & coplanar_mask);
        }

        // 9. 输出结果
        for (int i = 0; i < 16; ++i)
        {
            results[i] = (intersect_mask >> i) & 1;
        }

        return true;
    }

    // 单对接口（调用批量接口）
    bool TriInt(const float *data1, const float *data2)
    {
        TriPairBatch batch;
        // 填充第0对
        batch.p1x[0] = data1[0];
        batch.p1y[0] = data1[1];
        batch.p1z[0] = data1[2];
        batch.q1x[0] = data1[3];
        batch.q1y[0] = data1[4];
        batch.q1z[0] = data1[5];
        batch.r1x[0] = data1[6];
        batch.r1y[0] = data1[7];
        batch.r1z[0] = data1[8];
        batch.p2x[0] = data2[0];
        batch.p2y[0] = data2[1];
        batch.p2z[0] = data2[2];
        batch.q2x[0] = data2[3];
        batch.q2y[0] = data2[4];
        batch.q2z[0] = data2[5];
        batch.r2x[0] = data2[6];
        batch.r2y[0] = data2[7];
        batch.r2z[0] = data2[8];
        // 其余15对填充远距离安全值（确保不相交）
        for (int i = 1; i < 16; ++i)
        {
            batch.p1x[i] = batch.p1y[i] = batch.p1z[i] = 1e6f;
            batch.q1x[i] = batch.q1y[i] = batch.q1z[i] = 1e6f;
            batch.r1x[i] = batch.r1y[i] = batch.r1z[i] = 1e6f;
            batch.p2x[i] = batch.p2y[i] = batch.p2z[i] = 1e6f + 1;
            batch.q2x[i] = batch.q2y[i] = batch.q2z[i] = 1e6f + 1;
            batch.r2x[i] = batch.r2y[i] = batch.r2z[i] = 1e6f + 1;
        }
        bool results[16];
        TriIntBatch(batch, results);
        return results[0];
    }

} // namespace tdbase