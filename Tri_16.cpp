// TriInt_ds_soa.cpp
// 修正版：使用SOA模式存储16对三角形，提供批量相交检测接口（标量实现）
// 编译：与main.cpp一起编译

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <algorithm>
#include <cfloat>

typedef float PQP_REAL; // 使用单精度

namespace tdbase
{
    static const PQP_REAL EPSILON = 1e-6f; // 合适的容差

    // ======================= 向量操作函数 =======================
    static inline void vec_sub(PQP_REAL dest[3], const PQP_REAL v1[3], const PQP_REAL v2[3])
    {
        dest[0] = v1[0] - v2[0];
        dest[1] = v1[1] - v2[1];
        dest[2] = v1[2] - v2[2];
    }

    static inline void vec_cross(PQP_REAL dest[3], const PQP_REAL v1[3], const PQP_REAL v2[3])
    {
        dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
        dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
        dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
    }

    static inline PQP_REAL vec_dot(const PQP_REAL v1[3], const PQP_REAL v2[3])
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    static inline void vec_scale(PQP_REAL dest[3], PQP_REAL alpha, const PQP_REAL v[3])
    {
        dest[0] = alpha * v[0];
        dest[1] = alpha * v[1];
        dest[2] = alpha * v[2];
    }

    static inline void vec_add(PQP_REAL dest[3], const PQP_REAL v1[3], const PQP_REAL v2[3])
    {
        dest[0] = v1[0] + v2[0];
        dest[1] = v1[1] + v2[1];
        dest[2] = v1[2] + v2[2];
    }

    static inline PQP_REAL orient_2d(const PQP_REAL a[2], const PQP_REAL b[2], const PQP_REAL c[2])
    {
        return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
    }

    static inline PQP_REAL point_plane_distance(const PQP_REAL point[3],
                                                const PQP_REAL plane_point[3],
                                                const PQP_REAL normal[3])
    {
        PQP_REAL v[3];
        vec_sub(v, point, plane_point);
        return vec_dot(v, normal);
    }

    static inline void compute_triangle_normal(const PQP_REAL p[3], const PQP_REAL q[3], const PQP_REAL r[3],
                                               PQP_REAL normal[3])
    {
        PQP_REAL v1[3], v2[3];
        vec_sub(v1, q, p);
        vec_sub(v2, r, p);
        vec_cross(normal, v1, v2);
    }

    static inline bool all_same_side(PQP_REAL d1, PQP_REAL d2, PQP_REAL d3)
    {
        if (fabs(d1) <= EPSILON || fabs(d2) <= EPSILON || fabs(d3) <= EPSILON)
            return false;
        return ((d1 * d2) > 0.0f) && ((d1 * d3) > 0.0f);
    }

    static inline void normalize_triangle_2d(PQP_REAL p[2], PQP_REAL q[2], PQP_REAL r[2])
    {
        if (orient_2d(p, q, r) < 0.0f)
        {
            std::swap(q[0], r[0]);
            std::swap(q[1], r[1]);
        }
    }

    static bool point_in_triangle_2d(const PQP_REAL p[2],
                                     const PQP_REAL t1[2], const PQP_REAL t2[2], const PQP_REAL t3[2])
    {
        PQP_REAL v0[2] = {t3[0] - t1[0], t3[1] - t1[1]};
        PQP_REAL v1[2] = {t2[0] - t1[0], t2[1] - t1[1]};
        PQP_REAL v2[2] = {p[0] - t1[0], p[1] - t1[1]};

        PQP_REAL dot00 = v0[0] * v0[0] + v0[1] * v0[1];
        PQP_REAL dot01 = v0[0] * v1[0] + v0[1] * v1[1];
        PQP_REAL dot02 = v0[0] * v2[0] + v0[1] * v2[1];
        PQP_REAL dot11 = v1[0] * v1[0] + v1[1] * v1[1];
        PQP_REAL dot12 = v1[0] * v2[0] + v1[1] * v2[1];

        PQP_REAL invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        PQP_REAL u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        PQP_REAL v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        const PQP_REAL eps = EPSILON;
        return (u >= -eps) && (v >= -eps) && (u + v <= 1.0f + eps);
    }

    static bool segments_intersect_2d(const PQP_REAL a1[2], const PQP_REAL a2[2],
                                      const PQP_REAL b1[2], const PQP_REAL b2[2])
    {
        PQP_REAL o1 = orient_2d(a1, a2, b1);
        PQP_REAL o2 = orient_2d(a1, a2, b2);
        PQP_REAL o3 = orient_2d(b1, b2, a1);
        PQP_REAL o4 = orient_2d(b1, b2, a2);

        if (fabs(o1) < EPSILON && point_in_triangle_2d(b1, a1, a2, a1))
            return true;
        if (fabs(o2) < EPSILON && point_in_triangle_2d(b2, a1, a2, a1))
            return true;
        if (fabs(o3) < EPSILON && point_in_triangle_2d(a1, b1, b2, b1))
            return true;
        if (fabs(o4) < EPSILON && point_in_triangle_2d(a2, b1, b2, b1))
            return true;

        return (o1 * o2 < -EPSILON) && (o3 * o4 < -EPSILON);
    }

    static bool ccw_tri_tri_intersection_2d(PQP_REAL p1[2], PQP_REAL q1[2], PQP_REAL r1[2],
                                            PQP_REAL p2[2], PQP_REAL q2[2], PQP_REAL r2[2])
    {
        // 测试顶点包含
        if (point_in_triangle_2d(p1, p2, q2, r2) ||
            point_in_triangle_2d(q1, p2, q2, r2) ||
            point_in_triangle_2d(r1, p2, q2, r2) ||
            point_in_triangle_2d(p2, p1, q1, r1) ||
            point_in_triangle_2d(q2, p1, q1, r1) ||
            point_in_triangle_2d(r2, p1, q1, r1))
            return true;

        // 测试边相交
        PQP_REAL edges1[3][2] = {{p1[0], p1[1]}, {q1[0], q1[1]}, {r1[0], r1[1]}};
        PQP_REAL edges2[3][2] = {{p2[0], p2[1]}, {q2[0], q2[1]}, {r2[0], r2[1]}};

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                PQP_REAL a1[2] = {edges1[i][0], edges1[i][1]};
                PQP_REAL a2[2] = {edges1[(i + 1) % 3][0], edges1[(i + 1) % 3][1]};
                PQP_REAL b1[2] = {edges2[j][0], edges2[j][1]};
                PQP_REAL b2[2] = {edges2[(j + 1) % 3][0], edges2[(j + 1) % 3][1]};

                if (segments_intersect_2d(a1, a2, b1, b2))
                    return true;
            }
        }

        return false;
    }

    static int tri_tri_overlap_test_2d(PQP_REAL p1[2], PQP_REAL q1[2], PQP_REAL r1[2],
                                       PQP_REAL p2[2], PQP_REAL q2[2], PQP_REAL r2[2])
    {
        PQP_REAL p1c[2], q1c[2], r1c[2];
        PQP_REAL p2c[2], q2c[2], r2c[2];

        memcpy(p1c, p1, 2 * sizeof(PQP_REAL));
        memcpy(q1c, q1, 2 * sizeof(PQP_REAL));
        memcpy(r1c, r1, 2 * sizeof(PQP_REAL));
        memcpy(p2c, p2, 2 * sizeof(PQP_REAL));
        memcpy(q2c, q2, 2 * sizeof(PQP_REAL));
        memcpy(r2c, r2, 2 * sizeof(PQP_REAL));

        normalize_triangle_2d(p1c, q1c, r1c);
        normalize_triangle_2d(p2c, q2c, r2c);

        return ccw_tri_tri_intersection_2d(p1c, q1c, r1c, p2c, q2c, r2c) ? 1 : 0;
    }

    int coplanar_tri_tri3d(const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
                           const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
                           const PQP_REAL N1[3], const PQP_REAL N2[3])
    {
        PQP_REAL n_x = fabs(N1[0]);
        PQP_REAL n_y = fabs(N1[1]);
        PQP_REAL n_z = fabs(N1[2]);

        PQP_REAL P1[2], Q1[2], R1[2];
        PQP_REAL P2[2], Q2[2], R2[2];

        if ((n_x > n_z) && (n_x >= n_y))
        {
            P1[0] = q1[2]; P1[1] = q1[1];
            Q1[0] = p1[2]; Q1[1] = p1[1];
            R1[0] = r1[2]; R1[1] = r1[1];
            P2[0] = q2[2]; P2[1] = q2[1];
            Q2[0] = p2[2]; Q2[1] = p2[1];
            R2[0] = r2[2]; R2[1] = r2[1];
        }
        else if ((n_y > n_z) && (n_y >= n_x))
        {
            P1[0] = q1[0]; P1[1] = q1[2];
            Q1[0] = p1[0]; Q1[1] = p1[2];
            R1[0] = r1[0]; R1[1] = r1[2];
            P2[0] = q2[0]; P2[1] = q2[2];
            Q2[0] = p2[0]; Q2[1] = p2[2];
            R2[0] = r2[0]; R2[1] = r2[2];
        }
        else
        {
            P1[0] = p1[0]; P1[1] = p1[1];
            Q1[0] = q1[0]; Q1[1] = q1[1];
            R1[0] = r1[0]; R1[1] = r1[1];
            P2[0] = p2[0]; P2[1] = p2[1];
            Q2[0] = q2[0]; Q2[1] = q2[1];
            R2[0] = r2[0]; R2[1] = r2[1];
        }

        return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);
    }

    static void project_triangle(const PQP_REAL tri[3][3], const PQP_REAL axis[3],
                                 PQP_REAL &min_val, PQP_REAL &max_val)
    {
        min_val = vec_dot(tri[0], axis);
        max_val = min_val;
        for (int i = 1; i < 3; ++i) {
            PQP_REAL val = vec_dot(tri[i], axis);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    static bool tri_tri_overlap_test_3d_non_coplanar(
        const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
        const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
        const PQP_REAL N1[3], const PQP_REAL N2[3],
        PQP_REAL /*dp1*/, PQP_REAL /*dq1*/, PQP_REAL /*dr1*/,
        PQP_REAL /*dp2*/, PQP_REAL /*dq2*/, PQP_REAL /*dr2*/)
    {
        const PQP_REAL tri1[3][3] = {{p1[0],p1[1],p1[2]}, {q1[0],q1[1],q1[2]}, {r1[0],r1[1],r1[2]}};
        const PQP_REAL tri2[3][3] = {{p2[0],p2[1],p2[2]}, {q2[0],q2[1],q2[2]}, {r2[0],r2[1],r2[2]}};

        PQP_REAL axes[15][3];
        int axis_cnt = 0;

        axes[axis_cnt][0] = N1[0]; axes[axis_cnt][1] = N1[1]; axes[axis_cnt][2] = N1[2];
        axis_cnt++;
        axes[axis_cnt][0] = N2[0]; axes[axis_cnt][1] = N2[1]; axes[axis_cnt][2] = N2[2];
        axis_cnt++;

        PQP_REAL edges1[3][3], edges2[3][3];
        vec_sub(edges1[0], q1, p1);
        vec_sub(edges1[1], r1, q1);
        vec_sub(edges1[2], p1, r1);
        vec_sub(edges2[0], q2, p2);
        vec_sub(edges2[1], r2, q2);
        vec_sub(edges2[2], p2, r2);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                PQP_REAL axis[3];
                vec_cross(axis, edges1[i], edges2[j]);
                PQP_REAL len = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
                if (len > EPSILON) {
                    axis[0] /= len; axis[1] /= len; axis[2] /= len;
                    axes[axis_cnt][0] = axis[0];
                    axes[axis_cnt][1] = axis[1];
                    axes[axis_cnt][2] = axis[2];
                    axis_cnt++;
                }
            }
        }

        for (int i = 0; i < axis_cnt; ++i) {
            PQP_REAL min1, max1, min2, max2;
            project_triangle(tri1, axes[i], min1, max1);
            project_triangle(tri2, axes[i], min2, max2);
            if (max1 < min2 - EPSILON || max2 < min1 - EPSILON)
                return false;
        }
        return true;
    }

    int tri_tri_overlap_test_3d(PQP_REAL p1[3], PQP_REAL q1[3], PQP_REAL r1[3],
                                PQP_REAL p2[3], PQP_REAL q2[3], PQP_REAL r2[3])
    {
        PQP_REAL N1[3], N2[3];
        compute_triangle_normal(p1, q1, r1, N1);
        compute_triangle_normal(p2, q2, r2, N2);

        PQP_REAL dp1 = point_plane_distance(p1, r2, N2);
        PQP_REAL dq1 = point_plane_distance(q1, r2, N2);
        PQP_REAL dr1 = point_plane_distance(r1, r2, N2);

        if (all_same_side(dp1, dq1, dr1))
            return 0;

        PQP_REAL dp2 = point_plane_distance(p2, r1, N1);
        PQP_REAL dq2 = point_plane_distance(q2, r1, N1);
        PQP_REAL dr2 = point_plane_distance(r2, r1, N1);

        if (all_same_side(dp2, dq2, dr2))
            return 0;

        bool coplanar_case = (fabs(dp1) < EPSILON && fabs(dq1) < EPSILON && fabs(dr1) < EPSILON) ||
                             (fabs(dp2) < EPSILON && fabs(dq2) < EPSILON && fabs(dr2) < EPSILON);

        if (coplanar_case)
        {
            return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2);
        }

        return tri_tri_overlap_test_3d_non_coplanar(p1, q1, r1, p2, q2, r2,
                                                    N1, N2, dp1, dq1, dr1, dp2, dq2, dr2)
                   ? 1 : 0;
    }

    // ======================= SOA 数据结构（16对三角形） =======================
    struct alignas(16) TriPairBatch   // 16字节对齐便于后续SIMD，但不强制
    {
        // 三角形1的顶点坐标（每个分量独立存储）
        float p1x[16], p1y[16], p1z[16];
        float q1x[16], q1y[16], q1z[16];
        float r1x[16], r1y[16], r1z[16];
        // 三角形2的顶点坐标
        float p2x[16], p2y[16], p2z[16];
        float q2x[16], q2y[16], q2z[16];
        float r2x[16], r2y[16], r2z[16];
    };

    // 批量相交检测：输入16对三角形（SOA布局），输出结果数组（长度16）
    // 返回true表示所有计算完成（总是成功）
    bool TriIntBatch(const TriPairBatch& batch, bool results[16])
    {
        for (int i = 0; i < 16; ++i)
        {
            // 从SOA中提取第i对三角形的顶点
            PQP_REAL p1[3] = {batch.p1x[i], batch.p1y[i], batch.p1z[i]};
            PQP_REAL q1[3] = {batch.q1x[i], batch.q1y[i], batch.q1z[i]};
            PQP_REAL r1[3] = {batch.r1x[i], batch.r1y[i], batch.r1z[i]};
            PQP_REAL p2[3] = {batch.p2x[i], batch.p2y[i], batch.p2z[i]};
            PQP_REAL q2[3] = {batch.q2x[i], batch.q2y[i], batch.q2z[i]};
            PQP_REAL r2[3] = {batch.r2x[i], batch.r2y[i], batch.r2z[i]};

            // 调用原有的单对相交检测函数
            results[i] = (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0);
        }
        return true;
    }

    // 保留原有接口：计算一对三角形是否相交
    bool TriInt(const PQP_REAL *data1, const PQP_REAL *data2)
    {
        // 构建只包含一对三角形的SOA结构
        TriPairBatch batch;
        // 填充第0对
        batch.p1x[0] = data1[0]; batch.p1y[0] = data1[1]; batch.p1z[0] = data1[2];
        batch.q1x[0] = data1[3]; batch.q1y[0] = data1[4]; batch.q1z[0] = data1[5];
        batch.r1x[0] = data1[6]; batch.r1y[0] = data1[7]; batch.r1z[0] = data1[8];
        batch.p2x[0] = data2[0]; batch.p2y[0] = data2[1]; batch.p2z[0] = data2[2];
        batch.q2x[0] = data2[3]; batch.q2y[0] = data2[4]; batch.q2z[0] = data2[5];
        batch.r2x[0] = data2[6]; batch.r2y[0] = data2[7]; batch.r2z[0] = data2[8];
        // 其余对填充默认值（不会被使用）
        for (int i = 1; i < 16; ++i) {
            batch.p1x[i] = batch.p1y[i] = batch.p1z[i] = 0;
            batch.q1x[i] = batch.q1y[i] = batch.q1z[i] = 0;
            batch.r1x[i] = batch.r1y[i] = batch.r1z[i] = 0;
            batch.p2x[i] = batch.p2y[i] = batch.p2z[i] = 0;
            batch.q2x[i] = batch.q2y[i] = batch.q2z[i] = 0;
            batch.r2x[i] = batch.r2y[i] = batch.r2z[i] = 0;
        }
        bool results[16];
        TriIntBatch(batch, results);
        return results[0];
    }

    // ======================= 线段构造相关函数（原代码完整） =======================
    struct IntersectionSegment
    {
        PQP_REAL source[3];
        PQP_REAL target[3];
        bool coplanar;
    };

    static bool construct_intersection_segment(
        const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
        const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
        const PQP_REAL N1[3], const PQP_REAL N2[3],
        IntersectionSegment &result)
    {
        PQP_REAL v1[3], v2[3], v[3], N[3];
        PQP_REAL alpha;

        vec_sub(v1, q1, p1);
        vec_sub(v2, r2, p1);
        vec_cross(N, v1, v2);
        vec_sub(v, p2, p1);

        if (vec_dot(v, N) > EPSILON)
        {
            vec_sub(v1, r1, p1);
            vec_cross(N, v1, v2);
            if (vec_dot(v, N) <= EPSILON)
            {
                vec_sub(v2, q2, p1);
                vec_cross(N, v1, v2);
                if (vec_dot(v, N) > EPSILON)
                {
                    vec_sub(v1, p1, p2);
                    vec_sub(v2, p1, r1);
                    PQP_REAL denom = vec_dot(v2, N2);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N2) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.source, p1, v1);

                    vec_sub(v1, p2, p1);
                    vec_sub(v2, p2, r2);
                    denom = vec_dot(v2, N1);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N1) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.target, p2, v1);
                    return true;
                }
                else
                {
                    vec_sub(v1, p2, p1);
                    vec_sub(v2, p2, q2);
                    PQP_REAL denom = vec_dot(v2, N1);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N1) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.source, p2, v1);

                    vec_sub(v1, p2, p1);
                    vec_sub(v2, p2, r2);
                    denom = vec_dot(v2, N1);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N1) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.target, p2, v1);
                    return true;
                }
            }
            else
            {
                return false;
            }
        }
        else
        {
            vec_sub(v2, q2, p1);
            vec_cross(N, v1, v2);
            if (vec_dot(v, N) < -EPSILON)
            {
                return false;
            }
            else
            {
                vec_sub(v1, r1, p1);
                vec_cross(N, v1, v2);
                if (vec_dot(v, N) >= -EPSILON)
                {
                    vec_sub(v1, p1, p2);
                    vec_sub(v2, p1, r1);
                    PQP_REAL denom = vec_dot(v2, N2);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N2) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.source, p1, v1);

                    vec_sub(v1, p1, p2);
                    vec_sub(v2, p1, q1);
                    denom = vec_dot(v2, N2);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N2) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.target, p1, v1);
                    return true;
                }
                else
                {
                    vec_sub(v1, p2, p1);
                    vec_sub(v2, p2, q2);
                    PQP_REAL denom = vec_dot(v2, N1);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N1) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.source, p2, v1);

                    vec_sub(v1, p1, p2);
                    vec_sub(v2, p1, q1);
                    denom = vec_dot(v2, N2);
                    if (fabs(denom) < EPSILON) return false;
                    alpha = vec_dot(v1, N2) / denom;
                    vec_scale(v1, alpha, v2);
                    vec_sub(result.target, p1, v1);
                    return true;
                }
            }
        }
    }

    static bool tri_tri_intersection_with_segment(
        const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
        const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
        IntersectionSegment &result)
    {
        PQP_REAL N1[3], N2[3];
        compute_triangle_normal(p1, q1, r1, N1);
        compute_triangle_normal(p2, q2, r2, N2);

        PQP_REAL dp1 = point_plane_distance(p1, r2, N2);
        PQP_REAL dq1 = point_plane_distance(q1, r2, N2);
        PQP_REAL dr1 = point_plane_distance(r1, r2, N2);

        if (all_same_side(dp1, dq1, dr1))
            return false;

        PQP_REAL dp2 = point_plane_distance(p2, r1, N1);
        PQP_REAL dq2 = point_plane_distance(q2, r1, N1);
        PQP_REAL dr2 = point_plane_distance(r2, r1, N1);

        if (all_same_side(dp2, dq2, dr2))
            return false;

        bool coplanar_case = (fabs(dp1) < EPSILON && fabs(dq1) < EPSILON && fabs(dr1) < EPSILON) ||
                             (fabs(dp2) < EPSILON && fabs(dq2) < EPSILON && fabs(dr2) < EPSILON);

        if (coplanar_case)
        {
            result.coplanar = true;
            result.source[0] = result.source[1] = result.source[2] = 0;
            result.target[0] = result.target[1] = result.target[2] = 0;
            return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2) != 0;
        }

        if (tri_tri_overlap_test_3d_non_coplanar(p1, q1, r1, p2, q2, r2,
                                                 N1, N2, dp1, dq1, dr1, dp2, dq2, dr2))
        {
            result.coplanar = false;
            if (!construct_intersection_segment(p1, q1, r1, p2, q2, r2, N1, N2, result))
            {
                // 构造失败，输出零线段，但返回 true
                result.source[0] = result.source[1] = result.source[2] = 0;
                result.target[0] = result.target[1] = result.target[2] = 0;
            }
            return true;
        }

        return false;
    }

    bool TriIntWithSegment(const PQP_REAL *data1, const PQP_REAL *data2,
                           PQP_REAL source[3], PQP_REAL target[3])
    {
        const PQP_REAL *p1 = data1;
        const PQP_REAL *q1 = data1 + 3;
        const PQP_REAL *r1 = data1 + 6;
        const PQP_REAL *p2 = data2;
        const PQP_REAL *q2 = data2 + 3;
        const PQP_REAL *r2 = data2 + 6;

        PQP_REAL p1_copy[3] = {p1[0], p1[1], p1[2]};
        PQP_REAL q1_copy[3] = {q1[0], q1[1], q1[2]};
        PQP_REAL r1_copy[3] = {r1[0], r1[1], r1[2]};
        PQP_REAL p2_copy[3] = {p2[0], p2[1], p2[2]};
        PQP_REAL q2_copy[3] = {q2[0], q2[1], q2[2]};
        PQP_REAL r2_copy[3] = {r2[0], r2[1], r2[2]};

        IntersectionSegment seg;
        if (tri_tri_intersection_with_segment(p1_copy, q1_copy, r1_copy, p2_copy, q2_copy, r2_copy, seg))
        {
            source[0] = seg.source[0];
            source[1] = seg.source[1];
            source[2] = seg.source[2];
            target[0] = seg.target[0];
            target[1] = seg.target[1];
            target[2] = seg.target[2];
            return true;
        }

        return false;
    }

} // namespace tdbase