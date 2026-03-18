// 由 Aaron 修改，用于更好地检测共面情况

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <memory.h>
#include <time.h>

// 使用 PQP_REAL 作为浮点类型，在 PQP_Compile.h 中定义
#include "PQP_Compile.h"

namespace tdbase
{

// 零值测试宏：如果值等于0则返回true
#define ZERO_TEST(x) (x == 0)
  // 另一种零值测试方式：使用epsilon进行比较
  // #define ZERO_TEST(x)  ((x) > -0.001 && (x) < .001)

  /* 函数原型声明 */

  // 三维三角形-三角形重叠测试函数
  // 参数：两个三角形的顶点坐标 (p1,q1,r1) 和 (p2,q2,r2)
  // 返回值：1 表示重叠，0 表示不重叠
  int tri_tri_overlap_test_3d(PQP_REAL p1[3], PQP_REAL q1[3], PQP_REAL r1[3],
                              PQP_REAL p2[3], PQP_REAL q2[3], PQP_REAL r2[3]);

  // 共面三角形-三角形三维重叠测试
  // 参数：两个三角形的顶点坐标及其法向量
  // 返回值：1 表示共面且重叠，0 表示不重叠
  int coplanar_tri_tri3d(const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
                         const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
                         const PQP_REAL N1[3], const PQP_REAL N2[3]);

  // 二维三角形-三角形重叠测试函数
  // 参数：两个二维三角形的顶点坐标
  // 返回值：1 表示重叠，0 表示不重叠
  int tri_tri_overlap_test_2d(PQP_REAL p1[2], PQP_REAL q1[2], PQP_REAL r1[2],
                              PQP_REAL p2[2], PQP_REAL q2[2], PQP_REAL r2[2]);

/* coplanar_tri_tri3d 函数说明：
 * 该函数检测两个三角形是否共面
 * source 和 target 是相交线段的端点（如果存在相交的话）
 */

/* 三维向量操作宏定义 */

// CROSS: 计算两个三维向量的叉积，结果存入 dest
// dest = v1 × v2
#define CROSS(dest, v1, v2)                \
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

// DOT: 计算两个三维向量的点积
// 返回 v1 · v2
#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

// SUB: 计算两个三维向量的差
// dest = v1 - v2
#define SUB(dest, v1, v2)  \
  dest[0] = v1[0] - v2[0]; \
  dest[1] = v1[1] - v2[1]; \
  dest[2] = v1[2] - v2[2];

// SCALAR: 计算标量与向量的乘积
// dest = alpha * v
#define SCALAR(dest, alpha, v) \
  dest[0] = alpha * v[0];      \
  dest[1] = alpha * v[1];      \
  dest[2] = alpha * v[2];

// CHECK_MIN_MAX: 检查两个三角形在某个方向上的最小-最大值是否分离
// 如果分离则返回0（不重叠），否则返回1（重叠）
// 这是一种分离轴测试（Separating Axis Test）的实现
#define CHECK_MIN_MAX(p1, q1, r1, p2, q2, r2) \
  {                                           \
    SUB(v1, p2, q1)                           \
    SUB(v2, p1, q1)                           \
    CROSS(N1, v1, v2)                         \
    SUB(v1, q2, q1)                           \
    if (DOT(v1, N1) > 0.0f)                   \
      return 0;                               \
    SUB(v1, p2, p1)                           \
    SUB(v2, r1, p1)                           \
    CROSS(N1, v1, v2)                         \
    SUB(v1, r2, p1)                           \
    if (DOT(v1, N1) > 0.0f)                   \
      return 0;                               \
    else                                      \
      return 1;                               \
  }

  /* T2 顶点的规范排列宏
   * 根据 dp2, dq2, dr2 的符号，对 T2 的顶点进行重新排序
   * 以便进行一致的分离轴测试
   */

#define TRI_TRI_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)            \
  {                                                                  \
    if (dp2 > 0.0f)                                                  \
    {                                                                \
      if (dq2 > 0.0f)                                                \
        CHECK_MIN_MAX(p1, r1, q1, r2, p2, q2)                        \
      else if (dr2 > 0.0f)                                           \
        CHECK_MIN_MAX(p1, r1, q1, q2, r2, p2)                        \
      else                                                           \
        CHECK_MIN_MAX(p1, q1, r1, p2, q2, r2)                        \
    }                                                                \
    else if (dp2 < 0.0f)                                             \
    {                                                                \
      if (dq2 < 0.0f)                                                \
        CHECK_MIN_MAX(p1, q1, r1, r2, p2, q2)                        \
      else if (dr2 < 0.0f)                                           \
        CHECK_MIN_MAX(p1, q1, r1, q2, r2, p2)                        \
      else                                                           \
        CHECK_MIN_MAX(p1, r1, q1, p2, q2, r2)                        \
    }                                                                \
    else                                                             \
    {                                                                \
      if (dq2 < 0.0f)                                                \
      {                                                              \
        if (dr2 >= 0.0f)                                             \
          CHECK_MIN_MAX(p1, r1, q1, q2, r2, p2)                      \
        else                                                         \
          CHECK_MIN_MAX(p1, q1, r1, p2, q2, r2)                      \
      }                                                              \
      else if (dq2 > 0.0f)                                           \
      {                                                              \
        if (dr2 > 0.0f)                                              \
          CHECK_MIN_MAX(p1, r1, q1, p2, q2, r2)                      \
        else                                                         \
          CHECK_MIN_MAX(p1, q1, r1, q2, r2, p2)                      \
      }                                                              \
      else                                                           \
      {                                                              \
        if (dr2 > 0.0f)                                              \
          CHECK_MIN_MAX(p1, q1, r1, r2, p2, q2)                      \
        else if (dr2 < 0.0f)                                         \
          CHECK_MIN_MAX(p1, r1, q1, r2, p2, q2)                      \
        else                                                         \
          return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2); \
      }                                                              \
    }                                                                \
  }

  /*
   *
   *  三维三角形-三角形重叠测试函数
   *
   *  该函数实现了一个高效的三维三角形相交检测算法
   *  基于分离轴定理（Separating Axis Theorem）
   *  首先测试两个三角形所在平面是否分离，然后测试三角形之间的分离轴
   *
   */

  int tri_tri_overlap_test_3d(PQP_REAL p1[3], PQP_REAL q1[3], PQP_REAL r1[3],

                              PQP_REAL p2[3], PQP_REAL q2[3], PQP_REAL r2[3])
  {
    PQP_REAL dp1, dq1, dr1, dp2, dq2, dr2;
    PQP_REAL v1[3], v2[3];
    PQP_REAL N1[3], N2[3];

    /* 计算 p1, q1 和 r1 到三角形(p2,q2,r2)所在平面的距离符号
     * dp1, dq1, dr1 分别为各点到平面的有符号距离
     * 如果三个点都在平面的同一侧，则三角形不重叠
     */

    SUB(v1, p2, r2)
    SUB(v2, q2, r2)
    CROSS(N2, v1, v2)

    SUB(v1, p1, r2)
    dp1 = DOT(v1, N2);
    SUB(v1, q1, r2)
    dq1 = DOT(v1, N2);
    SUB(v1, r1, r2)
    dr1 = DOT(v1, N2);

    // 如果三个顶点都在平面的同一侧（距离同号），则不相交
    if (((dp1 * dq1) > 0.0f) && ((dp1 * dr1) > 0.0f))
      return 0;

    /* 计算 p2, q2 和 r2 到三角形(p1,q1,r1)所在平面的距离符号
     * 同样的测试应用于第二个三角形
     */

    SUB(v1, q1, p1)
    SUB(v2, r1, p1)
    CROSS(N1, v1, v2)

    SUB(v1, p2, r1)
    dp2 = DOT(v1, N1);
    SUB(v1, q2, r1)
    dq2 = DOT(v1, N1);
    SUB(v1, r2, r1)
    dr2 = DOT(v1, N1);

    // 如果三个顶点都在平面的同一侧，则不相交
    if (((dp2 * dq2) > 0.0f) && ((dp2 * dr2) > 0.0f))
      return 0;

    /* T1 顶点的规范排列
     * 根据 dp1, dq1, dr1 的符号对顶点进行排序
     * 以便后续使用 TRI_TRI_3D 宏进行一致的测试
     */

    // 根据第一个三角形顶点到第二个三角形平面的距离符号，进行顶点排列
    if (dp1 > 0.0f)
    {
      if (dq1 > 0.0f)
        TRI_TRI_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
      else if (dr1 > 0.0f)
        TRI_TRI_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
      else
        TRI_TRI_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
    }
    else if (dp1 < 0.0f)
    {
      if (dq1 < 0.0f)
        TRI_TRI_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
      else if (dr1 < 0.0f)
        TRI_TRI_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
      else
        TRI_TRI_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
    }
    else
    {
      if (dq1 < 0.0f)
      {
        if (dr1 >= 0.0f)
          TRI_TRI_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
        else
          TRI_TRI_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
      }
      else if (dq1 > 0.0f)
      {
        if (dr1 > 0.0f)
          TRI_TRI_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
        else
          TRI_TRI_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
      }
      else
      {
        if (dr1 > 0.0f)
          TRI_TRI_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
        else if (dr1 < 0.0f)
          TRI_TRI_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
        // 三个顶点都在平面上，需要进行共面测试
        else
          return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2);
      }
    }
  };

  // 共面三角形-三角形三维重叠测试
  // 当两个三角形共面时，使用二维重叠测试
  // 参数：两个三角形的顶点坐标及其法向量
  int coplanar_tri_tri3d(const PQP_REAL p1[3], const PQP_REAL q1[3], const PQP_REAL r1[3],
                         const PQP_REAL p2[3], const PQP_REAL q2[3], const PQP_REAL r2[3],
                         const PQP_REAL normal_1[3], const PQP_REAL normal_2[3])
  {

    PQP_REAL P1[2], Q1[2], R1[2];
    PQP_REAL P2[2], Q2[2], R2[2];

    PQP_REAL n_x, n_y, n_z;

    // 获取法向量各分量绝对值，用于确定投影平面
    n_x = ((normal_1[0] < 0) ? -normal_1[0] : normal_1[0]);
    n_y = ((normal_1[1] < 0) ? -normal_1[1] : normal_1[1]);
    n_z = ((normal_1[2] < 0) ? -normal_1[2] : normal_1[2]);

    /* 将三维三角形投影到二维平面，使投影面积最大
     * 根据法向量确定最佳投影平面：
     * - 如果 n_x 最大，投影到 YZ 平面
     * - 否则，如果 n_y 最大，投影到 XZ 平面
     * - 否则，投影到 XY 平面
     */

    // 投影到 YZ 平面（使用 x 分量作为判断依据）
    if ((n_x > n_z) && (n_x >= n_y))
    {

      P1[0] = q1[2];
      P1[1] = q1[1];
      Q1[0] = p1[2];
      Q1[1] = p1[1];
      R1[0] = r1[2];
      R1[1] = r1[1];

      P2[0] = q2[2];
      P2[1] = q2[1];
      Q2[0] = p2[2];
      Q2[1] = p2[1];
      R2[0] = r2[2];
      R2[1] = r2[1];
    }
    // 投影到 XZ 平面（使用 y 分量作为判断依据）
    else if ((n_y > n_z) && (n_y >= n_x))
    {

      P1[0] = q1[0];
      P1[1] = q1[2];
      Q1[0] = p1[0];
      Q1[1] = p1[2];
      R1[0] = r1[0];
      R1[1] = r1[2];

      P2[0] = q2[0];
      P2[1] = q2[2];
      Q2[0] = p2[0];
      Q2[1] = p2[2];
      R2[0] = r2[0];
      R2[1] = r2[2];
    }
    // 投影到 XY 平面（z 分量最大）
    else
    {

      P1[0] = p1[0];
      P1[1] = p1[1];
      Q1[0] = q1[0];
      Q1[1] = q1[1];
      R1[0] = r1[0];
      R1[1] = r1[1];

      P2[0] = p2[0];
      P2[1] = p2[1];
      Q2[0] = q2[0];
      Q2[1] = q2[1];
      R2[0] = r2[0];
      R2[1] = r2[1];
    }

    // 调用二维三角形重叠测试
    return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);
  };

  /*
   *
   *  三维三角形-三角形求交函数
   *
   *  该函数不仅测试三角形是否相交，还计算相交线段（如果存在）
   */

  /*
     当三角形肯定相交时调用此宏
     如果两个三角形不共面，它构建两个三角形的相交线段
     source 和 target 是相交线段的两个端点
  */

#define CONSTRUCT_INTERSECTION(p1, q1, r1, p2, q2, r2) \
  {                                                    \
    SUB(v1, q1, p1)                                    \
    SUB(v2, r2, p1)                                    \
    CROSS(N, v1, v2)                                   \
    SUB(v, p2, p1)                                     \
    if (DOT(v, N) > 0.0f)                              \
    {                                                  \
      SUB(v1, r1, p1)                                  \
      CROSS(N, v1, v2)                                 \
      if (DOT(v, N) <= 0.0f)                           \
      {                                                \
        SUB(v2, q2, p1)                                \
        CROSS(N, v1, v2)                               \
        if (DOT(v, N) > 0.0f)                          \
        {                                              \
          SUB(v1, p1, p2)                              \
          SUB(v2, p1, r1)                              \
          alpha = DOT(v1, N2) / DOT(v2, N2);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(source, p1, v1)                          \
          SUB(v1, p2, p1)                              \
          SUB(v2, p2, r2)                              \
          alpha = DOT(v1, N1) / DOT(v2, N1);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(target, p2, v1)                          \
          return 1;                                    \
        }                                              \
        else                                           \
        {                                              \
          SUB(v1, p2, p1)                              \
          SUB(v2, p2, q2)                              \
          alpha = DOT(v1, N1) / DOT(v2, N1);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(source, p2, v1)                          \
          SUB(v1, p2, p1)                              \
          SUB(v2, p2, r2)                              \
          alpha = DOT(v1, N1) / DOT(v2, N1);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(target, p2, v1)                          \
          return 1;                                    \
        }                                              \
      }                                                \
      else                                             \
      {                                                \
        return 0;                                      \
      }                                                \
    }                                                  \
    else                                               \
    {                                                  \
      SUB(v2, q2, p1)                                  \
      CROSS(N, v1, v2)                                 \
      if (DOT(v, N) < 0.0f)                            \
      {                                                \
        return 0;                                      \
      }                                                \
      else                                             \
      {                                                \
        SUB(v1, r1, p1)                                \
        CROSS(N, v1, v2)                               \
        if (DOT(v, N) >= 0.0f)                         \
        {                                              \
          SUB(v1, p1, p2)                              \
          SUB(v2, p1, r1)                              \
          alpha = DOT(v1, N2) / DOT(v2, N2);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(source, p1, v1)                          \
          SUB(v1, p1, p2)                              \
          SUB(v2, p1, q1)                              \
          alpha = DOT(v1, N2) / DOT(v2, N2);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(target, p1, v1)                          \
          return 1;                                    \
        }                                              \
        else                                           \
        {                                              \
          SUB(v1, p2, p1)                              \
          SUB(v2, p2, q2)                              \
          alpha = DOT(v1, N1) / DOT(v2, N1);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(source, p2, v1)                          \
          SUB(v1, p1, p2)                              \
          SUB(v2, p1, q1)                              \
          alpha = DOT(v1, N2) / DOT(v2, N2);           \
          SCALAR(v1, alpha, v2)                        \
          SUB(target, p1, v1)                          \
          return 1;                                    \
        }                                              \
      }                                                \
    }                                                  \
  }

  // TRI_TRI_INTER_3D: 三维三角形求交宏，根据顶点位置关系调用 CONSTRUCT_INTERSECTION
#define TRI_TRI_INTER_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)      \
  {                                                                  \
    if (dp2 > 0.0f)                                                  \
    {                                                                \
      if (dq2 > 0.0f)                                                \
        CONSTRUCT_INTERSECTION(p1, r1, q1, r2, p2, q2)               \
      else if (dr2 > 0.0f)                                           \
        CONSTRUCT_INTERSECTION(p1, r1, q1, q2, r2, p2)               \
      else                                                           \
        CONSTRUCT_INTERSECTION(p1, q1, r1, p2, q2, r2)               \
    }                                                                \
    else if (dp2 < 0.0f)                                             \
    {                                                                \
      if (dq2 < 0.0f)                                                \
        CONSTRUCT_INTERSECTION(p1, q1, r1, r2, p2, q2)               \
      else if (dr2 < 0.0f)                                           \
        CONSTRUCT_INTERSECTION(p1, q1, r1, q2, r2, p2)               \
      else                                                           \
        CONSTRUCT_INTERSECTION(p1, r1, q1, p2, q2, r2)               \
    }                                                                \
    else                                                             \
    {                                                                \
      if (dq2 < 0.0f)                                                \
      {                                                              \
        if (dr2 >= 0.0f)                                             \
          CONSTRUCT_INTERSECTION(p1, r1, q1, q2, r2, p2)             \
        else                                                         \
          CONSTRUCT_INTERSECTION(p1, q1, r1, p2, q2, r2)             \
      }                                                              \
      else if (dq2 > 0.0f)                                           \
      {                                                              \
        if (dr2 > 0.0f)                                              \
          CONSTRUCT_INTERSECTION(p1, r1, q1, p2, q2, r2)             \
        else                                                         \
          CONSTRUCT_INTERSECTION(p1, q1, r1, q2, r2, p2)             \
      }                                                              \
      else                                                           \
      {                                                              \
        if (dr2 > 0.0f)                                              \
          CONSTRUCT_INTERSECTION(p1, q1, r1, r2, p2, q2)             \
        else if (dr2 < 0.0f)                                         \
          CONSTRUCT_INTERSECTION(p1, r1, q1, r2, p2, q2)             \
        else                                                         \
        {                                                            \
          return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2); \
        }                                                            \
      }                                                              \
    }                                                                \
  }

  /*
     以下版本计算两个三角形相交的线段（如果存在）
     coplanar 返回三角形是否共面
     source 和 target 是相交线段的端点
  */

  // TriInt: 主求交函数，检测两个三角形是否相交并返回布尔值
  // 参数 data1 和 data2 是包含两个三角形顶点坐标的数组
  // 格式：[p1x,p1y,p1z, q1x,q1y,q1z, r1x,r1y,r1z]（每个三角形9个浮点数）
  bool TriInt(const PQP_REAL *data1, const PQP_REAL *data2)
  {
    const PQP_REAL *p1 = data1;
    const PQP_REAL *q1 = data1 + 3;
    const PQP_REAL *r1 = data1 + 6;
    const PQP_REAL *p2 = data2;
    const PQP_REAL *q2 = data2 + 3;
    const PQP_REAL *r2 = data2 + 6;

    PQP_REAL dp1, dq1, dr1, dp2, dq2, dr2;
    PQP_REAL v1[3], v2[3], v[3];
    PQP_REAL N1[3], N2[3], N[3];
    PQP_REAL alpha;
    PQP_REAL source[3];
    PQP_REAL target[3];

    // 计算 p1, q1, r1 到三角形(p2,q2,r2)所在平面的有符号距离

    SUB(v1, p2, r2)
    SUB(v2, q2, r2)
    CROSS(N2, v1, v2)

    SUB(v1, p1, r2)
    dp1 = DOT(v1, N2);
    SUB(v1, q1, r2)
    dq1 = DOT(v1, N2);
    SUB(v1, r1, r2)
    dr1 = DOT(v1, N2);

    // 如果第一个三角形的所有顶点都在第二个三角形平面的同一侧，则不相交
    if (((dp1 * dq1) > 0.0f) && ((dp1 * dr1) > 0.0f))
      return 0;

    // 计算 p2, q2, r2 到三角形(p1,q1,r1)所在平面的有符号距离

    SUB(v1, q1, p1)
    SUB(v2, r1, p1)
    CROSS(N1, v1, v2)

    SUB(v1, p2, r1)
    dp2 = DOT(v1, N1);
    SUB(v1, q2, r1)
    dq2 = DOT(v1, N1);
    SUB(v1, r2, r1)
    dr2 = DOT(v1, N1);

    // 如果第二个三角形的所有顶点都在第一个三角形平面的同一侧，则不相交
    if (((dp2 * dq2) > 0.0f) && ((dp2 * dr2) > 0.0f))
      return 0;

    // T1 顶点的规范排列

    //  printf("d1 = [%f %f %f], d2 = [%f %f %f]\n", dp1, dq1, dr1, dp2, dq2, dr2);
    /*
    // 由 Aaron 添加
    // 用于处理共面情况的测试（已注释）
    if (ZERO_TEST(dp1) || ZERO_TEST(dq1) ||ZERO_TEST(dr1) ||ZERO_TEST(dp2) ||ZERO_TEST(dq2) ||ZERO_TEST(dr2))
      {
        coplanar = 1;
        return 0;
      }
    */

    // 根据距离符号进行顶点排列，然后进行求交测试
    if (dp1 > 0.0f)
    {
      if (dq1 > 0.0f)
        TRI_TRI_INTER_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
      else if (dr1 > 0.0f)
        TRI_TRI_INTER_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)

      else
        TRI_TRI_INTER_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
    }
    else if (dp1 < 0.0f)
    {
      if (dq1 < 0.0f)
        TRI_TRI_INTER_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
      else if (dr1 < 0.0f)
        TRI_TRI_INTER_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
      else
        TRI_TRI_INTER_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
    }
    else
    {
      if (dq1 < 0.0f)
      {
        if (dr1 >= 0.0f)
          TRI_TRI_INTER_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
        else
          TRI_TRI_INTER_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
      }
      else if (dq1 > 0.0f)
      {
        if (dr1 > 0.0f)
          TRI_TRI_INTER_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
        else
          TRI_TRI_INTER_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
      }
      else
      {
        if (dr1 > 0.0f)
          TRI_TRI_INTER_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
        else if (dr1 < 0.0f)
          TRI_TRI_INTER_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
        else
        {
          // 三角形共面，进行共面测试
          return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2);
        }
      }
    }
  };

/*
 *
 *  二维三角形-三角形重叠测试函数
 *
 *  用于共面情况下的二维投影测试
 */

/* 二维向量操作宏定义 */

// ORIENT_2D: 计算三个二维点的定向面积（叉积）
// 返回值 > 0 表示逆时针，< 0 表示顺时针，= 0 表示共线
#define ORIENT_2D(a, b, c) ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))

// INTERSECTION_TEST_VERTEX: 测试三角形的一个顶点是否在另一个三角形内部
// 使用一系列有向面积测试
#define INTERSECTION_TEST_VERTEX(P1, Q1, R1, P2, Q2, R2) \
  {                                                      \
    if (ORIENT_2D(R2, P2, Q1) >= 0.0f)                   \
      if (ORIENT_2D(R2, Q2, Q1) <= 0.0f)                 \
        if (ORIENT_2D(P1, P2, Q1) > 0.0f)                \
        {                                                \
          if (ORIENT_2D(P1, Q2, Q1) <= 0.0f)             \
            return 1;                                    \
          else                                           \
            return 0;                                    \
        }                                                \
        else                                             \
        {                                                \
          if (ORIENT_2D(P1, P2, R1) >= 0.0f)             \
            if (ORIENT_2D(Q1, R1, P2) >= 0.0f)           \
              return 1;                                  \
            else                                         \
              return 0;                                  \
          else                                           \
            return 0;                                    \
        }                                                \
      else if (ORIENT_2D(P1, Q2, Q1) <= 0.0f)            \
        if (ORIENT_2D(R2, Q2, R1) <= 0.0f)               \
          if (ORIENT_2D(Q1, R1, Q2) >= 0.0f)             \
            return 1;                                    \
          else                                           \
            return 0;                                    \
        else                                             \
          return 0;                                      \
      else                                               \
        return 0;                                        \
    else if (ORIENT_2D(R2, P2, R1) >= 0.0f)              \
      if (ORIENT_2D(Q1, R1, R2) >= 0.0f)                 \
        if (ORIENT_2D(P1, P2, R1) >= 0.0f)               \
          return 1;                                      \
        else                                             \
          return 0;                                      \
      else if (ORIENT_2D(Q1, R1, Q2) >= 0.0f)            \
      {                                                  \
        if (ORIENT_2D(R2, R1, Q2) >= 0.0f)               \
          return 1;                                      \
        else                                             \
          return 0;                                      \
      }                                                  \
      else                                               \
        return 0;                                        \
    else                                                 \
      return 0;                                          \
  };

// INTERSECTION_TEST_EDGE: 测试两个三角形的边是否相交
#define INTERSECTION_TEST_EDGE(P1, Q1, R1, P2, Q2, R2) \
  {                                                    \
    if (ORIENT_2D(R2, P2, Q1) >= 0.0f)                 \
    {                                                  \
      if (ORIENT_2D(P1, P2, Q1) >= 0.0f)               \
      {                                                \
        if (ORIENT_2D(P1, Q1, R2) >= 0.0f)             \
          return 1;                                    \
        else                                           \
          return 0;                                    \
      }                                                \
      else                                             \
      {                                                \
        if (ORIENT_2D(Q1, R1, P2) >= 0.0f)             \
        {                                              \
          if (ORIENT_2D(R1, P1, P2) >= 0.0f)           \
            return 1;                                  \
          else                                         \
            return 0;                                  \
        }                                              \
        else                                           \
          return 0;                                    \
      }                                                \
    }                                                  \
    else                                               \
    {                                                  \
      if (ORIENT_2D(R2, P2, R1) >= 0.0f)               \
      {                                                \
        if (ORIENT_2D(P1, P2, R1) >= 0.0f)             \
        {                                              \
          if (ORIENT_2D(P1, R1, R2) >= 0.0f)           \
            return 1;                                  \
          else                                         \
          {                                            \
            if (ORIENT_2D(Q1, R1, R2) >= 0.0f)         \
              return 1;                                \
            else                                       \
              return 0;                                \
          }                                            \
        }                                              \
        else                                           \
          return 0;                                    \
      }                                                \
      else                                             \
        return 0;                                      \
    }                                                  \
  }

  // ccw_tri_tri_intersection_2d: 逆时针二维三角形相交测试
  // 根据顶点的相对位置关系进行一系列测试
  int ccw_tri_tri_intersection_2d(PQP_REAL p1[2], PQP_REAL q1[2], PQP_REAL r1[2],
                                  PQP_REAL p2[2], PQP_REAL q2[2], PQP_REAL r2[2])
  {
    if (ORIENT_2D(p2, q2, p1) >= 0.0f)
    {
      if (ORIENT_2D(q2, r2, p1) >= 0.0f)
      {
        if (ORIENT_2D(r2, p2, p1) >= 0.0f)
          return 1;
        else
          INTERSECTION_TEST_EDGE(p1, q1, r1, p2, q2, r2)
      }
      else
      {
        if (ORIENT_2D(r2, p2, p1) >= 0.0f)
          INTERSECTION_TEST_EDGE(p1, q1, r1, r2, p2, q2)
        else
          INTERSECTION_TEST_VERTEX(p1, q1, r1, p2, q2, r2)
      }
    }
    else
    {
      if (ORIENT_2D(q2, r2, p1) >= 0.0f)
      {
        if (ORIENT_2D(r2, p2, p1) >= 0.0f)
          INTERSECTION_TEST_EDGE(p1, q1, r1, q2, r2, p2)
        else
          INTERSECTION_TEST_VERTEX(p1, q1, r1, q2, r2, p2)
      }
      else
        INTERSECTION_TEST_VERTEX(p1, q1, r1, r2, p2, q2)
    }
  };

  // tri_tri_overlap_test_2d: 二维三角形重叠测试主函数
  // 首先规范化三角形的顶点顺序（确保逆时针）
  // 然后调用 ccw_tri_tri_intersection_2d 进行测试
  int tri_tri_overlap_test_2d(PQP_REAL p1[2], PQP_REAL q1[2], PQP_REAL r1[2],
                              PQP_REAL p2[2], PQP_REAL q2[2], PQP_REAL r2[2])
  {
    // 第一个三角形逆时针
    if (ORIENT_2D(p1, q1, r1) < 0.0f)
      // 第二个三角形逆时针
      if (ORIENT_2D(p2, q2, r2) < 0.0f)
        return ccw_tri_tri_intersection_2d(p1, r1, q1, p2, r2, q2);
      else
        return ccw_tri_tri_intersection_2d(p1, r1, q1, p2, q2, r2);
    else
      // 第二个三角形逆时针
      if (ORIENT_2D(p2, q2, r2) < 0.0f)
        return ccw_tri_tri_intersection_2d(p1, q1, r1, p2, r2, q2);
      else
        return ccw_tri_tri_intersection_2d(p1, q1, r1, p2, q2, r2);
  }

} // namespace tdbase
