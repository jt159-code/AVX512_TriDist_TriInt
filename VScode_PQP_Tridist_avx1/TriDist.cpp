//--------------------------------------------------------------------------
// 文件: TriDist.cpp (AVX512向量化准备版本)
// 作者: Eric Larsen (修改: 为AVX512向量化做准备)
// 描述:
// 包含 SegPoints() 函数，用于计算两条线段之间的最近点对；
// 以及 TriDist() 函数，用于计算两个三角形之间的最近点对。
// 修改目标: 将SegPoints重构为数据驱动的形式，便于后续AVX512向量化
//--------------------------------------------------------------------------

#include "MatVec.h"
#include <vector>
#include <cstring>
#include <cmath>

#ifdef _WIN32
#include <float.h>
#include <math.h>
using namespace std;
#endif

//--------------------------------------------------------------------------
// 枚举类型: 定义SegPoints的所有可能分支类型
//--------------------------------------------------------------------------
enum SegPointBranchType
{
    BRANCH_T_LT_0,        // t < 0
    BRANCH_T_GT_1,        // t > 1
    BRANCH_U_LT_0,        // u <= 0
    BRANCH_U_GT_1,        // u >= 1
    BRANCH_VALID_T_U,     // 0 <= t <= 1 && 0 <= u <= 1
    BRANCH_T_LT_0_U_LT_0, // t < 0 && u <= 0
    BRANCH_T_LT_0_U_GT_1, // t < 0 && u >= 1
    BRANCH_T_GT_1_U_LT_0, // t > 1 && u <= 0
    BRANCH_T_GT_1_U_GT_1, // t > 1 && u >= 1
    BRANCH_COUNT          // 分支总数
};

//--------------------------------------------------------------------------
// 结构体: 存储单个线段对的所有数据和中间结果
//--------------------------------------------------------------------------
struct SegPointsData
{
    // 输入数据
    PQP_REAL P[3]; // 线段1的原点
    PQP_REAL A[3]; // 线段1的方向向量
    PQP_REAL Q[3]; // 线段2的原点
    PQP_REAL B[3]; // 线段2的方向向量

    // 中间计算结果
    PQP_REAL T[3];    // T = Q - P
    PQP_REAL A_dot_A; // A·A
    PQP_REAL B_dot_B; // B·B
    PQP_REAL A_dot_B; // A·B
    PQP_REAL A_dot_T; // A·T
    PQP_REAL B_dot_T; // B·T
    PQP_REAL denom;   // 分母 = A_dot_A * B_dot_B - A_dot_B * A_dot_B

    // 参数t和u
    PQP_REAL t;
    PQP_REAL u;

    // 分支类型
    SegPointBranchType branch;

    // 输出数据
    PQP_REAL X[3];   // 线段1上的最近点
    PQP_REAL Y[3];   // 线段2上的最近点
    PQP_REAL VEC[3]; // 从X指向Y的向量

    // 构造函数：初始化所有数据为0
    SegPointsData()
    {
        memset(P, 0, sizeof(P));
        memset(A, 0, sizeof(A));
        memset(Q, 0, sizeof(Q));
        memset(B, 0, sizeof(B));
        memset(T, 0, sizeof(T));
        memset(X, 0, sizeof(X));
        memset(Y, 0, sizeof(Y));
        memset(VEC, 0, sizeof(VEC));
        A_dot_A = B_dot_B = A_dot_B = A_dot_T = B_dot_T = denom = 0;
        t = u = 0;
        branch = BRANCH_VALID_T_U;
    }
};

//--------------------------------------------------------------------------
// 函数: ComputeBranchType
// 描述: 根据t和u的值确定分支类型
//--------------------------------------------------------------------------
inline SegPointBranchType ComputeBranchType(PQP_REAL t, PQP_REAL u)
{
    bool t_lt_0 = (t < 0) || isnan(t);
    bool t_gt_1 = (t > 1) || isnan(t);
    bool u_lt_0 = (u <= 0) || isnan(u);
    bool u_gt_1 = (u >= 1) || isnan(u);

    if (t_lt_0)
    {
        if (u_lt_0)
            return BRANCH_T_LT_0_U_LT_0;
        if (u_gt_1)
            return BRANCH_T_LT_0_U_GT_1;
        return BRANCH_T_LT_0;
    }

    if (t_gt_1)
    {
        if (u_lt_0)
            return BRANCH_T_GT_1_U_LT_0;
        if (u_gt_1)
            return BRANCH_T_GT_1_U_GT_1;
        return BRANCH_T_GT_1;
    }

    if (u_lt_0)
        return BRANCH_U_LT_0;
    if (u_gt_1)
        return BRANCH_U_GT_1;

    return BRANCH_VALID_T_U;
}

//--------------------------------------------------------------------------
// 函数: PrepareSegPointsData
// 描述: 为单个线段对准备所有中间数据和分支类型
//--------------------------------------------------------------------------
inline void PrepareSegPointsData(SegPointsData &data)
{
    // 计算T = Q - P
    VmV(data.T, data.Q, data.P);

    // 计算点积
    data.A_dot_A = VdotV(data.A, data.A);
    data.B_dot_B = VdotV(data.B, data.B);
    data.A_dot_B = VdotV(data.A, data.B);
    data.A_dot_T = VdotV(data.A, data.T);
    data.B_dot_T = VdotV(data.B, data.T);

    // 计算分母
    data.denom = data.A_dot_A * data.B_dot_B - data.A_dot_B * data.A_dot_B;

    // 计算t
    data.t = (data.A_dot_T * data.B_dot_B - data.B_dot_T * data.A_dot_B) / data.denom;

    // 计算u
    data.u = (data.t * data.A_dot_B - data.B_dot_T) / data.B_dot_B;

    // 确定分支类型
    data.branch = ComputeBranchType(data.t, data.u);
}

//--------------------------------------------------------------------------
// 分支处理函数组: 每个分支对应一个计算内核
//--------------------------------------------------------------------------

// 分支: BRANCH_T_LT_0 (t < 0)
inline void ProcessBranch_T_LT_0(SegPointsData &data)
{
    // Y = Q
    VcV(data.Y, data.Q);

    // t = A_dot_T / A_dot_A
    data.t = data.A_dot_T / data.A_dot_A;

    if ((data.t <= 0) || isnan(data.t))
    {
        // X = P
        VcV(data.X, data.P);
        // VEC = Q - P
        VmV(data.VEC, data.Q, data.P);
    }
    else if (data.t >= 1)
    {
        // X = P + A
        VpV(data.X, data.P, data.A);
        // VEC = Q - X
        VmV(data.VEC, data.Q, data.X);
    }
    else
    {
        // X = P + t*A
        VpVxS(data.X, data.P, data.A, data.t);
        // TMP = T × A
        PQP_REAL TMP[3];
        VcrossV(TMP, data.T, data.A);
        // VEC = A × (T × A)
        VcrossV(data.VEC, data.A, TMP);
    }
}

// 分支: BRANCH_T_GT_1 (t > 1)
inline void ProcessBranch_T_GT_1(SegPointsData &data)
{
    // Y = Q + B
    VpV(data.Y, data.Q, data.B);

    // t = (A_dot_B + A_dot_T) / A_dot_A
    data.t = (data.A_dot_B + data.A_dot_T) / data.A_dot_A;

    if ((data.t <= 0) || isnan(data.t))
    {
        // X = P
        VcV(data.X, data.P);
        // VEC = Y - P
        VmV(data.VEC, data.Y, data.P);
    }
    else if (data.t >= 1)
    {
        // X = P + A
        VpV(data.X, data.P, data.A);
        // VEC = Y - X
        VmV(data.VEC, data.Y, data.X);
    }
    else
    {
        // X = P + t*A
        VpVxS(data.X, data.P, data.A, data.t);
        // T = Y - P
        PQP_REAL T[3];
        VmV(T, data.Y, data.P);
        // TMP = (Y-P) × A
        PQP_REAL TMP[3];
        VcrossV(TMP, T, data.A);
        // VEC = A × ((Y-P)×A)
        VcrossV(data.VEC, data.A, TMP);
    }
}

// 分支: BRANCH_U_LT_0 (u <= 0)
inline void ProcessBranch_U_LT_0(SegPointsData &data)
{
    // Y = Q
    VcV(data.Y, data.Q);

    // t = A_dot_T / A_dot_A
    data.t = data.A_dot_T / data.A_dot_A;

    if ((data.t <= 0) || isnan(data.t))
    {
        // X = P
        VcV(data.X, data.P);
        // VEC = Q - P
        VmV(data.VEC, data.Q, data.P);
    }
    else if (data.t >= 1)
    {
        // X = P + A
        VpV(data.X, data.P, data.A);
        // VEC = Q - X
        VmV(data.VEC, data.Q, data.X);
    }
    else
    {
        // X = P + t*A
        VpVxS(data.X, data.P, data.A, data.t);
        // TMP = T × A
        PQP_REAL TMP[3];
        VcrossV(TMP, data.T, data.A);
        // VEC = A × (T × A)
        VcrossV(data.VEC, data.A, TMP);
    }
}

// 分支: BRANCH_U_GT_1 (u >= 1)
inline void ProcessBranch_U_GT_1(SegPointsData &data)
{
    // Y = Q + B
    VpV(data.Y, data.Q, data.B);

    // t = (A_dot_B + A_dot_T) / A_dot_A
    data.t = (data.A_dot_B + data.A_dot_T) / data.A_dot_A;

    if ((data.t <= 0) || isnan(data.t))
    {
        // X = P
        VcV(data.X, data.P);
        // VEC = Y - P
        VmV(data.VEC, data.Y, data.P);
    }
    else if (data.t >= 1)
    {
        // X = P + A
        VpV(data.X, data.P, data.A);
        // VEC = Y - X
        VmV(data.VEC, data.Y, data.X);
    }
    else
    {
        // X = P + t*A
        VpVxS(data.X, data.P, data.A, data.t);
        // T = Y - P
        PQP_REAL T[3];
        VmV(T, data.Y, data.P);
        // TMP = (Y-P) × A
        PQP_REAL TMP[3];
        VcrossV(TMP, T, data.A);
        // VEC = A × ((Y-P)×A)
        VcrossV(data.VEC, data.A, TMP);
    }
}

// 分支: BRANCH_VALID_T_U (0 <= t <= 1 && 0 <= u <= 1)
inline void ProcessBranch_VALID_T_U(SegPointsData &data)
{
    // Y = Q + u*B
    VpVxS(data.Y, data.Q, data.B, data.u);

    if ((data.t <= 0) || isnan(data.t))
    {
        // X = P
        VcV(data.X, data.P);
        // TMP = (Q-P) × B
        PQP_REAL TMP[3];
        VcrossV(TMP, data.T, data.B);
        // VEC = B × ((Q-P)×B)
        VcrossV(data.VEC, data.B, TMP);
    }
    else if (data.t >= 1)
    {
        // X = P + A
        VpV(data.X, data.P, data.A);
        // T = Q - X
        PQP_REAL T[3];
        VmV(T, data.Q, data.X);
        // TMP = (Q-X) × B
        PQP_REAL TMP[3];
        VcrossV(TMP, T, data.B);
        // VEC = B × ((Q-X)×B)
        VcrossV(data.VEC, data.B, TMP);
    }
    else
    {
        // X = P + t*A
        VpVxS(data.X, data.P, data.A, data.t);
        // VEC = A × B
        VcrossV(data.VEC, data.A, data.B);
        // 如果VEC与T方向相反，则反向
        if (VdotV(data.VEC, data.T) < 0)
        {
            VxS(data.VEC, data.VEC, -1);
        }
    }
}

// 分支: BRANCH_T_LT_0_U_LT_0 (t < 0 && u <= 0)
inline void ProcessBranch_T_LT_0_U_LT_0(SegPointsData &data)
{
    // X = P
    VcV(data.X, data.P);
    // Y = Q
    VcV(data.Y, data.Q);
    // VEC = Q - P
    VmV(data.VEC, data.Q, data.P);
}

// 分支: BRANCH_T_LT_0_U_GT_1 (t < 0 && u >= 1)
inline void ProcessBranch_T_LT_0_U_GT_1(SegPointsData &data)
{
    // X = P
    VcV(data.X, data.P);
    // Y = Q + B
    VpV(data.Y, data.Q, data.B);
    // VEC = Y - P
    VmV(data.VEC, data.Y, data.P);
}

// 分支: BRANCH_T_GT_1_U_LT_0 (t > 1 && u <= 0)
inline void ProcessBranch_T_GT_1_U_LT_0(SegPointsData &data)
{
    // X = P + A
    VpV(data.X, data.P, data.A);
    // Y = Q
    VcV(data.Y, data.Q);
    // VEC = Y - X
    VmV(data.VEC, data.Y, data.X);
}

// 分支: BRANCH_T_GT_1_U_GT_1 (t > 1 && u >= 1)
inline void ProcessBranch_T_GT_1_U_GT_1(SegPointsData &data)
{
    // X = P + A
    VpV(data.X, data.P, data.A);
    // Y = Q + B
    VpV(data.Y, data.Q, data.B);
    // VEC = Y - X
    VmV(data.VEC, data.Y, data.X);
}

//--------------------------------------------------------------------------
// 函数: ProcessSegPointsData
// 描述: 根据分支类型调用对应的处理函数
//--------------------------------------------------------------------------
inline void ProcessSegPointsData(SegPointsData &data)
{
    switch (data.branch)
    {
    case BRANCH_T_LT_0:
        ProcessBranch_T_LT_0(data);
        break;
    case BRANCH_T_GT_1:
        ProcessBranch_T_GT_1(data);
        break;
    case BRANCH_U_LT_0:
        ProcessBranch_U_LT_0(data);
        break;
    case BRANCH_U_GT_1:
        ProcessBranch_U_GT_1(data);
        break;
    case BRANCH_VALID_T_U:
        ProcessBranch_VALID_T_U(data);
        break;
    case BRANCH_T_LT_0_U_LT_0:
        ProcessBranch_T_LT_0_U_LT_0(data);
        break;
    case BRANCH_T_LT_0_U_GT_1:
        ProcessBranch_T_LT_0_U_GT_1(data);
        break;
    case BRANCH_T_GT_1_U_LT_0:
        ProcessBranch_T_GT_1_U_LT_0(data);
        break;
    case BRANCH_T_GT_1_U_GT_1:
        ProcessBranch_T_GT_1_U_GT_1(data);
        break;
    default:
        // 不应该到达这里
        break;
    }
}

//--------------------------------------------------------------------------
// 函数: SegPointsVectorized
// 描述: 批量处理多个线段对的最近点计算，为向量化做准备
// 输入:
//   numPairs - 线段对的数量
//   P, A - 线段1的原点和方向向量数组
//   Q, B - 线段2的原点和方向向量数组
// 输出:
//   VEC, X, Y - 结果向量和最近点数组
//--------------------------------------------------------------------------
void SegPointsVectorized(
    PQP_REAL VEC[][3],
    PQP_REAL X[][3],
    PQP_REAL Y[][3],
    const PQP_REAL P[][3],
    const PQP_REAL A[][3],
    const PQP_REAL Q[][3],
    const PQP_REAL B[][3],
    int numPairs)
{
    if (numPairs <= 0)
        return;

    // 步骤1: 为所有线段对准备数据
    std::vector<SegPointsData> dataArray(numPairs);

    for (int i = 0; i < numPairs; ++i)
    {
        // 复制输入数据
        memcpy(dataArray[i].P, P[i], 3 * sizeof(PQP_REAL));
        memcpy(dataArray[i].A, A[i], 3 * sizeof(PQP_REAL));
        memcpy(dataArray[i].Q, Q[i], 3 * sizeof(PQP_REAL));
        memcpy(dataArray[i].B, B[i], 3 * sizeof(PQP_REAL));

        // 准备中间数据
        PrepareSegPointsData(dataArray[i]);
    }

    // 步骤2: 按分支类型分组处理
    // 这里为了简单起见，逐个处理
    // 在实际向量化实现中，应该按分支类型分组后批量处理
    for (int i = 0; i < numPairs; ++i)
    {
        ProcessSegPointsData(dataArray[i]);
    }

    // 步骤3: 输出结果
    for (int i = 0; i < numPairs; ++i)
    {
        memcpy(VEC[i], dataArray[i].VEC, 3 * sizeof(PQP_REAL));
        memcpy(X[i], dataArray[i].X, 3 * sizeof(PQP_REAL));
        memcpy(Y[i], dataArray[i].Y, 3 * sizeof(PQP_REAL));
    }
}

//--------------------------------------------------------------------------
// 函数: SegPoints (保持原有接口以兼容现有代码)
// 描述: 计算两条线段之间的最近点对 (单个线段对版本)
//--------------------------------------------------------------------------
void SegPoints(
    PQP_REAL VEC[3],
    PQP_REAL X[3],
    PQP_REAL Y[3],
    const PQP_REAL P[3],
    const PQP_REAL A[3],
    const PQP_REAL Q[3],
    const PQP_REAL B[3])
{
    // 准备单个线段对的数据
    SegPointsData data;
    memcpy(data.P, P, 3 * sizeof(PQP_REAL));
    memcpy(data.A, A, 3 * sizeof(PQP_REAL));
    memcpy(data.Q, Q, 3 * sizeof(PQP_REAL));
    memcpy(data.B, B, 3 * sizeof(PQP_REAL));

    // 准备中间数据并处理
    PrepareSegPointsData(data);
    ProcessSegPointsData(data);

    // 输出结果
    memcpy(VEC, data.VEC, 3 * sizeof(PQP_REAL));
    memcpy(X, data.X, 3 * sizeof(PQP_REAL));
    memcpy(Y, data.Y, 3 * sizeof(PQP_REAL));
}

//--------------------------------------------------------------------------
// TriDist()
//
// 计算两个三角形之间的最近点对，并返回它们之间的距离。
//
// S 和 T 是三角形，存储方式为 tri[点索引][坐标分量]。
//
// 如果三角形不相交（分离），则 P 和 Q 分别给出 S 和 T 上的最近点。
// 但如果三角形重叠，P 和 Q 基本上是三角形上任意一对点，而不是
// 交点上的重合点（这一点可能与预期不同）。
//--------------------------------------------------------------------------

PQP_REAL
TriDist(PQP_REAL P[3], PQP_REAL Q[3],
        const PQP_REAL S[3][3], const PQP_REAL T[3][3])
{
    // 计算 6 条边的方向向量

    PQP_REAL Sv[3][3], Tv[3][3];
    PQP_REAL VEC[3];

    VmV(Sv[0], S[1], S[0]); // Sv[0] = S[1] - S[0]
    VmV(Sv[1], S[2], S[1]); // Sv[1] = S[2] - S[1]
    VmV(Sv[2], S[0], S[2]); // Sv[2] = S[0] - S[2]

    VmV(Tv[0], T[1], T[0]); // Tv[0] = T[1] - T[0]
    VmV(Tv[1], T[2], T[1]); // Tv[1] = T[2] - T[1]
    VmV(Tv[2], T[0], T[2]); // Tv[2] = T[0] - T[2]

    // 对于每一对边，连接边对最近点的向量定义了一个“平板”（slab），
    // 由平行于该向量的平面在端点处包围。如果能够证明每个三角形中
    // 不在该边上的顶点都在平板之外，那么该边对的最近点就是三角形的最近点。
    // 即使这些测试失败，记录找到的最近点以及是否已证明三角形分离仍可能有帮助。

    PQP_REAL V[3];
    PQP_REAL Z[3];
    PQP_REAL minP[3], minQ[3], mindd;
    int shown_disjoint = 0;

    mindd = VdistV2(S[0], T[0]) + 1; // 初始最小值设为一个足够大的值（S[0]到T[0]距离平方+1）

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 计算边 i 和边 j 上的最近点，以及它们之间的向量（和距离平方）

            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            VmV(V, Q, P);              // V = Q - P
            PQP_REAL dd = VdotV(V, V); // dd = |V|^2

            // 仅当距离平方小于当前最小值时才验证这对最近点

            if (dd <= mindd)
            {
                VcV(minP, P); // 保存候选点
                VcV(minQ, Q);
                mindd = dd;

                VmV(Z, S[(i + 2) % 3], P);  // Z = S[第三个顶点] - P
                PQP_REAL a = VdotV(Z, VEC); // a = Z · VEC
                VmV(Z, T[(j + 2) % 3], Q);  // Z = T[第三个顶点] - Q
                PQP_REAL b = VdotV(Z, VEC); // b = Z · VEC

                if ((a <= 0) && (b >= 0))
                    return sqrt(dd); // 若满足条件，则该边对最近点即为全局最近点

                PQP_REAL p = VdotV(V, VEC); // p = V · VEC

                if (a < 0)
                    a = 0;
                if (b > 0)
                    b = 0;
                if ((p - a + b) > 0)
                    shown_disjoint = 1; // 如果 p - a + b > 0，可证明三角形分离
            }
        }
    }

    // 没有边对能确定最近点，此时可能的情况：
    // 1. 其中一个最近点是顶点，另一个点在三角形内部。
    // 2. 三角形重叠。
    // 3. 一个三角形的边与另一个三角形的面平行。如果情况1和2不成立，
    //    那么之前9对边检查中找到的最近点可以作为三角形的最近点。
    // 4. 可能是三角形退化（点几乎共线或重合）。此时即使边对包含最近点，
    //    上述某些测试也可能失败。

    // 首先检查情况1

    PQP_REAL Sn[3], Snl;
    VcrossV(Sn, Sv[0], Sv[1]); // 计算三角形 S 的法向量
    Snl = VdotV(Sn, Sn);       // 法向量长度的平方

    // 如果叉积长度足够大（非退化）

    if (Snl > 1e-15)
    {
        // 计算 T 的三个顶点在法向上的投影长度

        PQP_REAL Tp[3];

        VmV(V, S[0], T[0]);   // V = S[0] - T[0]
        Tp[0] = VdotV(V, Sn); // Tp[0] = (S[0] - T[0])·Sn

        VmV(V, S[0], T[1]);   // V = S[0] - T[1]
        Tp[1] = VdotV(V, Sn); // Tp[1] = (S[0] - T[1])·Sn

        VmV(V, S[0], T[2]);   // V = S[0] - T[2]
        Tp[2] = VdotV(V, Sn); // Tp[2] = (S[0] - T[2])·Sn

        // 如果 Sn 是分离方向，找出投影绝对值最小的顶点

        int point = -1;
        if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
        {
            if (Tp[0] < Tp[1])
                point = 0;
            else
                point = 1;
            if (Tp[2] < Tp[point])
                point = 2;
        }
        else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
        {
            if (Tp[0] > Tp[1])
                point = 0;
            else
                point = 1;
            if (Tp[2] > Tp[point])
                point = 2;
        }

        // 如果 Sn 是分离方向

        if (point >= 0)
        {
            shown_disjoint = 1;

            // 测试该顶点投影到另一个三角形上时，是否落在三角形内部

            VmV(V, T[point], S[0]); // V = T[point] - S[0]
            VcrossV(Z, Sn, Sv[0]);  // Z = Sn × Sv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, T[point], S[1]); // V = T[point] - S[1]
                VcrossV(Z, Sn, Sv[1]);  // Z = Sn × Sv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, T[point], S[2]); // V = T[point] - S[2]
                    VcrossV(Z, Sn, Sv[2]);  // Z = Sn × Sv[2]
                    if (VdotV(V, Z) > 0)
                    {
                        // T[point] 通过了测试，它是三角形 T 上的最近点；
                        // 另一个点位于三角形 S 的面上

                        VpVxS(P, T[point], Sn, Tp[point] / Snl); // P = T[point] + (Tp[point]/Snl) * Sn
                        VcV(Q, T[point]);                        // Q = T[point]
                        return sqrt(VdistV2(P, Q));              // 返回距离
                    }
                }
            }
        }
    }

    PQP_REAL Tn[3], Tnl;
    VcrossV(Tn, Tv[0], Tv[1]); // 计算三角形 T 的法向量
    Tnl = VdotV(Tn, Tn);       // 法向量长度的平方

    if (Tnl > 1e-15)
    {
        PQP_REAL Sp[3];

        VmV(V, T[0], S[0]);   // V = T[0] - S[0]
        Sp[0] = VdotV(V, Tn); // Sp[0] = (T[0] - S[0])·Tn

        VmV(V, T[0], S[1]);   // V = T[0] - S[1]
        Sp[1] = VdotV(V, Tn); // Sp[1] = (T[0] - S[1])·Tn

        VmV(V, T[0], S[2]);   // V = T[0] - S[2]
        Sp[2] = VdotV(V, Tn); // Sp[2] = (T[0] - S[2])·Tn

        int point = -1;
        if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
        {
            if (Sp[0] < Sp[1])
                point = 0;
            else
                point = 1;
            if (Sp[2] < Sp[point])
                point = 2;
        }
        else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
        {
            if (Sp[0] > Sp[1])
                point = 0;
            else
                point = 1;
            if (Sp[2] > Sp[point])
                point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;

            VmV(V, S[point], T[0]); // V = S[point] - T[0]
            VcrossV(Z, Tn, Tv[0]);  // Z = Tn × Tv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, S[point], T[1]); // V = S[point] - T[1]
                VcrossV(Z, Tn, Tv[1]);  // Z = Tn × Tv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, S[point], T[2]); // V = S[point] - T[2]
                    VcrossV(Z, Tn, Tv[2]);  // Z = Tn × Tv[2]
                    if (VdotV(V, Z) > 0)
                    {
                        VcV(P, S[point]);                        // P = S[point]
                        VpVxS(Q, S[point], Tn, Sp[point] / Tnl); // Q = S[point] + (Sp[point]/Tnl) * Tn
                        return sqrt(VdistV2(P, Q));
                    }
                }
            }
        }
    }

    // 无法证明情况1。
    // 如果上述测试之一已证明三角形分离，则假定情况3或4，
    // 否则认为情况2（三角形重叠）。

    if (shown_disjoint)
    {
        VcV(P, minP);
        VcV(Q, minQ);
        return sqrt(mindd);
    }
    else
        return 0;
}