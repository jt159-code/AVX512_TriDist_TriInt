//--------------------------------------------------------------------------
// 文件: TriDist.cpp
// 作者: Eric Larsen
// 描述:
// 包含 SegPoints() 函数，用于计算两条线段之间的最近点对；
// 以及 TriDist() 函数，用于计算两个三角形之间的最近点对。
//--------------------------------------------------------------------------

#include "MatVec.h"
#ifdef _WIN32
#include <float.h>
#define isnan _isnan
#endif

//--------------------------------------------------------------------------
// SegPoints() 
//
// 计算两条线段之间的最近点对。
// 实现基于以下论文中的算法：
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.   
//--------------------------------------------------------------------------

void
SegPoints(PQP_REAL VEC[3],
    PQP_REAL X[3], PQP_REAL Y[3],             // 最近点
    const PQP_REAL P[3], const PQP_REAL A[3], // 线段1的原点，方向向量
    const PQP_REAL Q[3], const PQP_REAL B[3]) // 线段2的原点，方向向量
{
    PQP_REAL T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
    PQP_REAL TMP[3];

    VmV(T, Q, P);                     // T = Q - P
    A_dot_A = VdotV(A, A);            // A·A
    B_dot_B = VdotV(B, B);            // B·B
    A_dot_B = VdotV(A, B);            // A·B
    A_dot_T = VdotV(A, T);            // A·T
    B_dot_T = VdotV(B, T);            // B·T

    // t 参数化射线 P + t*A
    // u 参数化射线 Q + u*B

    PQP_REAL t, u;

    // 计算射线 P + t*A 上离射线 Q + u*B 最近的点对应的 t

    PQP_REAL denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

    t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

    // 将结果限制在线段 P,A 内

    if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

    // 找出射线 Q + u*B 上离点 P + t*A 最近的点对应的 u

    u = (t * A_dot_B - B_dot_T) / B_dot_B;

    // 如果 u 在线段 Q,B 内，则 t 和 u 对应最近点；
    // 否则，将 u 限制在线段内，重新计算 t 并限制 t

    if ((u <= 0) || isnan(u)) {

        VcV(Y, Q);                     // Y = Q

        t = A_dot_T / A_dot_A;

        if ((t <= 0) || isnan(t)) {
            VcV(X, P);                   // X = P
            VmV(VEC, Q, P);              // VEC = Q - P
        }
        else if (t >= 1) {
            VpV(X, P, A);                // X = P + A
            VmV(VEC, Q, X);              // VEC = Q - X
        }
        else {
            VpVxS(X, P, A, t);           // X = P + t*A
            VcrossV(TMP, T, A);          // TMP = T × A
            VcrossV(VEC, A, TMP);        // VEC = A × (T × A)
        }
    }
    else if (u >= 1) {

        VpV(Y, Q, B);                  // Y = Q + B

        t = (A_dot_B + A_dot_T) / A_dot_A;

        if ((t <= 0) || isnan(t)) {
            VcV(X, P);                   // X = P
            VmV(VEC, Y, P);              // VEC = Y - P
        }
        else if (t >= 1) {
            VpV(X, P, A);                // X = P + A
            VmV(VEC, Y, X);              // VEC = Y - X
        }
        else {
            VpVxS(X, P, A, t);           // X = P + t*A
            VmV(T, Y, P);                // T = Y - P
            VcrossV(TMP, T, A);          // TMP = (Y-P) × A
            VcrossV(VEC, A, TMP);        // VEC = A × ((Y-P)×A)
        }
    }
    else {

        VpVxS(Y, Q, B, u);             // Y = Q + u*B

        if ((t <= 0) || isnan(t)) {
            VcV(X, P);                   // X = P
            VcrossV(TMP, T, B);          // TMP = (Q-P) × B
            VcrossV(VEC, B, TMP);        // VEC = B × ((Q-P)×B)
        }
        else if (t >= 1) {
            VpV(X, P, A);                // X = P + A
            VmV(T, Q, X);                // T = Q - X
            VcrossV(TMP, T, B);          // TMP = (Q-X) × B
            VcrossV(VEC, B, TMP);        // VEC = B × ((Q-X)×B)
        }
        else {
            VpVxS(X, P, A, t);           // X = P + t*A
            VcrossV(VEC, A, B);          // VEC = A × B
            if (VdotV(VEC, T) < 0) {
                VxS(VEC, VEC, -1);         // 如果 VEC 与 T 方向相反，则反向
            }
        }
    }
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

    VmV(Sv[0], S[1], S[0]);        // Sv[0] = S[1] - S[0]
    VmV(Sv[1], S[2], S[1]);        // Sv[1] = S[2] - S[1]
    VmV(Sv[2], S[0], S[2]);        // Sv[2] = S[0] - S[2]

    VmV(Tv[0], T[1], T[0]);        // Tv[0] = T[1] - T[0]
    VmV(Tv[1], T[2], T[1]);        // Tv[1] = T[2] - T[1]
    VmV(Tv[2], T[0], T[2]);        // Tv[2] = T[0] - T[2]

    // 对于每一对边，连接边对最近点的向量定义了一个“平板”（slab），
    // 由平行于该向量的平面在端点处包围。如果能够证明每个三角形中
    // 不在该边上的顶点都在平板之外，那么该边对的最近点就是三角形的最近点。
    // 即使这些测试失败，记录找到的最近点以及是否已证明三角形分离仍可能有帮助。

    PQP_REAL V[3];
    PQP_REAL Z[3];
    PQP_REAL minP[3], minQ[3], mindd;
    int shown_disjoint = 0;

    mindd = VdistV2(S[0], T[0]) + 1;  // 初始最小值设为一个足够大的值（S[0]到T[0]距离平方+1）

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 计算边 i 和边 j 上的最近点，以及它们之间的向量（和距离平方）

            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            VmV(V, Q, P);                   // V = Q - P
            PQP_REAL dd = VdotV(V, V);     // dd = |V|^2

            // 仅当距离平方小于当前最小值时才验证这对最近点

            if (dd <= mindd)
            {
                VcV(minP, P);                 // 保存候选点
                VcV(minQ, Q);
                mindd = dd;

                VmV(Z, S[(i + 2) % 3], P);         // Z = S[第三个顶点] - P
                PQP_REAL a = VdotV(Z, VEC);   // a = Z · VEC
                VmV(Z, T[(j + 2) % 3], Q);         // Z = T[第三个顶点] - Q
                PQP_REAL b = VdotV(Z, VEC);   // b = Z · VEC

                if ((a <= 0) && (b >= 0)) return sqrt(dd);  // 若满足条件，则该边对最近点即为全局最近点

                PQP_REAL p = VdotV(V, VEC);  // p = V · VEC

                if (a < 0) a = 0;
                if (b > 0) b = 0;
                if ((p - a + b) > 0) shown_disjoint = 1;    // 如果 p - a + b > 0，可证明三角形分离
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
    VcrossV(Sn, Sv[0], Sv[1]);      // 计算三角形 S 的法向量
    Snl = VdotV(Sn, Sn);            // 法向量长度的平方

    // 如果叉积长度足够大（非退化）

    if (Snl > 1e-15)
    {
        // 计算 T 的三个顶点在法向上的投影长度

        PQP_REAL Tp[3];

        VmV(V, S[0], T[0]);            // V = S[0] - T[0]
        Tp[0] = VdotV(V, Sn);         // Tp[0] = (S[0] - T[0])·Sn

        VmV(V, S[0], T[1]);            // V = S[0] - T[1]
        Tp[1] = VdotV(V, Sn);         // Tp[1] = (S[0] - T[1])·Sn

        VmV(V, S[0], T[2]);            // V = S[0] - T[2]
        Tp[2] = VdotV(V, Sn);         // Tp[2] = (S[0] - T[2])·Sn

        // 如果 Sn 是分离方向，找出投影绝对值最小的顶点

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

        // 如果 Sn 是分离方向

        if (point >= 0)
        {
            shown_disjoint = 1;

            // 测试该顶点投影到另一个三角形上时，是否落在三角形内部

            VmV(V, T[point], S[0]);        // V = T[point] - S[0]
            VcrossV(Z, Sn, Sv[0]);         // Z = Sn × Sv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, T[point], S[1]);      // V = T[point] - S[1]
                VcrossV(Z, Sn, Sv[1]);       // Z = Sn × Sv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, T[point], S[2]);    // V = T[point] - S[2]
                    VcrossV(Z, Sn, Sv[2]);     // Z = Sn × Sv[2]
                    if (VdotV(V, Z) > 0)
                    {
                        // T[point] 通过了测试，它是三角形 T 上的最近点；
                        // 另一个点位于三角形 S 的面上

                        VpVxS(P, T[point], Sn, Tp[point] / Snl);   // P = T[point] + (Tp[point]/Snl) * Sn
                        VcV(Q, T[point]);                       // Q = T[point]
                        return sqrt(VdistV2(P, Q));             // 返回距离
                    }
                }
            }
        }
    }

    PQP_REAL Tn[3], Tnl;
    VcrossV(Tn, Tv[0], Tv[1]);       // 计算三角形 T 的法向量
    Tnl = VdotV(Tn, Tn);             // 法向量长度的平方      

    if (Tnl > 1e-15)
    {
        PQP_REAL Sp[3];

        VmV(V, T[0], S[0]);             // V = T[0] - S[0]
        Sp[0] = VdotV(V, Tn);          // Sp[0] = (T[0] - S[0])·Tn

        VmV(V, T[0], S[1]);             // V = T[0] - S[1]
        Sp[1] = VdotV(V, Tn);          // Sp[1] = (T[0] - S[1])·Tn

        VmV(V, T[0], S[2]);             // V = T[0] - S[2]
        Sp[2] = VdotV(V, Tn);          // Sp[2] = (T[0] - S[2])·Tn

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

            VmV(V, S[point], T[0]);        // V = S[point] - T[0]
            VcrossV(Z, Tn, Tv[0]);         // Z = Tn × Tv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, S[point], T[1]);      // V = S[point] - T[1]
                VcrossV(Z, Tn, Tv[1]);       // Z = Tn × Tv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, S[point], T[2]);    // V = S[point] - T[2]
                    VcrossV(Z, Tn, Tv[2]);     // Z = Tn × Tv[2]
                    if (VdotV(V, Z) > 0)
                    {
                        VcV(P, S[point]);                       // P = S[point]
                        VpVxS(Q, S[point], Tn, Sp[point] / Tnl);   // Q = S[point] + (Sp[point]/Tnl) * Tn
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
    else return 0;
}