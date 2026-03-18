//--------------------------------------------------------------------------
// 文件: TriDist.cpp
// 作者: Eric Larsen
// 说明:
// 包含 SegPoints() 函数用于计算两条线段之间的最近距离属性，
// 以及 TriDist() 函数用于计算两个三角形之间的最近距离属性。
//--------------------------------------------------------------------------

#include "MatVec.h"
#ifdef _WIN32
#include <float.h>
#include <math.h>
using namespace std;
#endif

//--------------------------------------------------------------------------
// SegPoints()
//
// 计算线段之间的最近距离。
// 实现了Lumelsky提出的算法：
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.
//--------------------------------------------------------------------------

void SegPoints(PQP_REAL VEC[3],
               PQP_REAL X[3], PQP_REAL Y[3],             // 最近点
               const PQP_REAL P[3], const PQP_REAL A[3], // 线段1的原点，方向向量
               const PQP_REAL Q[3], const PQP_REAL B[3]) // 线段2的原点，方向向量
{
    PQP_REAL T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
    PQP_REAL TMP[3];

    VmV(T, Q, P);          // T = Q - P
    A_dot_A = VdotV(A, A); // A点A
    B_dot_B = VdotV(B, B); // B点B
    A_dot_B = VdotV(A, B); // A点B
    A_dot_T = VdotV(A, T); // A点T
    B_dot_T = VdotV(B, T); // B点T

    // t 参数描述点 P + t*A
    // u 参数描述点 Q + u*B

    PQP_REAL t, u;

    // 计算点 P + t*A 与点 Q + u*B 的垂线对应的 t

    PQP_REAL denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

    t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

    // 限制点在直线 P,A 上

    if ((t < 0) || isnan(t))
        t = 0;
    else if (t > 1)
        t = 1;

    // 找出点 Q + u*B 到点 P + t*A 的垂线对应的 u

    u = (t * A_dot_B - B_dot_T) / B_dot_B;

    // 如果 u 在线段 Q,B 内，则 t 和 u 是对应的最近点；
    // 否则，将 u 限制到线段内，并重新计算 t 得到新的 t

    if ((u <= 0) || isnan(u))
    {

        VcV(Y, Q); // Y = Q

        t = A_dot_T / A_dot_A;

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);      // X = P
            VmV(VEC, Q, P); // VEC = Q - P
        }
        else if (t >= 1)
        {
            VpV(X, P, A);   // X = P + A
            VmV(VEC, Q, X); // VEC = Q - X
        }
        else
        {
            VpVxS(X, P, A, t);    // X = P + t*A
            VcrossV(TMP, T, A);   // TMP = T 叉 A
            VcrossV(VEC, A, TMP); // VEC = A 叉 (T 叉 A)
        }
    }
    else if (u >= 1)
    {

        VpV(Y, Q, B); // Y = Q + B

        t = (A_dot_B + A_dot_T) / A_dot_A;

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);      // X = P
            VmV(VEC, Y, P); // VEC = Y - P
        }
        else if (t >= 1)
        {
            VpV(X, P, A);   // X = P + A
            VmV(VEC, Y, X); // VEC = Y - X
        }
        else
        {
            VpVxS(X, P, A, t);    // X = P + t*A
            VmV(T, Y, P);         // T = Y - P
            VcrossV(TMP, T, A);   // TMP = (Y-P) 叉 A
            VcrossV(VEC, A, TMP); // VEC = A 叉 ((Y-P)叉A)
        }
    }
    else
    {

        VpVxS(Y, Q, B, u); // Y = Q + u*B

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);            // X = P
            VcrossV(TMP, T, B);   // TMP = (Q-P) 叉 B
            VcrossV(VEC, B, TMP); // VEC = B 叉 ((Q-P)叉B)
        }
        else if (t >= 1)
        {
            VpV(X, P, A);         // X = P + A
            VmV(T, Q, X);         // T = Q - X
            VcrossV(TMP, T, B);   // TMP = (Q-X) 叉 B
            VcrossV(VEC, B, TMP); // VEC = B 叉 ((Q-X)叉B)
        }
        else
        {
            VpVxS(X, P, A, t);  // X = P + t*A
            VcrossV(VEC, A, B); // VEC = A 叉 B
            if (VdotV(VEC, T) < 0)
            {
                VxS(VEC, VEC, -1); // 使 VEC 与 T 方向相反，反转它
            }
        }
    }
}

//--------------------------------------------------------------------------
// TriDist()
//
// 计算两个三角形之间的最近距离属性，返回它们之间的距离。
// S 和 T 是两个三角形，存储格式为 tri[顶点索引][坐标分量]：
//
// 如果两个三角形不相交（或相距很远），P 和 Q 分别是 S 和 T 上的最近点。
// 如果两个三角形重叠，P 和 Q 可能是任意一对重叠点，或者是某条边上的公共点（虽然这种情况不常见）。
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

    // 对于每一对边，从边对开始构建一个"分割平面"（slab）。这个平面在给定的边向量方向上，
    // 范围在两个三角形的端点处。这样能够保证每个三角形的所有顶点都在这个平面之外。
    // 如果使用这些平面失败，则记录找到的最小距离，以及是否证明了不相交。不相交的证据可以用于后期的计算。

    PQP_REAL V[3];
    PQP_REAL Z[3];
    PQP_REAL minP[3], minQ[3], mindd;
    int shown_disjoint = 0;

    mindd = VdistV2(S[0], T[0]) + 1; // 初始化最小值为一个足够大的值（S[0]和T[0]的距离平方+1）

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 在边 i 和边 j 上找最近点，以及它们之间的距离和平面法向量

            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            VmV(V, Q, P);              // V = Q - P
            PQP_REAL dd = VdotV(V, V); // dd = |V|^2

            // 如果距离平方小于等于当前最小值，则验证这些点

            if (dd <= mindd)
            {
                VcV(minP, P); // 保存所选点
                VcV(minQ, Q);
                mindd = dd;

                VmV(Z, S[(i + 2) % 3], P);  // Z = S[第三边起点] - P
                PQP_REAL a = VdotV(Z, VEC); // a = Z 点 VEC
                VmV(Z, T[(j + 2) % 3], Q);  // Z = T[第三边起点] - Q
                PQP_REAL b = VdotV(Z, VEC); // b = Z 点 VEC

                if ((a <= 0) && (b >= 0))
                    return sqrt(dd); // 如果三角形被该边对分割，则边对的最近点即为全局最近点

                PQP_REAL p = VdotV(V, VEC); // p = V 点 VEC

                if (a < 0)
                    a = 0;
                if (b > 0)
                    b = 0;
                if ((p - a + b) > 0)
                    shown_disjoint = 1; // 如果 p - a + b > 0，则证明三角形不相交
            }
        }
    }

    // 没有边对确定最近点，此时可能的情况：
    // 1. 一个三角形的某个顶点，正好位于另一个三角形的内部
    // 2. 两个三角形重叠
    // 3. 一个三角形的边与另一个三角形的平面平行（情况1和2可以合并），
    //    那么之前9对边计算中找到的最小距离即为三角形的最近点。
    // 4. 如果两个三角形非常接近或重叠，有时使用边对方法会失败，
    //    在某些情况下也可能失败。

    // 首先检查情况1

    PQP_REAL Sn[3], Snl;
    VcrossV(Sn, Sv[0], Sv[1]); // 计算三角形 S 的法向量
    Snl = VdotV(Sn, Sn);       // 计算法向量长度的平方

    // 如果法向量足够大（非退化）

    if (Snl > 1e-15)
    {
        // 计算 T 的顶点在法向量上的投影

        PQP_REAL Tp[3];

        VmV(V, S[0], T[0]);   // V = S[0] - T[0]
        Tp[0] = VdotV(V, Sn); // Tp[0] = (S[0] - T[0])点Sn

        VmV(V, S[0], T[1]);   // V = S[0] - T[1]
        Tp[1] = VdotV(V, Sn); // Tp[1] = (S[0] - T[1])点Sn

        VmV(V, S[0], T[2]);   // V = S[0] - T[2]
        Tp[2] = VdotV(V, Sn); // Tp[2] = (S[0] - T[2])点Sn

        // 判断 Sn 是否与法向量方向相同，找出投影中值最小的顶点

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

        // 如果 Sn 是否与法向量方向相反

        if (point >= 0)
        {
            shown_disjoint = 1;

            // 检查该顶点是否投影到三角形 S 的内部

            VmV(V, T[point], S[0]); // V = T[point] - S[0]
            VcrossV(Z, Sn, Sv[0]);  // Z = Sn 叉 Sv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, T[point], S[1]); // V = T[point] - S[1]
                VcrossV(Z, Sn, Sv[1]);  // Z = Sn 叉 Sv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, T[point], S[2]); // V = T[point] - S[2]
                    VcrossV(Z, Sn, Sv[2]);  // Z = Sn 叉 Sv[2]
                    if (VdotV(V, Z) > 0)
                    {
                        // T[point] 通过了测试，因此是 T 上的最近点；
                        // 其最近位置是从 T[point] 向三角形 S 投影

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
    Tnl = VdotV(Tn, Tn);       // 计算法向量长度的平方

    if (Tnl > 1e-15)
    {
        PQP_REAL Sp[3];

        VmV(V, T[0], S[0]);   // V = T[0] - S[0]
        Sp[0] = VdotV(V, Tn); // Sp[0] = (T[0] - S[0])点Tn

        VmV(V, T[0], S[1]);   // V = T[0] - S[1]
        Sp[1] = VdotV(V, Tn); // Sp[1] = (T[0] - S[1])点Tn

        VmV(V, T[0], S[2]);   // V = T[0] - S[2]
        Sp[2] = VdotV(V, Tn); // Sp[2] = (T[0] - S[2])点Tn

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
            VcrossV(Z, Tn, Tv[0]);  // Z = Tn 叉 Tv[0]
            if (VdotV(V, Z) > 0)
            {
                VmV(V, S[point], T[1]); // V = S[point] - T[1]
                VcrossV(Z, Tn, Tv[1]);  // Z = Tn 叉 Tv[1]
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, S[point], T[2]); // V = S[point] - T[2]
                    VcrossV(Z, Tn, Tv[2]);  // Z = Tn 叉 Tv[2]
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

    // 无法证明情况1
    // 如果我们证明了三角形不相交，则无需证明情况3和4
    // 否则，我们将它们视为情况2，即三角形重叠

    if (shown_disjoint)
    {
        VcV(P, minP);
        VcV(Q, minQ);
        return sqrt(mindd);
    }
    else
        return 0;
}
