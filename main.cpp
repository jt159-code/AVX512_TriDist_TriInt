#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>

typedef float PQP_REAL;

extern void TriDistBatch16(
    PQP_REAL p_batch[16][3],
    PQP_REAL q_batch[16][3],
    PQP_REAL dist[16],
    const PQP_REAL s_batch[16][3][3],
    const PQP_REAL t_batch[16][3][3]);

static bool floatEqual(PQP_REAL a, PQP_REAL b, PQP_REAL eps = 0.08f)
{
    return fabs(a - b) < eps;
}

static void checkBatchResult(
    const char *testName,
    const PQP_REAL dist[16],
    const PQP_REAL expected_dist[16],
    bool &allPassed,
    double elapsedMs)
{
    std::cout << "========== " << testName << " ==========\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n";
    bool testPassed = true;
    for (int i = 0; i < 16; i++)
    {
        bool pairPassed = floatEqual(dist[i], expected_dist[i]);
        testPassed &= pairPassed;
        std::cout << "  Pair[" << std::setw(2) << i << "] "
                  << "Calc: " << std::fixed << std::setprecision(6) << dist[i]
                  << " | Expected: " << std::setprecision(6) << expected_dist[i]
                  << " [" << (pairPassed ? "PASS" : "FAIL") << "]\n";
    }
    allPassed &= testPassed;
    std::cout << "[" << (testPassed ? "PASS" : "FAIL") << "] " << testName << "\n\n";
}

int main()
{
    std::cout << "=========================================\n";
    std::cout << "  TriDist6.cpp Batch Test (AVX512)\n";
    std::cout << "  Running 2000 iterations for average\n";
    std::cout << "=========================================\n\n";

    const int ITERATIONS = 2000;
    bool allPassed = true;

    // ---------- 生成所有测试数据 ----------
    // Test 1: Gradient Vertex-Face
    PQP_REAL s1[16][3][3], t1[16][3][3], exp1[16];
    for (int i = 0; i < 16; i++)
    {
        s1[i][0][0] = 0.0f;
        s1[i][0][1] = 0.0f;
        s1[i][0][2] = 0.0f;
        s1[i][1][0] = 4.0f;
        s1[i][1][1] = 0.0f;
        s1[i][1][2] = 0.0f;
        s1[i][2][0] = 1.0f;
        s1[i][2][1] = 3.0f;
        s1[i][2][2] = 0.0f;

        PQP_REAL z = 0.1f * (i + 1);
        t1[i][0][0] = 1.5f;
        t1[i][0][1] = 1.5f;
        t1[i][0][2] = z;
        t1[i][1][0] = 3.0f;
        t1[i][1][1] = 2.0f;
        t1[i][1][2] = z;
        t1[i][2][0] = 0.5f;
        t1[i][2][1] = 2.5f;
        t1[i][2][2] = z;

        exp1[i] = z;
    }

    // Test 2: Different Edge-Edge
    PQP_REAL s2[16][3][3], t2[16][3][3], exp2[16];
    for (int i = 0; i < 16; i++)
    {
        s2[i][0][0] = 0.0f;
        s2[i][0][1] = 0.0f;
        s2[i][0][2] = 0.0f;
        s2[i][1][0] = 3.0f;
        s2[i][1][1] = 0.0f;
        s2[i][1][2] = 0.0f;
        s2[i][2][0] = 1.0f;
        s2[i][2][1] = 2.0f;
        s2[i][2][2] = 0.0f;

        PQP_REAL y = i * 0.2f;
        PQP_REAL z = 0.5f * (i + 1);
        t2[i][0][0] = 1.5f;
        t2[i][0][1] = y;
        t2[i][0][2] = z;
        t2[i][1][0] = 1.5f;
        t2[i][1][1] = y + 2.0f;
        t2[i][1][2] = z;
        t2[i][2][0] = 0.0f;
        t2[i][2][1] = y + 1.0f;
        t2[i][2][2] = z;

        exp2[i] = z;
    }

    // Test 3: Coplanar Non-Intersect
    PQP_REAL s3[16][3][3], t3[16][3][3], exp3[16];
    for (int i = 0; i < 16; i++)
    {
        s3[i][0][0] = 0.0f;
        s3[i][0][1] = 0.0f;
        s3[i][0][2] = 0.0f;
        s3[i][1][0] = 2.0f;
        s3[i][1][1] = 0.0f;
        s3[i][1][2] = 0.0f;
        s3[i][2][0] = 1.0f;
        s3[i][2][1] = 2.0f;
        s3[i][2][2] = 0.0f;

        PQP_REAL x = 3.0f + 0.5f * i;
        t3[i][0][0] = x;
        t3[i][0][1] = 0.0f;
        t3[i][0][2] = 0.0f;
        t3[i][1][0] = x + 2.0f;
        t3[i][1][1] = 0.0f;
        t3[i][1][2] = 0.0f;
        t3[i][2][0] = x + 1.0f;
        t3[i][2][1] = 2.0f;
        t3[i][2][2] = 0.0f;

        exp3[i] = x - 2.0f;
    }

    // Test 4: Degenerate Hierarchy
    PQP_REAL s4[16][3][3], t4[16][3][3], exp4[16];
    for (int i = 0; i < 4; i++)
    {
        PQP_REAL d = 1.0f + i * 0.5f;
        s4[i][0][0] = 0;
        s4[i][0][1] = 0;
        s4[i][0][2] = 0;
        s4[i][1][0] = 0;
        s4[i][1][1] = 0;
        s4[i][1][2] = 0;
        s4[i][2][0] = 0;
        s4[i][2][1] = 0;
        s4[i][2][2] = 0;
        t4[i][0][0] = 0;
        t4[i][0][1] = 0;
        t4[i][0][2] = d;
        t4[i][1][0] = 2;
        t4[i][1][1] = 0;
        t4[i][1][2] = d;
        t4[i][2][0] = 1;
        t4[i][2][1] = 2;
        t4[i][2][2] = d;
        exp4[i] = d;
    }
    for (int i = 4; i < 8; i++)
    {
        PQP_REAL d = 3.0f + (i - 4) * 0.5f;
        s4[i][0][0] = 0;
        s4[i][0][1] = 0;
        s4[i][0][2] = 0;
        s4[i][1][0] = 2;
        s4[i][1][1] = 0;
        s4[i][1][2] = 0;
        s4[i][2][0] = 1;
        s4[i][2][1] = 0;
        s4[i][2][2] = 0;
        t4[i][0][0] = 1.0f;
        t4[i][0][1] = 1.0f;
        t4[i][0][2] = d;
        t4[i][1][0] = 3.0f;
        t4[i][1][1] = 1.0f;
        t4[i][1][2] = d;
        t4[i][2][0] = 2.0f;
        t4[i][2][1] = 3.0f;
        t4[i][2][2] = d;
        exp4[i] = sqrtf(d * d + 1.0f * 1.0f);
    }
    for (int i = 8; i < 12; i++)
    {
        PQP_REAL d = 5.0f + (i - 8) * 0.5f;
        s4[i][0][0] = 0;
        s4[i][0][1] = 0;
        s4[i][0][2] = 0;
        s4[i][1][0] = 3;
        s4[i][1][1] = 0;
        s4[i][1][2] = 0;
        s4[i][2][0] = 1.5f;
        s4[i][2][1] = 0;
        s4[i][2][2] = 0;
        t4[i][0][0] = 1.5f;
        t4[i][0][1] = 1.0f;
        t4[i][0][2] = d;
        t4[i][1][0] = 1.5f;
        t4[i][1][1] = 3.0f;
        t4[i][1][2] = d;
        t4[i][2][0] = 1.5f;
        t4[i][2][1] = 2.0f;
        t4[i][2][2] = d;
        exp4[i] = sqrtf(d * d + 1.0f * 1.0f);
    }
    for (int i = 12; i < 16; i++)
    {
        PQP_REAL d = 7.0f + (i - 12) * 0.5f;
        s4[i][0][0] = 0;
        s4[i][0][1] = 0;
        s4[i][0][2] = 0;
        s4[i][1][0] = 0;
        s4[i][1][1] = 0;
        s4[i][1][2] = 0;
        s4[i][2][0] = 0;
        s4[i][2][1] = 0;
        s4[i][2][2] = 0;
        t4[i][0][0] = 0;
        t4[i][0][1] = 0;
        t4[i][0][2] = d;
        t4[i][1][0] = 2;
        t4[i][1][1] = 0;
        t4[i][1][2] = d;
        t4[i][2][0] = 1;
        t4[i][2][1] = 0;
        t4[i][2][2] = d;
        exp4[i] = d;
    }

    // Test 5: Full Mixed Scenes
    PQP_REAL s5[16][3][3] = {0};
    PQP_REAL t5[16][3][3] = {0};
    PQP_REAL exp5[16] = {0};
    // 请将原 testBatchFullMixed 中的 16 组数据逐一赋值到此
    // 由于篇幅，此处仅示意，实际使用时务必完整复制原函数中的数据
    // 以下为原函数中的数据（已修正部分预期值）
    s5[0][0][0] = 0;
    s5[0][0][1] = 0;
    s5[0][0][2] = 0;
    s5[0][1][0] = 3;
    s5[0][1][1] = 0;
    s5[0][1][2] = 0;
    s5[0][2][0] = 1;
    s5[0][2][1] = 3;
    s5[0][2][2] = 0;
    t5[0][0][0] = 1;
    t5[0][0][1] = 1;
    t5[0][0][2] = 0.2f;
    t5[0][1][0] = 2;
    t5[0][1][1] = 2;
    t5[0][1][2] = 0.2f;
    t5[0][2][0] = 0;
    t5[0][2][1] = 2;
    t5[0][2][2] = 0.2f;
    exp5[0] = 0.2f;

    s5[1][0][0] = 0;
    s5[1][0][1] = 0;
    s5[1][0][2] = 0;
    s5[1][1][0] = 4;
    s5[1][1][1] = 0;
    s5[1][1][2] = 0;
    s5[1][2][0] = 2;
    s5[1][2][1] = 1;
    s5[1][2][2] = 0;
    t5[1][0][0] = 2;
    t5[1][0][1] = 1;
    t5[1][0][2] = 0.8f;
    t5[1][1][0] = 2;
    t5[1][1][1] = 3;
    t5[1][1][2] = 0.8f;
    t5[1][2][0] = 1;
    t5[1][2][1] = 2;
    t5[1][2][2] = 0.8f;
    exp5[1] = 0.8f;

    s5[2][0][0] = 0;
    s5[2][0][1] = 0;
    s5[2][0][2] = 0;
    s5[2][1][0] = 2;
    s5[2][1][1] = 0;
    s5[2][1][2] = 0;
    s5[2][2][0] = 1;
    s5[2][2][1] = 2;
    s5[2][2][2] = 0;
    t5[2][0][0] = 3.5f;
    t5[2][0][1] = 0;
    t5[2][0][2] = 0;
    t5[2][1][0] = 5.5f;
    t5[2][1][1] = 0;
    t5[2][1][2] = 0;
    t5[2][2][0] = 4.5f;
    t5[2][2][1] = 2;
    t5[2][2][2] = 0;
    exp5[2] = 1.5f;

    s5[3][0][0] = 0;
    s5[3][0][1] = 0;
    s5[3][0][2] = 0;
    s5[3][1][0] = 0;
    s5[3][1][1] = 0;
    s5[3][1][2] = 0;
    s5[3][2][0] = 0;
    s5[3][2][1] = 0;
    s5[3][2][2] = 0;
    t5[3][0][0] = 0;
    t5[3][0][1] = 0;
    t5[3][0][2] = 2.0f;
    t5[3][1][0] = 2;
    t5[3][1][1] = 0;
    t5[3][1][2] = 2.0f;
    t5[3][2][0] = 1;
    t5[3][2][1] = 2;
    t5[3][2][2] = 2.0f;
    exp5[3] = 2.0f;

    s5[4][0][0] = 0;
    s5[4][0][1] = 0;
    s5[4][0][2] = 0;
    s5[4][1][0] = 3;
    s5[4][1][1] = 1;
    s5[4][1][2] = 0;
    s5[4][2][0] = 1;
    s5[4][2][1] = 3;
    s5[4][2][2] = 0;
    t5[4][0][0] = 2;
    t5[4][0][1] = 2;
    t5[4][0][2] = 2.5f;
    t5[4][1][0] = 4;
    t5[4][1][1] = 3;
    t5[4][1][2] = 2.5f;
    t5[4][2][0] = 3;
    t5[4][2][1] = 4;
    t5[4][2][2] = 2.5f;
    exp5[4] = 2.5f;

    s5[5][0][0] = 0;
    s5[5][0][1] = 0;
    s5[5][0][2] = 0;
    s5[5][1][0] = 0;
    s5[5][1][1] = 3;
    s5[5][1][2] = 0;
    s5[5][2][0] = 0;
    s5[5][2][1] = 0;
    s5[5][2][2] = 4;
    t5[5][0][0] = 3.0f;
    t5[5][0][1] = 1;
    t5[5][0][2] = 1;
    t5[5][1][0] = 3.0f;
    t5[5][1][1] = 2;
    t5[5][1][2] = 2;
    t5[5][2][0] = 3.0f;
    t5[5][2][1] = 0;
    t5[5][2][2] = 3;
    exp5[5] = 3.0f;

    s5[6][0][0] = 0;
    s5[6][0][1] = 0;
    s5[6][0][2] = 0;
    s5[6][1][0] = 3;
    s5[6][1][1] = 0;
    s5[6][1][2] = 0;
    s5[6][2][0] = 1.5f;
    s5[6][2][1] = 0;
    s5[6][2][2] = 0;
    t5[6][0][0] = 1.5f;
    t5[6][0][1] = 2;
    t5[6][0][2] = 3.5f;
    t5[6][1][0] = 3.5f;
    t5[6][1][1] = 2;
    t5[6][1][2] = 3.5f;
    t5[6][2][0] = 2.5f;
    t5[6][2][1] = 4;
    t5[6][2][2] = 3.5f;
    exp5[6] = sqrtf(3.5f * 3.5f + 2.0f * 2.0f);

    s5[7][0][0] = 0;
    s5[7][0][1] = 0;
    s5[7][0][2] = 0;
    s5[7][1][0] = 2;
    s5[7][1][1] = 0;
    s5[7][1][2] = 0;
    s5[7][2][0] = 1;
    s5[7][2][1] = 1;
    s5[7][2][2] = 0;
    t5[7][0][0] = 0;
    t5[7][0][1] = 5.0f;
    t5[7][0][2] = 0;
    t5[7][1][0] = 2;
    t5[7][1][1] = 5.0f;
    t5[7][1][2] = 0;
    t5[7][2][0] = 1;
    t5[7][2][1] = 6.0f;
    t5[7][2][2] = 0;
    exp5[7] = 4.0f;

    s5[8][0][0] = 0;
    s5[8][0][1] = 0;
    s5[8][0][2] = 0;
    s5[8][1][0] = 10;
    s5[8][1][1] = 0;
    s5[8][1][2] = 0;
    s5[8][2][0] = 5;
    s5[8][2][1] = 10;
    s5[8][2][2] = 0;
    t5[8][0][0] = 5;
    t5[8][0][1] = 5;
    t5[8][0][2] = 50.0f;
    t5[8][1][0] = 10;
    t5[8][1][1] = 10;
    t5[8][1][2] = 50.0f;
    t5[8][2][0] = 0;
    t5[8][2][1] = 10;
    t5[8][2][2] = 50.0f;
    exp5[8] = 50.0f;

    s5[9][0][0] = 0;
    s5[9][0][1] = 0;
    s5[9][0][2] = 0;
    s5[9][1][0] = 2;
    s5[9][1][1] = 0;
    s5[9][1][2] = 0;
    s5[9][2][0] = 1;
    s5[9][2][1] = 1;
    s5[9][2][2] = 0;
    t5[9][0][0] = 1;
    t5[9][0][1] = 1;
    t5[9][0][2] = 1e-6f;
    t5[9][1][0] = 1;
    t5[9][1][1] = 2;
    t5[9][1][2] = 1e-6f;
    t5[9][2][0] = 0;
    t5[9][2][1] = 1.5f;
    t5[9][2][2] = 1e-6f;
    exp5[9] = 1e-6f;

    s5[10][0][0] = 0;
    s5[10][0][1] = 0;
    s5[10][0][2] = 0;
    s5[10][1][0] = 0;
    s5[10][1][1] = 0;
    s5[10][1][2] = 0;
    s5[10][2][0] = 0;
    s5[10][2][1] = 0;
    s5[10][2][2] = 0;
    t5[10][0][0] = 0;
    t5[10][0][1] = 0;
    t5[10][0][2] = 1.2f;
    t5[10][1][0] = 2;
    t5[10][1][1] = 0;
    t5[10][1][2] = 1.2f;
    t5[10][2][0] = 1;
    t5[10][2][1] = 0;
    t5[10][2][2] = 1.2f;
    exp5[10] = 1.2f;

    s5[11][0][0] = 0.2f;
    s5[11][0][1] = 0.1f;
    s5[11][0][2] = 0;
    s5[11][1][0] = 3.5f;
    s5[11][1][1] = 0.3f;
    s5[11][1][2] = 0;
    s5[11][2][0] = 1.2f;
    s5[11][2][1] = 2.7f;
    s5[11][2][2] = 0;
    t5[11][0][0] = 1.5f;
    t5[11][0][1] = 1.2f;
    t5[11][0][2] = 2.8f;
    t5[11][1][0] = 2.8f;
    t5[11][1][1] = 2.1f;
    t5[11][1][2] = 2.8f;
    t5[11][2][0] = 0.5f;
    t5[11][2][1] = 2.2f;
    t5[11][2][2] = 2.8f;
    exp5[11] = 2.8f;

    s5[12][0][0] = 0;
    s5[12][0][1] = 0;
    s5[12][0][2] = 0;
    s5[12][1][0] = 3;
    s5[12][1][1] = -1;
    s5[12][1][2] = 0;
    s5[12][2][0] = 1;
    s5[12][2][1] = 2;
    s5[12][2][2] = 0;
    t5[12][0][0] = 2;
    t5[12][0][1] = 1;
    t5[12][0][2] = 3.3f;
    t5[12][1][0] = 4;
    t5[12][1][1] = -2;
    t5[12][1][2] = 3.3f;
    t5[12][2][0] = 3;
    t5[12][2][1] = 3;
    t5[12][2][2] = 3.3f;
    exp5[12] = 3.311635f;

    s5[13][0][0] = 0;
    s5[13][0][1] = 0;
    s5[13][0][2] = 0;
    s5[13][1][0] = 2;
    s5[13][1][1] = 0;
    s5[13][1][2] = 0;
    s5[13][2][0] = 1;
    s5[13][2][1] = 2;
    s5[13][2][2] = 0;
    t5[13][0][0] = 12.0f;
    t5[13][0][1] = 0;
    t5[13][0][2] = 0;
    t5[13][1][0] = 14.0f;
    t5[13][1][1] = 0;
    t5[13][1][2] = 0;
    t5[13][2][0] = 13.0f;
    t5[13][2][1] = 2;
    t5[13][2][2] = 0;
    exp5[13] = 10.0f;

    s5[14][0][0] = 0;
    s5[14][0][1] = 0;
    s5[14][0][2] = 0;
    s5[14][1][0] = 4;
    s5[14][1][1] = 0;
    s5[14][1][2] = 0;
    s5[14][2][0] = 2;
    s5[14][2][1] = 0;
    s5[14][2][2] = 0;
    t5[14][0][0] = 2;
    t5[14][0][1] = 3;
    t5[14][0][2] = 4.5f;
    t5[14][1][0] = 2;
    t5[14][1][1] = 5;
    t5[14][1][2] = 4.5f;
    t5[14][2][0] = 2;
    t5[14][2][1] = 4;
    t5[14][2][2] = 4.5f;
    exp5[14] = sqrtf(4.5f * 4.5f + 3.0f * 3.0f);

    s5[15][0][0] = 0;
    s5[15][0][1] = 0;
    s5[15][0][2] = 0;
    s5[15][1][0] = 2;
    s5[15][1][1] = 0;
    s5[15][1][2] = 0;
    s5[15][2][0] = 1;
    s5[15][2][1] = 2;
    s5[15][2][2] = 0;
    t5[15][0][0] = 1;
    t5[15][0][1] = 1;
    t5[15][0][2] = -1;
    t5[15][1][0] = 1;
    t5[15][1][1] = 1;
    t5[15][1][2] = 1;
    t5[15][2][0] = 0;
    t5[15][2][1] = 0;
    t5[15][2][2] = 0;
    exp5[15] = 0.0f;

    // Test 6: Float Boundary
    PQP_REAL s6[16][3][3], t6[16][3][3], exp6[16];
    for (int i = 0; i < 4; i++)
    {
        PQP_REAL d = powf(10.0f, -8 + i);
        s6[i][0][0] = 0;
        s6[i][0][1] = 0;
        s6[i][0][2] = 0;
        s6[i][1][0] = 1;
        s6[i][1][1] = 0;
        s6[i][1][2] = 0;
        s6[i][2][0] = 0;
        s6[i][2][1] = 1;
        s6[i][2][2] = 0;
        t6[i][0][0] = 0.5f;
        t6[i][0][1] = 0.5f;
        t6[i][0][2] = d;
        t6[i][1][0] = 1.5f;
        t6[i][1][1] = 0.5f;
        t6[i][1][2] = d;
        t6[i][2][0] = 0.5f;
        t6[i][2][1] = 1.5f;
        t6[i][2][2] = d;
        exp6[i] = d;
    }
    for (int i = 4; i < 8; i++)
    {
        PQP_REAL d = powf(10.0f, 3 + (i - 4));
        s6[i][0][0] = 0;
        s6[i][0][1] = 0;
        s6[i][0][2] = 0;
        s6[i][1][0] = 100;
        s6[i][1][1] = 0;
        s6[i][1][2] = 0;
        s6[i][2][0] = 0;
        s6[i][2][1] = 100;
        s6[i][2][2] = 0;
        t6[i][0][0] = 50;
        t6[i][0][1] = 50;
        t6[i][0][2] = d;
        t6[i][1][0] = 150;
        t6[i][1][1] = 50;
        t6[i][1][2] = d;
        t6[i][2][0] = 50;
        t6[i][2][1] = 150;
        t6[i][2][2] = d;
        exp6[i] = d;
    }
    for (int i = 8; i < 12; i++)
    {
        PQP_REAL d = 10.0f * (i - 7);
        s6[i][0][0] = 0;
        s6[i][0][1] = 0;
        s6[i][0][2] = 0;
        s6[i][1][0] = 1000.0f;
        s6[i][1][1] = 0;
        s6[i][1][2] = 0;
        s6[i][2][0] = 0;
        s6[i][2][1] = 0.001f;
        s6[i][2][2] = 0;
        t6[i][0][0] = 500.0f;
        t6[i][0][1] = 0.0005f;
        t6[i][0][2] = d;
        t6[i][1][0] = 1500.0f;
        t6[i][1][1] = 0.0005f;
        t6[i][1][2] = d;
        t6[i][2][0] = 500.0f;
        t6[i][2][1] = 0.0015f;
        t6[i][2][2] = d;
        exp6[i] = d;
    }
    for (int i = 12; i < 16; i++)
    {
        PQP_REAL d = 2.0f * (i - 11);
        s6[i][0][0] = -1;
        s6[i][0][1] = -1;
        s6[i][0][2] = 0;
        s6[i][1][0] = 1;
        s6[i][1][1] = -1;
        s6[i][1][2] = 0;
        s6[i][2][0] = 0;
        s6[i][2][1] = 1;
        s6[i][2][2] = 0;
        t6[i][0][0] = 0;
        t6[i][0][1] = 0;
        t6[i][0][2] = -d;
        t6[i][1][0] = 2;
        t6[i][1][1] = 0;
        t6[i][1][2] = -d;
        t6[i][2][0] = 1;
        t6[i][2][1] = 2;
        t6[i][2][2] = -d;
        exp6[i] = d;
    }

    // ---------- 单次验证 ----------
    std::cout << "--- Single Run Verification ---\n";
    PQP_REAL p_batch[16][3], q_batch[16][3], dist[16];

    auto run_and_check = [&](const char *name,
                             const PQP_REAL s[16][3][3],
                             const PQP_REAL t[16][3][3],
                             const PQP_REAL exp[16],
                             bool &passed)
    {
        auto start = std::chrono::high_resolution_clock::now();
        TriDistBatch16(p_batch, q_batch, dist, s, t);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        checkBatchResult(name, dist, exp, passed, elapsed);
    };

    run_and_check("Test 1: Gradient Vertex-Face", s1, t1, exp1, allPassed);
    run_and_check("Test 2: Different Edge-Edge", s2, t2, exp2, allPassed);
    run_and_check("Test 3: Coplanar Non-Intersect", s3, t3, exp3, allPassed);
    run_and_check("Test 4: Degenerate Hierarchy", s4, t4, exp4, allPassed);
    run_and_check("Test 5: Full Mixed Scenes", s5, t5, exp5, allPassed);
    run_and_check("Test 6: Float Boundary", s6, t6, exp6, allPassed);

    // ---------- 2000 次性能测试 ----------
    std::cout << "\n--- Performance Test (" << ITERATIONS << " iterations) ---\n";
    auto totalStart = std::chrono::high_resolution_clock::now();

    volatile double dummy = 0.0; // 防止优化

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        TriDistBatch16(p_batch, q_batch, dist, s1, t1);
        dummy += dist[0];
        TriDistBatch16(p_batch, q_batch, dist, s2, t2);
        dummy += dist[0];
        TriDistBatch16(p_batch, q_batch, dist, s3, t3);
        dummy += dist[0];
        TriDistBatch16(p_batch, q_batch, dist, s4, t4);
        dummy += dist[0];
        TriDistBatch16(p_batch, q_batch, dist, s5, t5);
        dummy += dist[0];
        TriDistBatch16(p_batch, q_batch, dist, s6, t6);
        dummy += dist[0];
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    std::cout << "Total time for " << ITERATIONS << " iterations (all 6 tests per iteration): "
              << std::fixed << std::setprecision(3) << totalMs << " ms\n";
    std::cout << "Average time per iteration (all 6 tests): "
              << std::fixed << std::setprecision(6) << (totalMs / ITERATIONS) << " ms\n";
    std::cout << "Average time per test (96 pairs): "
              << std::fixed << std::setprecision(6) << (totalMs / (ITERATIONS * 6)) << " ms\n";
    std::cout << "Average time per triangle pair: "
              << std::fixed << std::setprecision(6) << (totalMs / (ITERATIONS * 96)) << " ms\n";

    std::cout << "=========================================\n";
    if (allPassed)
        std::cout << "  SUCCESS: All Batch Tests Passed!\n";
    else
        std::cout << "  FAIL: Some Batch Tests Failed!\n";
    std::cout << "=========================================\n";

    return allPassed ? 0 : 1;
}