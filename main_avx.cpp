// main.cpp
// 与 Tri_avx.cpp 配合使用的测试程序

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float PQP_REAL;

namespace tdbase {
    // 必须与 Tri_avx.cpp 中的定义完全一致（对齐64字节）
    struct alignas(64) TriPairBatch {
        float p1x[16], p1y[16], p1z[16];
        float q1x[16], q1y[16], q1z[16];
        float r1x[16], r1y[16], r1z[16];
        float p2x[16], p2y[16], p2z[16];
        float q2x[16], q2y[16], q2z[16];
        float r2x[16], r2y[16], r2z[16];
    };

    bool TriIntBatch(const TriPairBatch& batch, bool results[16]);
    bool TriInt(const PQP_REAL* data1, const PQP_REAL* data2);
}

using namespace tdbase;

struct TestCase {
    const char* description;
    PQP_REAL tri1[9];
    PQP_REAL tri2[9];
    bool expected;      // true=相交, false=不相交
};

void print_triangle(const PQP_REAL tri[9], const char* name) {
    printf("%s: (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f)\n",
           name,
           tri[0], tri[1], tri[2],
           tri[3], tri[4], tri[5],
           tri[6], tri[7], tri[8]);
}

int main() {
    TestCase tests[] = {
        {"分离 - 两个三角形在Z=0平面但不相交",
         {0,0,0, 1,0,0, 0,1,0},
         {2,2,0, 3,2,0, 2,3,0},
         false},

        {"顶点包含 - 三角形2的顶点(1,1,0)在三角形1内部",
         {0,0,0, 2,0,0, 0,2,0},
         {1,1,0, 3,1,0, 1,3,0},
         true},

        {"边穿过 - 三角形2的边(0.5,-0.5,0)-(0.5,1.5,0)穿过三角形1",
         {0,0,0, 2,0,0, 0,2,0},
         {0.5f,-0.5f,0, 0.5f,1.5f,0, 1.5f,0.5f,0},
         true},

        {"共面相交 - 两个三角形在同一平面且有重叠",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0,0, 3,0,0, 1,2,0},
         true},

        {"共面不相交 - 同一平面但分离",
         {0,0,0, 1,0,0, 0,1,0},
         {2,0,0, 3,0,0, 2,1,0},
         false},

        {"平行分离 - Z=0 和 Z=1 平面，投影无重叠",
         {0,0,0, 1,0,0, 0,1,0},
         {0,0,1, 1,0,1, 0,1,1},
         false},

        {"点接触 - 共享顶点(0,0,0)",
         {0,0,0, 1,0,0, 0,1,0},
         {0,0,0, 1,0,0.1f, 0,1,0.1f},
         true},

        {"边重叠 - 共享一条边的一部分",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0,0, 3,0,0, 1,2,0},
         true},

        {"完全包含 - 三角形2完全在三角形1内部",
         {0,0,0, 4,0,0, 0,4,0},
         {1,1,0, 2,1,0, 1,2,0},
         true},

        {"线相交 - 两个三角形相交于一条线段",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0.5f,1, 1,0.5f,-1, 2,0.5f,0},
         true},

        {"边点接触 - 两个三角形的边相交于一点",
         {0,0,0, 2,0,0, 0,2,0},
         {1,-1,0, 1,1,1, 1,1,-1},
         true},

        {"远距离分离 - 完全分开",
         {0,0,0, 1,0,0, 0,1,0},
         {10,10,10, 11,10,10, 10,11,10},
         false},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    printf("Running %d test cases...\n\n", num_tests);

    // 创建 batch，所有数据初始化为安全值（远距离不相交三角形）
    TriPairBatch batch;
    // 使用一个安全的远距离三角形对，确保未使用的索引不会影响结果
    PQP_REAL safe_tri1[9] = {100,100,100, 101,100,100, 100,101,100};
    PQP_REAL safe_tri2[9] = {200,200,200, 201,200,200, 200,201,200};

    for (int i = 0; i < 16; ++i) {
        // 三角形1
        batch.p1x[i] = safe_tri1[0]; batch.p1y[i] = safe_tri1[1]; batch.p1z[i] = safe_tri1[2];
        batch.q1x[i] = safe_tri1[3]; batch.q1y[i] = safe_tri1[4]; batch.q1z[i] = safe_tri1[5];
        batch.r1x[i] = safe_tri1[6]; batch.r1y[i] = safe_tri1[7]; batch.r1z[i] = safe_tri1[8];
        // 三角形2
        batch.p2x[i] = safe_tri2[0]; batch.p2y[i] = safe_tri2[1]; batch.p2z[i] = safe_tri2[2];
        batch.q2x[i] = safe_tri2[3]; batch.q2y[i] = safe_tri2[4]; batch.q2z[i] = safe_tri2[5];
        batch.r2x[i] = safe_tri2[6]; batch.r2y[i] = safe_tri2[7]; batch.r2z[i] = safe_tri2[8];
    }

    // 将测试用例填充到 batch 的前 num_tests 个位置
    for (int i = 0; i < num_tests; ++i) {
        const TestCase& tc = tests[i];
        batch.p1x[i] = tc.tri1[0]; batch.p1y[i] = tc.tri1[1]; batch.p1z[i] = tc.tri1[2];
        batch.q1x[i] = tc.tri1[3]; batch.q1y[i] = tc.tri1[4]; batch.q1z[i] = tc.tri1[5];
        batch.r1x[i] = tc.tri1[6]; batch.r1y[i] = tc.tri1[7]; batch.r1z[i] = tc.tri1[8];
        batch.p2x[i] = tc.tri2[0]; batch.p2y[i] = tc.tri2[1]; batch.p2z[i] = tc.tri2[2];
        batch.q2x[i] = tc.tri2[3]; batch.q2y[i] = tc.tri2[4]; batch.q2z[i] = tc.tri2[5];
        batch.r2x[i] = tc.tri2[6]; batch.r2y[i] = tc.tri2[7]; batch.r2z[i] = tc.tri2[8];
    }

    bool batch_results[16];
    TriIntBatch(batch, batch_results);

    bool all_batch_pass = true;
    bool all_single_pass = true;

    for (int i = 0; i < num_tests; ++i) {
        const TestCase& tc = tests[i];
        bool batch_result = batch_results[i];
        bool batch_pass = (batch_result == tc.expected);

        // 单对接口验证
        PQP_REAL data1[9] = {
            batch.p1x[i], batch.p1y[i], batch.p1z[i],
            batch.q1x[i], batch.q1y[i], batch.q1z[i],
            batch.r1x[i], batch.r1y[i], batch.r1z[i]
        };
        PQP_REAL data2[9] = {
            batch.p2x[i], batch.p2y[i], batch.p2z[i],
            batch.q2x[i], batch.q2y[i], batch.q2z[i],
            batch.r2x[i], batch.r2y[i], batch.r2z[i]
        };
        bool single_result = TriInt(data1, data2);
        bool single_pass = (single_result == tc.expected);

        printf("Test %2d: %s\n", i+1, tc.description);
        printf("  Expected: %s\n", tc.expected ? "Intersect" : "No intersect");
        printf("  Batch:   %s %s\n", batch_result ? "Intersect" : "No intersect",
               batch_pass ? "[PASS]" : "[FAIL]");
        printf("  Single:  %s %s\n", single_result ? "Intersect" : "No intersect",
               single_pass ? "[PASS]" : "[FAIL]");
        printf("\n");

        if (!batch_pass) all_batch_pass = false;
        if (!single_pass) all_single_pass = false;
    }

    printf("========================================\n");
    printf("Overall result: ");
    if (all_batch_pass && all_single_pass) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
        if (!all_batch_pass) printf("  Batch interface failures detected.\n");
        if (!all_single_pass) printf("  Single interface failures detected.\n");
    }

    return 0;
}