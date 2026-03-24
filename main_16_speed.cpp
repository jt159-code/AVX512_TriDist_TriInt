#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <chrono>   // 高精度计时

// ----- 手动声明来自 Tri_16.cpp 的内容 -----
typedef float PQP_REAL;

namespace tdbase {
    struct alignas(16) TriPairBatch {
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

// 测试用例结构
struct TestCase {
    const char* description;
    PQP_REAL tri1[9];   // p,q,r 各三个坐标
    PQP_REAL tri2[9];
    bool expected;      // true=相交, false=不相交
};

// 辅助函数：打印三角形顶点（用于调试）
void print_triangle(const PQP_REAL tri[9], const char* name) {
    printf("%s: (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f)\n",
           name,
           tri[0], tri[1], tri[2],
           tri[3], tri[4], tri[5],
           tri[6], tri[7], tri[8]);
}

int main() {
    // 定义测试用例（最多16个，这里使用12个）
    TestCase tests[] = {
        // 1. 完全分离
        {"分离 - 两个三角形在Z=0平面但不相交",
         {0,0,0, 1,0,0, 0,1,0},
         {2,2,0, 3,2,0, 2,3,0},
         false},

        // 2. 顶点在另一个三角形内（相交）
        {"顶点包含 - 三角形2的顶点(1,1,0)在三角形1内部",
         {0,0,0, 2,0,0, 0,2,0},
         {1,1,0, 3,1,0, 1,3,0},
         true},

        // 3. 边穿过另一个三角形（非共面相交）
        {"边穿过 - 三角形2的边(0.5,-0.5,0)-(0.5,1.5,0)穿过三角形1",
         {0,0,0, 2,0,0, 0,2,0},
         {0.5f,-0.5f,0, 0.5f,1.5f,0, 1.5f,0.5f,0},
         true},

        // 4. 共面相交通（共享重叠区域）
        {"共面相交 - 两个三角形在同一平面且有重叠",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0,0, 3,0,0, 1,2,0},
         true},

        // 5. 共面不相交
        {"共面不相交 - 同一平面但分离",
         {0,0,0, 1,0,0, 0,1,0},
         {2,0,0, 3,0,0, 2,1,0},
         false},

        // 6. 平行且分离（不同平面）
        {"平行分离 - Z=0 和 Z=1 平面，投影无重叠",
         {0,0,0, 1,0,0, 0,1,0},
         {0,0,1, 1,0,1, 0,1,1},
         false},

        // 7. 点接触（共享一个顶点）
        {"点接触 - 共享顶点(0,0,0)",
         {0,0,0, 1,0,0, 0,1,0},
         {0,0,0, 1,0,0.1f, 0,1,0.1f},
         true},

        // 8. 边重叠（共面且边部分重叠）
        {"边重叠 - 共享一条边的一部分",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0,0, 3,0,0, 1,2,0},
         true},

        // 9. 一个三角形完全包含另一个（共面）
        {"完全包含 - 三角形2完全在三角形1内部",
         {0,0,0, 4,0,0, 0,4,0},
         {1,1,0, 2,1,0, 1,2,0},
         true},

        // 10. 相交于一条线（非共面，但交线为线段）
        {"线相交 - 两个三角形相交于一条线段",
         {0,0,0, 2,0,0, 0,2,0},
         {1,0.5f,1, 1,0.5f,-1, 2,0.5f,0},
         true},

        // 11. 仅边接触（非共面，边与边相交于一点）
        {"边点接触 - 两个三角形的边相交于一点",
         {0,0,0, 2,0,0, 0,2,0},
         {1,-1,0, 1,1,1, 1,1,-1},
         true},

        // 12. 远距离分离（三维空间）
        {"远距离分离 - 完全分开",
         {0,0,0, 1,0,0, 0,1,0},
         {10,10,10, 11,10,10, 10,11,10},
         false},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    printf("Running %d test cases...\n\n", num_tests);

    // 用于批量测试的batch（16对）
    TriPairBatch batch;
    bool batch_results[16];
    bool all_batch_pass = true;
    bool all_single_pass = true;

    // 初始化batch为安全值（不相交的三角形对）
    // 使用远距离三角形填充未使用的索引
    PQP_REAL far_tri1[9] = {100,100,100, 101,100,100, 100,101,100};
    PQP_REAL far_tri2[9] = {200,200,200, 201,200,200, 200,201,200};
    for (int i = 0; i < 16; ++i) {
        // 复制远距离三角形到所有位置，稍后会覆盖测试用例
        for (int j = 0; j < 3; ++j) {
            batch.p1x[i] = far_tri1[j*3];
            batch.p1y[i] = far_tri1[j*3+1];
            batch.p1z[i] = far_tri1[j*3+2];
            batch.q1x[i] = far_tri1[j*3];
            batch.q1y[i] = far_tri1[j*3+1];
            batch.q1z[i] = far_tri1[j*3+2];
            batch.r1x[i] = far_tri1[j*3];
            batch.r1y[i] = far_tri1[j*3+1];
            batch.r1z[i] = far_tri1[j*3+2];
            batch.p2x[i] = far_tri2[j*3];
            batch.p2y[i] = far_tri2[j*3+1];
            batch.p2z[i] = far_tri2[j*3+2];
            batch.q2x[i] = far_tri2[j*3];
            batch.q2y[i] = far_tri2[j*3+1];
            batch.q2z[i] = far_tri2[j*3+2];
            batch.r2x[i] = far_tri2[j*3];
            batch.r2y[i] = far_tri2[j*3+1];
            batch.r2z[i] = far_tri2[j*3+2];
        }
    }

    // 将测试用例填充到batch的前num_tests个位置
    for (int i = 0; i < num_tests; ++i) {
        const TestCase& tc = tests[i];
        batch.p1x[i] = tc.tri1[0]; batch.p1y[i] = tc.tri1[1]; batch.p1z[i] = tc.tri1[2];
        batch.q1x[i] = tc.tri1[3]; batch.q1y[i] = tc.tri1[4]; batch.q1z[i] = tc.tri1[5];
        batch.r1x[i] = tc.tri1[6]; batch.r1y[i] = tc.tri1[7]; batch.r1z[i] = tc.tri1[8];
        batch.p2x[i] = tc.tri2[0]; batch.p2y[i] = tc.tri2[1]; batch.p2z[i] = tc.tri2[2];
        batch.q2x[i] = tc.tri2[3]; batch.q2y[i] = tc.tri2[4]; batch.q2z[i] = tc.tri2[5];
        batch.r2x[i] = tc.tri2[6]; batch.r2y[i] = tc.tri2[7]; batch.r2z[i] = tc.tri2[8];
    }

    // 调用批量检测（正确性验证）
    TriIntBatch(batch, batch_results);

    // 验证每个测试用例（正确性）
    for (int i = 0; i < num_tests; ++i) {
        const TestCase& tc = tests[i];
        bool batch_result = batch_results[i];
        bool batch_pass = (batch_result == tc.expected);

        // 单独调用单对接口进行交叉验证
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

        // 输出结果
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

    // 最终正确性报告
    printf("========================================\n");
    printf("Correctness result: ");
    if (all_batch_pass && all_single_pass) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
        if (!all_batch_pass) printf("  Batch interface failures detected.\n");
        if (!all_single_pass) printf("  Single interface failures detected.\n");
    }
    printf("\n");

    // ========== 性能测量 ==========
    const int REPEAT = 2000;   // 每个接口重复调用次数
    printf("Performance measurement: %d calls per interface per test case\n", REPEAT);
    printf("Results are average time per triangle pair (microseconds).\n\n");

    // 用于避免编译器优化
    volatile bool discard = false;

    // 对每个测试用例单独测速
    for (int i = 0; i < num_tests; ++i) {
        const TestCase& tc = tests[i];
        // 准备单对数据
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

        // ---- 单对接口测速 ----
        auto start_single = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPEAT; ++r) {
            bool res = TriInt(data1, data2);
            discard = discard || res;  // 防止优化
        }
        auto end_single = std::chrono::high_resolution_clock::now();
        double time_single_us = std::chrono::duration<double, std::micro>(end_single - start_single).count();
        double avg_single_us = time_single_us / REPEAT;  // 每对平均微秒

        // ---- 批量接口测速（每次调用处理 16 对）----
        // 注意：我们只测第 i 个测试用例，但批量接口同时处理 16 对，其中只有第 i 个是我们关心的。
        // 为了公平比较，我们计算平均每对的时间 = 总时间 / (REPEAT * 16)
        bool temp_results[16];
        auto start_batch = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPEAT; ++r) {
            TriIntBatch(batch, temp_results);
            discard = discard || temp_results[0];  // 随便取一个结果防止优化
        }
        auto end_batch = std::chrono::high_resolution_clock::now();
        double time_batch_us = std::chrono::duration<double, std::micro>(end_batch - start_batch).count();
        double avg_batch_per_pair_us = time_batch_us / (REPEAT * 16);  // 每对平均微秒

        printf("Test %2d: %s\n", i+1, tc.description);
        printf("  Single (per pair): %.3f us\n", avg_single_us);
        printf("  Batch  (per pair): %.3f us\n", avg_batch_per_pair_us);
        printf("\n");
    }

    // 可选：输出汇总（所有测试用例的平均）
    printf("========================================\n");
    printf("Performance summary: see per-case details above.\n");
    return 0;
}