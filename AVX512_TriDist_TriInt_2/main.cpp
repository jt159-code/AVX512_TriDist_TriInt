#include "TriDist.h"
#include <cstdio>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <locale.h>

// 跨平台 CPU 特性检测
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <x86intrin.h> // 用于 __builtin_cpu_supports
#endif

namespace tdbase
{
    bool TriInt(const PQP_REAL *data1, const PQP_REAL *data2);
}

void check_cpu_features()
{
    printf("\n=== CPU Feature Detection ===\n");

#ifdef _WIN32
    // Windows 版本：使用 __cpuid / __cpuidex
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);

    if (cpuInfo[2] & (1 << 28))
        printf("AVX: Supported\n");
    if (cpuInfo[2] & (1 << 5))
        printf("AVX2: Supported\n");

    int cpuInfo2[4];
    __cpuidex(cpuInfo2, 7, 0);
    if (cpuInfo2[1] & (1 << 16))
        printf("AVX512F: Supported\n");
#else
    // Linux / GCC 版本：使用 __builtin_cpu_supports
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx"))
        printf("AVX: Supported\n");
    if (__builtin_cpu_supports("avx2"))
        printf("AVX2: Supported\n");
    if (__builtin_cpu_supports("avx512f"))
        printf("AVX512F: Supported\n");
#endif

        // 编译器宏定义
#ifdef __AVX2__
    printf("Compiler: AVX2 enabled\n");
#endif
#ifdef __AVX512F__
    printf("Compiler: AVX512F enabled\n");
#endif
#ifdef __AVX512DQ__
    printf("Compiler: AVX512DQ enabled\n");
#endif

    printf("NOTE: This program does not use any SIMD instructions directly.\n");
    printf("      The above features are for reference only.\n");
}

int main()
{
    // 设置区域（Linux 下使用环境变量）
    setlocale(LC_ALL, "");

    using namespace std::chrono;

    printf("========================================\n");
    printf("Pure C++ Optimized Version - Batch Size n=16 + SoA\n");
    printf("16 triangle pairs, perfectly filling AVX512 registers\n");
    printf("Without using any SIMD instructions\n");
    printf("========================================\n\n");

    // Prepare test data - 16 triangle pairs
    PQP_REAL s_batch[16][3][3];
    PQP_REAL t_batch[16][3][3];
    PQP_REAL p_batch[16][3];
    PQP_REAL q_batch[16][3];
    PQP_REAL dist[16];

    // Initialize 16 triangle pairs with different offsets
    for (int i = 0; i < 16; i++)
    {
        float z_offset = i * 0.1f; // Gradually separate triangles

        // Triangle S: in z=0 plane
        s_batch[i][0][0] = 0;
        s_batch[i][0][1] = 0;
        s_batch[i][0][2] = 0;
        s_batch[i][1][0] = 1;
        s_batch[i][1][1] = 0;
        s_batch[i][1][2] = 0;
        s_batch[i][2][0] = 0;
        s_batch[i][2][1] = 1;
        s_batch[i][2][2] = 0;

        // Triangle T: in z=1 plane with offset
        t_batch[i][0][0] = 0;
        t_batch[i][0][1] = 0;
        t_batch[i][0][2] = 1 + z_offset;
        t_batch[i][1][0] = 1;
        t_batch[i][1][1] = 0;
        t_batch[i][1][2] = 1 + z_offset;
        t_batch[i][2][0] = 0;
        t_batch[i][2][1] = 1;
        t_batch[i][2][2] = 1 + z_offset;
    }

    // ========== Test batch computation ==========
    printf("=== Testing Batch Processing n=16 ===\n");

    auto start = high_resolution_clock::now();
    TriDistBatch16(p_batch, q_batch, dist, s_batch, t_batch);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Output first 8 results
    printf("First 8 results:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Pair %d: distance = %f, P=(%f,%f,%f), Q=(%f,%f,%f)\n",
               i, dist[i],
               p_batch[i][0], p_batch[i][1], p_batch[i][2],
               q_batch[i][0], q_batch[i][1], q_batch[i][2]);
    }
    printf("... (16 pairs total)\n");
    printf("Batch execution time: %lld microseconds\n\n", duration.count());

    // ========== Performance comparison test ==========
    const int NUM_BATCHES = 10000; // 10k batches, 16 pairs each, total 160k pairs
    const int TOTAL_PAIRS = NUM_BATCHES * 16;

    printf("=== Performance Comparison (%d triangle pairs) ===\n", TOTAL_PAIRS);

    // Prepare random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    std::vector<PQP_REAL> all_s(TOTAL_PAIRS * 3 * 3);
    std::vector<PQP_REAL> all_t(TOTAL_PAIRS * 3 * 3);

    for (int i = 0; i < TOTAL_PAIRS * 9; i++)
    {
        all_s[i] = dis(gen);
        all_t[i] = dis(gen);
    }

    // Test 1: Scalar version (one by one)
    printf("Running scalar version...\n");
    auto scalar_start = high_resolution_clock::now();
    for (int i = 0; i < TOTAL_PAIRS; i++)
    {
        PQP_REAL p[3], q[3];
        const PQP_REAL(*s)[3] = (const PQP_REAL(*)[3]) & all_s[i * 9];
        const PQP_REAL(*t)[3] = (const PQP_REAL(*)[3]) & all_t[i * 9];
        PQP_REAL d = TriDist(p, q, s, t);
    }
    auto scalar_end = high_resolution_clock::now();
    auto scalar_duration = duration_cast<microseconds>(scalar_end - scalar_start);

    // Test 2: 16-pair batch version
    printf("Running 16-pair batch version...\n");
    auto batch16_start = high_resolution_clock::now();
    for (int i = 0; i < NUM_BATCHES; i++)
    {
        const PQP_REAL(*s_batch_ptr)[3][3] = (const PQP_REAL(*)[3][3]) & all_s[i * 16 * 9];
        const PQP_REAL(*t_batch_ptr)[3][3] = (const PQP_REAL(*)[3][3]) & all_t[i * 16 * 9];
        PQP_REAL p_batch_local[16][3], q_batch_local[16][3], dist_local[16];

        TriDistBatch16(p_batch_local, q_batch_local, dist_local,
                       s_batch_ptr, t_batch_ptr);
    }
    auto batch16_end = high_resolution_clock::now();
    auto batch16_duration = duration_cast<microseconds>(batch16_end - batch16_start);

    // Test 3: 8-pair batch version (for comparison)
    printf("Running 8-pair batch version...\n");
    const int NUM_BATCHES8 = TOTAL_PAIRS / 8;
    auto batch8_start = high_resolution_clock::now();
    for (int i = 0; i < NUM_BATCHES8; i++)
    {
        const PQP_REAL(*s_batch_ptr)[3][3] = (const PQP_REAL(*)[3][3]) & all_s[i * 8 * 9];
        const PQP_REAL(*t_batch_ptr)[3][3] = (const PQP_REAL(*)[3][3]) & all_t[i * 8 * 9];
        PQP_REAL p_batch_local[8][3], q_batch_local[8][3], dist_local[8];

        TriDistBatch8(p_batch_local, q_batch_local, dist_local,
                      s_batch_ptr, t_batch_ptr);
    }
    auto batch8_end = high_resolution_clock::now();
    auto batch8_duration = duration_cast<microseconds>(batch8_end - batch8_start);

    // Output results
    printf("\n=== Performance Statistics ===\n");
    printf("Scalar version: %lld microseconds (%.3f us per pair)\n",
           scalar_duration.count(), scalar_duration.count() / (double)TOTAL_PAIRS);
    printf("8-pair batch: %lld microseconds (%.3f us per pair)\n",
           batch8_duration.count(), batch8_duration.count() / (double)TOTAL_PAIRS);
    printf("16-pair batch: %lld microseconds (%.3f us per pair)\n",
           batch16_duration.count(), batch16_duration.count() / (double)TOTAL_PAIRS);

    printf("\n=== Speedup ===\n");
    printf("8-pair vs Scalar: %.2fx\n",
           scalar_duration.count() / (double)batch8_duration.count());
    printf("16-pair vs Scalar: %.2fx\n",
           scalar_duration.count() / (double)batch16_duration.count());
    printf("16-pair vs 8-pair: %.2fx\n",
           batch8_duration.count() / (double)batch16_duration.count());

    // Average time per batch
    printf("\n=== Average Time Per Batch ===\n");
    printf("8-pair batch: %.3f us\n", batch8_duration.count() / (double)NUM_BATCHES8);
    printf("16-pair batch: %.3f us\n", batch16_duration.count() / (double)NUM_BATCHES);

    printf("\nPress Enter to exit...");
    getchar();

    check_cpu_features();

    return 0;
}