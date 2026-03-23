#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
//GPU 图表 
typedef float PQP_REAL;

extern void TriDistBatch16(
    PQP_REAL p_batch[16][3],
    PQP_REAL q_batch[16][3],
    PQP_REAL dist[16],
    const PQP_REAL s_batch[16][3][3],
    const PQP_REAL t_batch[16][3][3]);

// 读取 OFF 文件
struct Vertex
{
    float x, y, z;
};

struct Triangle
{
    int v0, v1, v2;
};

static bool loadOffFile(const char *filename, std::vector<Vertex> &vertices, std::vector<Triangle> &faces)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string header;
    file >> header;

    if (header != "OFF")
    {
        std::cerr << "Not a valid OFF file: " << filename << std::endl;
        return false;
    }

    int numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;

    std::cout << "OFF file: " << numVertices << " vertices, " << numFaces << " faces" << std::endl;

    // 读取顶点
    vertices.resize(numVertices);
    for (int i = 0; i < numVertices; i++)
    {
        file >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    }

    // 读取面
    faces.resize(numFaces);
    for (int i = 0; i < numFaces; i++)
    {
        int nverts;
        file >> nverts;
        if (nverts == 3)
        {
            file >> faces[i].v0 >> faces[i].v1 >> faces[i].v2;
        }
        else
        {
            // 非三角形，跳过
            int tmp;
            for (int j = 0; j < nverts; j++)
            {
                file >> tmp;
            }
            faces[i].v0 = faces[i].v1 = faces[i].v2 = 0;
        }
    }

    file.close();
    return true;
}

int main(int argc, char *argv[])
{
    const char *offFile = "input.off";
    if (argc > 1)
    {
        offFile = argv[1];
    }

    std::cout << "=========================================\n";
    std::cout << "  TriDist Batch Test (AVX512)\n";
    std::cout << "  Loading triangles from OFF file\n";
    std::cout << "=========================================\n\n";

    // 读取 OFF 文件
    std::vector<Vertex> vertices;
    std::vector<Triangle> faces;
    if (!loadOffFile(offFile, vertices, faces))
    {
        return 1;
    }

    // 随机数种子
    srand((unsigned int)time(nullptr));

    const int NUM_TESTS = 2000; // 测试批次数量
    const int BATCH_SIZE = 16;  // 每批处理16对三角形
    const int ITERATIONS = 100; // 重复测试次数

    // 只使用有效的三角形
    std::vector<Triangle> validFaces;
    for (size_t i = 0; i < faces.size(); i++)
    {
        if (faces[i].v0 >= 0 && faces[i].v0 < (int)vertices.size() &&
            faces[i].v1 >= 0 && faces[i].v1 < (int)vertices.size() &&
            faces[i].v2 >= 0 && faces[i].v2 < (int)vertices.size())
        {
            validFaces.push_back(faces[i]);
        }
    }

    std::cout << "Valid triangles: " << validFaces.size() << "\n";
    std::cout << "Testing with " << NUM_TESTS << " x " << BATCH_SIZE << " = " << (NUM_TESTS * BATCH_SIZE) << " triangle pairs\n";
    std::cout << "Performance test: " << ITERATIONS << " iterations\n\n";

    // 准备测试数据
    PQP_REAL s_batch[BATCH_SIZE][3][3];
    PQP_REAL t_batch[BATCH_SIZE][3][3];
    PQP_REAL p_batch[BATCH_SIZE][3], q_batch[BATCH_SIZE][3], dist[BATCH_SIZE];

    std::vector<PQP_REAL> all_s_data(NUM_TESTS * BATCH_SIZE * 9);
    std::vector<PQP_REAL> all_t_data(NUM_TESTS * BATCH_SIZE * 9);

    std::cout << "--- Generating random triangle pairs ---\n";

    for (int test = 0; test < NUM_TESTS; test++)
    {
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            // 随机选择两个不同的三角形
            int idxS = rand() % validFaces.size();
            int idxT = rand() % validFaces.size();

            const Triangle &triS = validFaces[idxS];
            const Triangle &triT = validFaces[idxT];

            // S 三角形
            s_batch[i][0][0] = vertices[triS.v0].x;
            s_batch[i][0][1] = vertices[triS.v0].y;
            s_batch[i][0][2] = vertices[triS.v0].z;

            s_batch[i][1][0] = vertices[triS.v1].x;
            s_batch[i][1][1] = vertices[triS.v1].y;
            s_batch[i][1][2] = vertices[triS.v1].z;

            s_batch[i][2][0] = vertices[triS.v2].x;
            s_batch[i][2][1] = vertices[triS.v2].y;
            s_batch[i][2][2] = vertices[triS.v2].z;

            // T 三角形
            t_batch[i][0][0] = vertices[triT.v0].x;
            t_batch[i][0][1] = vertices[triT.v0].y;
            t_batch[i][0][2] = vertices[triT.v0].z;

            t_batch[i][1][0] = vertices[triT.v1].x;
            t_batch[i][1][1] = vertices[triT.v1].y;
            t_batch[i][1][2] = vertices[triT.v1].z;

            t_batch[i][2][0] = vertices[triT.v2].x;
            t_batch[i][2][1] = vertices[triT.v2].y;
            t_batch[i][2][2] = vertices[triT.v2].z;

            // 保存数据
            int base = (test * BATCH_SIZE + i) * 9;
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    all_s_data[base + j * 3 + k] = s_batch[i][j][k];
                    all_t_data[base + j * 3 + k] = t_batch[i][j][k];
                }
            }
        }

        if ((test + 1) % 100 == 0)
        {
            std::cout << "  Generated " << (test + 1) << " / " << NUM_TESTS << " batches\n";
        }
    }

    std::cout << "\n--- Single Batch Verification ---\n";

    // // 验证单个 batch
    // for (int i = 0; i < BATCH_SIZE; i++)
    // {
    //     for (int j = 0; j < 3; j++)
    //     {
    //         for (int k = 0; k < 3; k++)
    //         {
    //             s_batch[i][j][k] = all_s_data[i * 9 + j * 3 + k];
    //             t_batch[i][j][k] = all_t_data[i * 9 + j * 3 + k];
    //         }
    //     }
    // }

    // auto start = std::chrono::high_resolution_clock::now();
    // TriDistBatch16(p_batch, q_batch, dist, s_batch, t_batch);
    // auto end = std::chrono::high_resolution_clock::now();
    // double singleTime = std::chrono::duration<double, std::milli>(end - start).count();

    // std::cout << "Single batch time: " << std::fixed << std::setprecision(3) << singleTime << " ms\n";
    // std::cout << "Sample distances:\n";
    // for (int i = 0; i < 4; i++)
    // {
    //     std::cout << "  Pair[" << i << "]: dist = " << dist[i] << "\n";
    // }
    // std::cout << "\n";

    // 性能测试
    std::cout << "--- Performance Test ---\n";

    std::vector<const PQP_REAL *> s_ptrs(NUM_TESTS);
    std::vector<const PQP_REAL *> t_ptrs(NUM_TESTS);
    for (int test = 0; test < NUM_TESTS; test++)
    {
        s_ptrs[test] = &all_s_data[test * BATCH_SIZE * 9];
        t_ptrs[test] = &all_t_data[test * BATCH_SIZE * 9];
    }

    volatile double dummy = 0.0; // 防止编译器优化

    auto totalStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        for (int test = 0; test < NUM_TESTS; test++)
        {
            const PQP_REAL *s_data = s_ptrs[test];
            const PQP_REAL *t_data = t_ptrs[test];

            for (int i = 0; i < BATCH_SIZE; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        s_batch[i][j][k] = s_data[i * 9 + j * 3 + k];
                        t_batch[i][j][k] = t_data[i * 9 + j * 3 + k];
                    }
                }
            }

            TriDistBatch16(p_batch, q_batch, dist, s_batch, t_batch);
            dummy += dist[0];
        }
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    int totalPairs = NUM_TESTS * BATCH_SIZE * ITERATIONS;

    std::cout << "Total time: " << std::fixed << std::setprecision(3) << totalMs << " ms\n";
    std::cout << "Total triangle pairs tested: " << totalPairs << "\n";
    std::cout << "\n";
    std::cout << "Performance summary:\n";
    std::cout << "  Time per " << NUM_TESTS << " batches (" << (NUM_TESTS * BATCH_SIZE) << " pairs): "
              << std::fixed << std::setprecision(3) << (totalMs / ITERATIONS) << " ms\n";
    std::cout << "  Time per batch (16 pairs): "
              << std::fixed << std::setprecision(6) << (totalMs / ITERATIONS / NUM_TESTS) << " ms\n";
    std::cout << "  Time per pair: "
              << std::fixed << std::setprecision(6) << (totalMs / totalPairs * 1000.0) << " us\n";
    std::cout << "  Throughput: "
              << std::fixed << std::setprecision(2) << (totalPairs / (totalMs / 1000.0) / 1000000.0) << " M pairs/sec\n";

    std::cout << "\n=========================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "=========================================\n";

    return 0;
}