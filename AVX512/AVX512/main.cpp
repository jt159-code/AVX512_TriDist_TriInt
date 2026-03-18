#include "TriDist.h"
#include <cstdio>

// 声明TriInt函数，它在tdbase命名空间中
namespace tdbase {
    bool TriInt(const PQP_REAL* data1, const PQP_REAL* data2);
}

int main()
{
    PQP_REAL tri1[3][3] = { {0,0,0}, {1,0,0}, {0,1,0} };
    PQP_REAL tri2[3][3] = { {0,0,1}, {1,0,1}, {0,1,1} };
    PQP_REAL p[3], q[3];
    PQP_REAL dist = TriDist(p, q, tri1, tri2);

    printf("最近距离 = %f\n", dist);
    printf("最近点 on tri1: (%f, %f, %f)\n", p[0], p[1], p[2]);
    printf("最近点 on tri2: (%f, %f, %f)\n", q[0], q[1], q[2]);

    // 测试相交性
    bool intersect = tdbase::TriInt(&tri1[0][0], &tri2[0][0]);
    printf("三角形相交: %s\n", intersect ? "是" : "否");

    // 额外测试一对相交的三角形（例如将tri2移到与tri1共面并重叠）
    PQP_REAL tri3[3][3] = { {0.2,0.2,0}, {0.8,0.2,0}, {0.2,0.8,0} }; // 在tri1内部
    intersect = tdbase::TriInt(&tri1[0][0], &tri3[0][0]);
    printf("三角形 tri1 和 tri3 相交: %s\n", intersect ? "是" : "否");

    return 0;
}