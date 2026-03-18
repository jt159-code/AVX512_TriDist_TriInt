

#include "TriDist.h"
#include <cstdio>

namespace tdbase
{
    bool TriInt(const PQP_REAL *data1, const PQP_REAL *data2);
}

int main()
{
    PQP_REAL tri1[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    PQP_REAL tri2[3][3] = {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    PQP_REAL p[3], q[3];
    PQP_REAL dist = TriDist(p, q, tri1, tri2);

    printf("Distance = %f\n", dist);
    printf("Closest point on tri1: (%f, %f, %f)\n", p[0], p[1], p[2]);
    printf("Closest point on tri2: (%f, %f, %f)\n", q[0], q[1], q[2]);

    // Test intersection
    bool intersect = tdbase::TriInt(&tri1[0][0], &tri2[0][0]);
    printf("Triangles intersect: %s\n", intersect ? "Yes" : "No");

    // Test overlapping triangles (tri3 is inside tri1)
    PQP_REAL tri3[3][3] = {{0.2, 0.2, 0}, {0.8, 0.2, 0}, {0.2, 0.8, 0}};
    intersect = tdbase::TriInt(&tri1[0][0], &tri3[0][0]);
    printf("Tri1 and Tri3 intersect: %s\n", intersect ? "Yes" : "No");

    // Wait for user input to keep window open
    printf("\nPress Enter to exit...");
    getchar();

    return 0;
}