#include <stdio.h>
#include "operations.h"

int main() {
    int n = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {1, 2, 3, 4, 5};
    int c[n];
    mult(a, b, c, n);

    printf("a = ");
    print_vector(a, n);

    printf("b = ");
    print_vector(b, n);

    printf("c = ");
    print_vector(c, n);
    printf("\n");
    return 0;
}