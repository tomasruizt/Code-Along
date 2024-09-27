#include <stdio.h>

void mult(int *a, int *b, int *c, int n) {
    int i = 0;
    for (i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void print_vector(int *a, int n) {
    printf("[ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("]");
    printf("\n");
}