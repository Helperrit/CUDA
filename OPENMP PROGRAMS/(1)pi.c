#include <stdio.h>
#include <omp.h>

int main() {
    long num_steps = 1000000; // Number of intervals
    double step = 1.0 / (double)num_steps;
    double pi = 0.0;

    #pragma omp parallel
    {
        double sum = 0.0;
        int i;
        #pragma omp for
        for (i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        // Use a critical section to update the global variable pi
        #pragma omp critical
        {
            pi += sum * step;
        }
    }

    printf("Calculated value of PI: %.15f\n", pi);
    return 0;
}
