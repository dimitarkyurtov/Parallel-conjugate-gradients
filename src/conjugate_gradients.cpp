#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <numeric>
#include <omp.h>

#define WORKLOAD_MEDIUM

#define NUM_THREADS 2

bool read_matrix_from_file(const char* filename, double** matrix_out, size_t* num_rows_out, size_t* num_cols_out)
{
    double* matrix;
    size_t num_rows;
    size_t num_cols;

    FILE* file = fopen(filename, "rb");
    if (file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



bool write_matrix_to_file(const char* filename, const double* matrix, size_t num_rows, size_t num_cols)
{
    FILE* file = fopen(filename, "wb");
    if (file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}

double dot_parallel(const double* x, const double* y, size_t size)
{
    double result = 0.0;

    #pragma omp parallel for schedule(static) reduction (+:result)
    for (int i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }

    return result;
}


void axpby_parallel(double alpha, const double* x, double beta, double* y, size_t size)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv_parallel(double alpha, const double* A, const double* x, double beta, double* y, size_t num_rows, size_t num_cols)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for (size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}

void conjugate_gradients(const double* A, const double* b, double* x, size_t size, int max_iters, double rel_error)
{
    double alpha, beta, bb, rr, rr_new;
    double* r = new double[size];
    double* p = new double[size];
    double* Ap = new double[size];
    int num_iters;

    memset(x, 0.0, size);
    std::copy(b, b + size, r);
    std::copy(b, b + size, p);

    // r = b
    // p = b

    bb = dot_parallel(b, b, size); // bb = b.b
    rr = bb; // rr = b.b
    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv_parallel(1.0, A, p, 0.0, Ap, size, size); // Ap = A*p
        alpha = rr / dot_parallel(p, Ap, size); //  alpha = rT*r/pTAp
        #pragma omp parallel num_threads(2)
        {
            size_t id{ static_cast<size_t>(omp_get_thread_num()) };

            if (id == 0)
            {
                axpby_parallel(alpha, p, 1.0, x, size); // x = alpha*p
            }
            else
            {
                axpby_parallel(-alpha, Ap, 1.0, r, size); // r = -alpha*Ap
            }
        }
        rr_new = dot_parallel(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if (std::sqrt(rr / bb) < rel_error) { break; }
        axpby_parallel(1.0, r, beta, p, size); // p = r + beta*p
    }
   
    

    delete[] r;
    delete[] p;
    delete[] Ap;

    if (num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}





int main(int argc, char** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");


#ifdef WORKLOAD_LOW
    const char* input_file_matrix = "matrix_low.bin";
    const char* input_file_rhs = "rhs_low.bin";
    const char* output_file_sol = "sol_low.bin";
#endif // WORKLOAD_LOW

#ifdef WORKLOAD_MEDIUM
    const char* input_file_matrix = "matrix_medium.bin";
    const char* input_file_rhs = "rhs_medium.bin";
    const char* output_file_sol = "sol_medium.bin";
#endif // WORKLOAD_MEDIUM

#ifdef WORKLOAD_HEAVY
    const char* input_file_matrix = "matrix_heavy.bin";
    const char* input_file_rhs = "rhs_heavy.bin";
    const char* output_file_sol = "sol_heavy.bin";
#endif // WORKLOAD_HEAVY

    int max_iters = 1000;
    double rel_error = 1e-9;

    if (argc > 1) input_file_matrix = argv[1];
    if (argc > 2) input_file_rhs = argv[2];
    if (argc > 3) output_file_sol = argv[3];
    if (argc > 4) max_iters = atoi(argv[4]);
    if (argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");



    double* matrix;
    double* rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if (!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if (!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");

        if (matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if (rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if (rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;
    }

    printf("Solving the system ...\n");
    double* sol = new double[size];

    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;
    typedef std::chrono::high_resolution_clock clock;

    auto start = clock::now();
    
    omp_set_num_threads(NUM_THREADS);
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    
    auto end = clock::now();
    printf("Time for completion: %d seconds\n", duration_cast<std::chrono::seconds>(end - start).count());
    printf("Done\n");
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if (!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    return 0;
}