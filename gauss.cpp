#include <iostream>
#include <vector>
#include "omp.h"

template <typename T>
int findCoeffMtrxRowWithMaxColumnElement(
    std::vector<std::vector<T>>& extended_matrix,
    int column_index,
    int coeff_matrix_size) {
  T max = std::abs(extended_matrix[column_index][column_index]);
  int row_index_with_max_clmn_elem = column_index;
  #pragma omp parallel
  {
    T thread_max = max;
    T thread_row_index_with_max_clmn_elem = row_index_with_max_clmn_elem;

    #pragma omp for
    for(int i = column_index + 1; i < coeff_matrix_size; i++) {
        T current_element = std::abs(extended_matrix[i][column_index]);
        if(current_element > thread_max) {
            thread_max = current_element;
            thread_row_index_with_max_clmn_elem = i;
        }
    }

    #pragma omp critical
    {
        if(max < thread_max) {
            max = thread_max;
            row_index_with_max_clmn_elem = thread_row_index_with_max_clmn_elem;
        }
    }
  }
  return row_index_with_max_clmn_elem;
}

template <typename T>
void triangulation(std::vector<std::vector<T>>& extended_matrix,
                   int coeff_matrix_size) {
    if(coeff_matrix_size == 0) {
        return;
    }
    
    const int extended_matrix_columns_number = coeff_matrix_size + 1;
    for(int i = 0; i < coeff_matrix_size - 1; i++) {
        unsigned row_index_with_max_clmn_elem = findCoeffMtrxRowWithMaxColumnElement(
            extended_matrix, i, coeff_matrix_size);
        if(row_index_with_max_clmn_elem != i) {
            std::swap(extended_matrix[i], extended_matrix[row_index_with_max_clmn_elem]);
        }

        #pragma omp parallel for
        for(int j = i + 1; j < coeff_matrix_size; j++) {
            T multiplier = -extended_matrix[j][i] / extended_matrix[i][i];

            for(int k = i; k < coeff_matrix_size; k++) {
                extended_matrix[j][k] += multiplier * extended_matrix[i][k];
            }
        }
    }
}


template <typename T>
std::vector<T> gaussSolving(std::vector<std::vector<T>>& coeff_matrix,
                             std::vector<T>& free_coeff_vector,
                             int coeff_matrix_size) {
    std::vector<T> solution(coeff_matrix_size);

    for(int i = 0; i < coeff_matrix_size; i++) {
        coeff_matrix[i].push_back(free_coeff_vector[i]);
    }
    triangulation(coeff_matrix, coeff_matrix_size);

    for(int i = coeff_matrix_size - 1; i >= 0; i--) {
        if(std::abs(coeff_matrix[i][i]) < 0.0001) {
            throw std::runtime_error("No solution");
        }
        solution[i] = coeff_matrix[i][coeff_matrix_size] / coeff_matrix[i][i];

        for(int j = 0; j < i; j++) {
            coeff_matrix[j][coeff_matrix_size] -= coeff_matrix[j][i] * solution[i];
        }
    }
    
    return solution;
}




int main(int argc, char** argv) {

    int var_count = std::atoi(argv[1]);

    std::vector<std::vector<double>> coeff_matrix(var_count);
    for(int i = 0; i < var_count; i++) {
        coeff_matrix[i].resize(var_count);
        for(int j = 0; j < var_count; j++) {
            coeff_matrix[i][j] = rand();
        }
    }
    std::vector<double> free_coeff(var_count);
    for(int i = 0; i < var_count; i++) {
        free_coeff[i] = rand();
    }

    double start_calc_time = omp_get_wtime();
    std::vector<double> solution = gaussSolving(coeff_matrix,
                                                free_coeff,
                                                var_count);
    
    double end_calc_time = omp_get_wtime();
    
    std::cout << "Solution[" << solution[0] << std::endl
        << solution[1] << std::endl
        << "..." << std::endl
        << solution[solution.size() - 1] << "]"
        << std::endl;

    std::cout << std::endl << "Calculation time: " << end_calc_time - start_calc_time
        << " seconds" << std::endl;

    return 0;
}