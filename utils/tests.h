#ifndef TESTS_H_
#define TESTS_H_

#include "test_utils.h"
#include "../two_layer_net.h"

int checkErrors(const arma::mat& Seq, const arma::mat& Par, 
				std::ofstream& ofs, std::vector<double>& errors);

int checkNNErrors(TwoLayerNet& seq_nn, TwoLayerNet& par_nn, 
						std::string filename);

void BenchmarkGEMM();

void BenchmarkGEMM_no_overwrite();

void BenchmarkGEMM_no_overwrite_transposeB();

void BenchmarkGEMM_no_tB();

void BenchmarkGEMM_no_na_tA();

void test_sigmoid_GPU();

#endif
