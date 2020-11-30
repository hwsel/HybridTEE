#ifndef ENCLAVE_TA
#define ENCLAVE_TA
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

void binarize_cpu_TA(float *input, int n, float *binary);
void binarize_weights_TA(float *weights, int n, int size, float *binary);
void swap_binary_TA(layer_TA *l);
void add_bias_TA(float *output, float *biases, int batch, int n, int size);
void scale_bias_TA(float *output, float *scales, int batch, int n, int size);
void forward_batchnorm_layer_enclave(layer_TA l, network_TA net);
void forward_connected_layer_enclave(layer_TA l, network_TA net);
void forward_softmax_layer_enclave(const layer_TA l, network_TA net);
COST_TYPE_TA get_cost_type_TA(char *s);
void forward_cost_layer_enclave(layer_TA l, network_TA net);
void forward_maxpool_layer_enclave(const layer_TA l, network_TA net);
void forward_dropout_layer_enclave(layer_TA l, network_TA net);
void aes_cbc_TA(char* xcrypt, float* gradient, int org_len);
void transpose_matrix_TA(float *a, int rows, int cols);
void calc_network_cost_enclave();

#endif
