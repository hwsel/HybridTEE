#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

int ecall_make_network(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches, int start_index);
int ecall_make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, char* activation_s, int acti_length, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot);
int ecall_make_connected_layer(int batch, int inputs, int outputs, char* activation_s, int acti_length, int batch_normalize, int adam);
int ecall_make_softmax_layer(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);
int ecall_make_cost_layer(int batch, int inputs, char* cost_type, int cost_size, float scale, float ratio, float noobject_scale, float thresh);
int ecall_make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
int ecall_make_dropout_layer(int batch, int inputs, float probability, int w, int h, int c, float* net_prev_output, int prev_size, float* net_prev_delta, int delta_size);
int ecall_make_avgpool_layer(int batch, int w, int h, int c);
int ecall_allocate_workspace(int workspace_size);
int ecall_transfer_weights(float* vec, int length, int layer_i, char type, int additional);
int ecall_net_output_return(float* net_output, int length, uint8_t* tag_buffer, int tag_length);
int ecall_forward_network(float* net_input, int l_inputs, uint8_t* tag_buffer, int l_tag, int net_train);
int ecall_attest_session_token(int session_token, char* attestation_buffer, int attest_size, uint8_t* tag_buffer, int l_tag);
int ecall_decrypt(uint32_t* encrypt_input, uint32_t input_length, uint32_t* plaintext_output, uint32_t output_length, uint8_t* key, uint32_t key_size);
int ecall_encrypt(uint32_t* plaintext_input, uint32_t input_length, uint32_t* encrypted_output, uint32_t output_length, uint8_t* key, uint32_t key_size);

sgx_status_t SGX_CDECL ocall_print(const char* str, float value);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
