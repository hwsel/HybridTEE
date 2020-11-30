#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_PRINT_DEFINED__
#define OCALL_PRINT_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print, (const char* str, float value));
#endif

sgx_status_t ecall_make_network(sgx_enclave_id_t eid, int* retval, int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches, int start_index);
sgx_status_t ecall_make_convolutional_layer(sgx_enclave_id_t eid, int* retval, int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, char* activation_s, int acti_length, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot);
sgx_status_t ecall_make_connected_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, int outputs, char* activation_s, int acti_length, int batch_normalize, int adam);
sgx_status_t ecall_make_softmax_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);
sgx_status_t ecall_make_cost_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, char* cost_type, int cost_size, float scale, float ratio, float noobject_scale, float thresh);
sgx_status_t ecall_make_maxpool_layer(sgx_enclave_id_t eid, int* retval, int batch, int h, int w, int c, int size, int stride, int padding);
sgx_status_t ecall_make_dropout_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, float probability, int w, int h, int c, float* net_prev_output, int prev_size, float* net_prev_delta, int delta_size);
sgx_status_t ecall_make_avgpool_layer(sgx_enclave_id_t eid, int* retval, int batch, int w, int h, int c);
sgx_status_t ecall_allocate_workspace(sgx_enclave_id_t eid, int* retval, int workspace_size);
sgx_status_t ecall_transfer_weights(sgx_enclave_id_t eid, int* retval, float* vec, int length, int layer_i, char type, int additional);
sgx_status_t ecall_net_output_return(sgx_enclave_id_t eid, int* retval, float* net_output, int length, uint8_t* tag_buffer, int tag_length);
sgx_status_t ecall_forward_network(sgx_enclave_id_t eid, int* retval, float* net_input, int l_inputs, uint8_t* tag_buffer, int l_tag, int net_train);
sgx_status_t ecall_attest_session_token(sgx_enclave_id_t eid, int* retval, int session_token, char* attestation_buffer, int attest_size, uint8_t* tag_buffer, int l_tag);
sgx_status_t ecall_decrypt(sgx_enclave_id_t eid, int* retval, uint32_t* encrypt_input, uint32_t input_length, uint32_t* plaintext_output, uint32_t output_length, uint8_t* key, uint32_t key_size);
sgx_status_t ecall_encrypt(sgx_enclave_id_t eid, int* retval, uint32_t* plaintext_input, uint32_t input_length, uint32_t* encrypted_output, uint32_t output_length, uint8_t* key, uint32_t key_size);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
