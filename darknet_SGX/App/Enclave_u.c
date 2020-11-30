#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_make_network_t {
	int ms_retval;
	int ms_n;
	float ms_learning_rate;
	float ms_momentum;
	float ms_decay;
	int ms_time_steps;
	int ms_notruth;
	int ms_batch;
	int ms_subdivisions;
	int ms_random;
	int ms_adam;
	float ms_B1;
	float ms_B2;
	float ms_eps;
	int ms_h;
	int ms_w;
	int ms_c;
	int ms_inputs;
	int ms_max_crop;
	int ms_min_crop;
	float ms_max_ratio;
	float ms_min_ratio;
	int ms_center;
	float ms_clip;
	float ms_angle;
	float ms_aspect;
	float ms_saturation;
	float ms_exposure;
	float ms_hue;
	int ms_burn_in;
	float ms_power;
	int ms_max_batches;
	int ms_start_index;
} ms_ecall_make_network_t;

typedef struct ms_ecall_make_convolutional_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_h;
	int ms_w;
	int ms_c;
	int ms_n;
	int ms_groups;
	int ms_size;
	int ms_stride;
	int ms_padding;
	char* ms_activation_s;
	int ms_acti_length;
	int ms_batch_normalize;
	int ms_binary;
	int ms_xnor;
	int ms_adam;
	int ms_flipped;
	float ms_dot;
} ms_ecall_make_convolutional_layer_t;

typedef struct ms_ecall_make_connected_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_inputs;
	int ms_outputs;
	char* ms_activation_s;
	int ms_acti_length;
	int ms_batch_normalize;
	int ms_adam;
} ms_ecall_make_connected_layer_t;

typedef struct ms_ecall_make_softmax_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_inputs;
	int ms_groups;
	float ms_temperature;
	int ms_w;
	int ms_h;
	int ms_c;
	int ms_spatial;
	int ms_noloss;
} ms_ecall_make_softmax_layer_t;

typedef struct ms_ecall_make_cost_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_inputs;
	char* ms_cost_type;
	int ms_cost_size;
	float ms_scale;
	float ms_ratio;
	float ms_noobject_scale;
	float ms_thresh;
} ms_ecall_make_cost_layer_t;

typedef struct ms_ecall_make_maxpool_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_h;
	int ms_w;
	int ms_c;
	int ms_size;
	int ms_stride;
	int ms_padding;
} ms_ecall_make_maxpool_layer_t;

typedef struct ms_ecall_make_dropout_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_inputs;
	float ms_probability;
	int ms_w;
	int ms_h;
	int ms_c;
	float* ms_net_prev_output;
	int ms_prev_size;
	float* ms_net_prev_delta;
	int ms_delta_size;
} ms_ecall_make_dropout_layer_t;

typedef struct ms_ecall_make_avgpool_layer_t {
	int ms_retval;
	int ms_batch;
	int ms_w;
	int ms_h;
	int ms_c;
} ms_ecall_make_avgpool_layer_t;

typedef struct ms_ecall_allocate_workspace_t {
	int ms_retval;
	int ms_workspace_size;
} ms_ecall_allocate_workspace_t;

typedef struct ms_ecall_transfer_weights_t {
	int ms_retval;
	float* ms_vec;
	int ms_length;
	int ms_layer_i;
	char ms_type;
	int ms_additional;
} ms_ecall_transfer_weights_t;

typedef struct ms_ecall_net_output_return_t {
	int ms_retval;
	float* ms_net_output;
	int ms_length;
	uint8_t* ms_tag_buffer;
	int ms_tag_length;
} ms_ecall_net_output_return_t;

typedef struct ms_ecall_forward_network_t {
	int ms_retval;
	float* ms_net_input;
	int ms_l_inputs;
	uint8_t* ms_tag_buffer;
	int ms_l_tag;
	int ms_net_train;
} ms_ecall_forward_network_t;

typedef struct ms_ecall_attest_session_token_t {
	int ms_retval;
	int ms_session_token;
	char* ms_attestation_buffer;
	int ms_attest_size;
	uint8_t* ms_tag_buffer;
	int ms_l_tag;
} ms_ecall_attest_session_token_t;

typedef struct ms_ecall_decrypt_t {
	int ms_retval;
	uint32_t* ms_encrypt_input;
	uint32_t ms_input_length;
	uint32_t* ms_plaintext_output;
	uint32_t ms_output_length;
	uint8_t* ms_key;
	uint32_t ms_key_size;
} ms_ecall_decrypt_t;

typedef struct ms_ecall_encrypt_t {
	int ms_retval;
	uint32_t* ms_plaintext_input;
	uint32_t ms_input_length;
	uint32_t* ms_encrypted_output;
	uint32_t ms_output_length;
	uint8_t* ms_key;
	uint32_t ms_key_size;
} ms_ecall_encrypt_t;

typedef struct ms_ocall_print_t {
	const char* ms_str;
	float ms_value;
} ms_ocall_print_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print(void* pms)
{
	ms_ocall_print_t* ms = SGX_CAST(ms_ocall_print_t*, pms);
	ocall_print(ms->ms_str, ms->ms_value);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[1];
} ocall_table_Enclave = {
	1,
	{
		(void*)Enclave_ocall_print,
	}
};
sgx_status_t ecall_make_network(sgx_enclave_id_t eid, int* retval, int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches, int start_index)
{
	sgx_status_t status;
	ms_ecall_make_network_t ms;
	ms.ms_n = n;
	ms.ms_learning_rate = learning_rate;
	ms.ms_momentum = momentum;
	ms.ms_decay = decay;
	ms.ms_time_steps = time_steps;
	ms.ms_notruth = notruth;
	ms.ms_batch = batch;
	ms.ms_subdivisions = subdivisions;
	ms.ms_random = random;
	ms.ms_adam = adam;
	ms.ms_B1 = B1;
	ms.ms_B2 = B2;
	ms.ms_eps = eps;
	ms.ms_h = h;
	ms.ms_w = w;
	ms.ms_c = c;
	ms.ms_inputs = inputs;
	ms.ms_max_crop = max_crop;
	ms.ms_min_crop = min_crop;
	ms.ms_max_ratio = max_ratio;
	ms.ms_min_ratio = min_ratio;
	ms.ms_center = center;
	ms.ms_clip = clip;
	ms.ms_angle = angle;
	ms.ms_aspect = aspect;
	ms.ms_saturation = saturation;
	ms.ms_exposure = exposure;
	ms.ms_hue = hue;
	ms.ms_burn_in = burn_in;
	ms.ms_power = power;
	ms.ms_max_batches = max_batches;
	ms.ms_start_index = start_index;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_convolutional_layer(sgx_enclave_id_t eid, int* retval, int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, char* activation_s, int acti_length, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
	sgx_status_t status;
	ms_ecall_make_convolutional_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_h = h;
	ms.ms_w = w;
	ms.ms_c = c;
	ms.ms_n = n;
	ms.ms_groups = groups;
	ms.ms_size = size;
	ms.ms_stride = stride;
	ms.ms_padding = padding;
	ms.ms_activation_s = activation_s;
	ms.ms_acti_length = acti_length;
	ms.ms_batch_normalize = batch_normalize;
	ms.ms_binary = binary;
	ms.ms_xnor = xnor;
	ms.ms_adam = adam;
	ms.ms_flipped = flipped;
	ms.ms_dot = dot;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_connected_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, int outputs, char* activation_s, int acti_length, int batch_normalize, int adam)
{
	sgx_status_t status;
	ms_ecall_make_connected_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_inputs = inputs;
	ms.ms_outputs = outputs;
	ms.ms_activation_s = activation_s;
	ms.ms_acti_length = acti_length;
	ms.ms_batch_normalize = batch_normalize;
	ms.ms_adam = adam;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_softmax_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
	sgx_status_t status;
	ms_ecall_make_softmax_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_inputs = inputs;
	ms.ms_groups = groups;
	ms.ms_temperature = temperature;
	ms.ms_w = w;
	ms.ms_h = h;
	ms.ms_c = c;
	ms.ms_spatial = spatial;
	ms.ms_noloss = noloss;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_cost_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, char* cost_type, int cost_size, float scale, float ratio, float noobject_scale, float thresh)
{
	sgx_status_t status;
	ms_ecall_make_cost_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_inputs = inputs;
	ms.ms_cost_type = cost_type;
	ms.ms_cost_size = cost_size;
	ms.ms_scale = scale;
	ms.ms_ratio = ratio;
	ms.ms_noobject_scale = noobject_scale;
	ms.ms_thresh = thresh;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_maxpool_layer(sgx_enclave_id_t eid, int* retval, int batch, int h, int w, int c, int size, int stride, int padding)
{
	sgx_status_t status;
	ms_ecall_make_maxpool_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_h = h;
	ms.ms_w = w;
	ms.ms_c = c;
	ms.ms_size = size;
	ms.ms_stride = stride;
	ms.ms_padding = padding;
	status = sgx_ecall(eid, 5, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_dropout_layer(sgx_enclave_id_t eid, int* retval, int batch, int inputs, float probability, int w, int h, int c, float* net_prev_output, int prev_size, float* net_prev_delta, int delta_size)
{
	sgx_status_t status;
	ms_ecall_make_dropout_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_inputs = inputs;
	ms.ms_probability = probability;
	ms.ms_w = w;
	ms.ms_h = h;
	ms.ms_c = c;
	ms.ms_net_prev_output = net_prev_output;
	ms.ms_prev_size = prev_size;
	ms.ms_net_prev_delta = net_prev_delta;
	ms.ms_delta_size = delta_size;
	status = sgx_ecall(eid, 6, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_make_avgpool_layer(sgx_enclave_id_t eid, int* retval, int batch, int w, int h, int c)
{
	sgx_status_t status;
	ms_ecall_make_avgpool_layer_t ms;
	ms.ms_batch = batch;
	ms.ms_w = w;
	ms.ms_h = h;
	ms.ms_c = c;
	status = sgx_ecall(eid, 7, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_allocate_workspace(sgx_enclave_id_t eid, int* retval, int workspace_size)
{
	sgx_status_t status;
	ms_ecall_allocate_workspace_t ms;
	ms.ms_workspace_size = workspace_size;
	status = sgx_ecall(eid, 8, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_transfer_weights(sgx_enclave_id_t eid, int* retval, float* vec, int length, int layer_i, char type, int additional)
{
	sgx_status_t status;
	ms_ecall_transfer_weights_t ms;
	ms.ms_vec = vec;
	ms.ms_length = length;
	ms.ms_layer_i = layer_i;
	ms.ms_type = type;
	ms.ms_additional = additional;
	status = sgx_ecall(eid, 9, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_net_output_return(sgx_enclave_id_t eid, int* retval, float* net_output, int length, uint8_t* tag_buffer, int tag_length)
{
	sgx_status_t status;
	ms_ecall_net_output_return_t ms;
	ms.ms_net_output = net_output;
	ms.ms_length = length;
	ms.ms_tag_buffer = tag_buffer;
	ms.ms_tag_length = tag_length;
	status = sgx_ecall(eid, 10, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_forward_network(sgx_enclave_id_t eid, int* retval, float* net_input, int l_inputs, uint8_t* tag_buffer, int l_tag, int net_train)
{
	sgx_status_t status;
	ms_ecall_forward_network_t ms;
	ms.ms_net_input = net_input;
	ms.ms_l_inputs = l_inputs;
	ms.ms_tag_buffer = tag_buffer;
	ms.ms_l_tag = l_tag;
	ms.ms_net_train = net_train;
	status = sgx_ecall(eid, 11, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_attest_session_token(sgx_enclave_id_t eid, int* retval, int session_token, char* attestation_buffer, int attest_size, uint8_t* tag_buffer, int l_tag)
{
	sgx_status_t status;
	ms_ecall_attest_session_token_t ms;
	ms.ms_session_token = session_token;
	ms.ms_attestation_buffer = attestation_buffer;
	ms.ms_attest_size = attest_size;
	ms.ms_tag_buffer = tag_buffer;
	ms.ms_l_tag = l_tag;
	status = sgx_ecall(eid, 12, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_decrypt(sgx_enclave_id_t eid, int* retval, uint32_t* encrypt_input, uint32_t input_length, uint32_t* plaintext_output, uint32_t output_length, uint8_t* key, uint32_t key_size)
{
	sgx_status_t status;
	ms_ecall_decrypt_t ms;
	ms.ms_encrypt_input = encrypt_input;
	ms.ms_input_length = input_length;
	ms.ms_plaintext_output = plaintext_output;
	ms.ms_output_length = output_length;
	ms.ms_key = key;
	ms.ms_key_size = key_size;
	status = sgx_ecall(eid, 13, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_encrypt(sgx_enclave_id_t eid, int* retval, uint32_t* plaintext_input, uint32_t input_length, uint32_t* encrypted_output, uint32_t output_length, uint8_t* key, uint32_t key_size)
{
	sgx_status_t status;
	ms_ecall_encrypt_t ms;
	ms.ms_plaintext_input = plaintext_input;
	ms.ms_input_length = input_length;
	ms.ms_encrypted_output = encrypted_output;
	ms.ms_output_length = output_length;
	ms.ms_key = key;
	ms.ms_key_size = key_size;
	status = sgx_ecall(eid, 14, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

