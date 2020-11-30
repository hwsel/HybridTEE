#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_ecall_make_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_network_t* ms = SGX_CAST(ms_ecall_make_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = ecall_make_network(ms->ms_n, ms->ms_learning_rate, ms->ms_momentum, ms->ms_decay, ms->ms_time_steps, ms->ms_notruth, ms->ms_batch, ms->ms_subdivisions, ms->ms_random, ms->ms_adam, ms->ms_B1, ms->ms_B2, ms->ms_eps, ms->ms_h, ms->ms_w, ms->ms_c, ms->ms_inputs, ms->ms_max_crop, ms->ms_min_crop, ms->ms_max_ratio, ms->ms_min_ratio, ms->ms_center, ms->ms_clip, ms->ms_angle, ms->ms_aspect, ms->ms_saturation, ms->ms_exposure, ms->ms_hue, ms->ms_burn_in, ms->ms_power, ms->ms_max_batches, ms->ms_start_index);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_convolutional_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_convolutional_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_convolutional_layer_t* ms = SGX_CAST(ms_ecall_make_convolutional_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_activation_s = ms->ms_activation_s;
	int _tmp_acti_length = ms->ms_acti_length;
	size_t _len_activation_s = _tmp_acti_length;
	char* _in_activation_s = NULL;

	CHECK_UNIQUE_POINTER(_tmp_activation_s, _len_activation_s);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_activation_s != NULL && _len_activation_s != 0) {
		if ( _len_activation_s % sizeof(*_tmp_activation_s) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_activation_s = (char*)malloc(_len_activation_s);
		if (_in_activation_s == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_activation_s, _len_activation_s, _tmp_activation_s, _len_activation_s)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_make_convolutional_layer(ms->ms_batch, ms->ms_h, ms->ms_w, ms->ms_c, ms->ms_n, ms->ms_groups, ms->ms_size, ms->ms_stride, ms->ms_padding, _in_activation_s, _tmp_acti_length, ms->ms_batch_normalize, ms->ms_binary, ms->ms_xnor, ms->ms_adam, ms->ms_flipped, ms->ms_dot);

err:
	if (_in_activation_s) free(_in_activation_s);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_connected_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_connected_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_connected_layer_t* ms = SGX_CAST(ms_ecall_make_connected_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_activation_s = ms->ms_activation_s;
	int _tmp_acti_length = ms->ms_acti_length;
	size_t _len_activation_s = _tmp_acti_length;
	char* _in_activation_s = NULL;

	CHECK_UNIQUE_POINTER(_tmp_activation_s, _len_activation_s);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_activation_s != NULL && _len_activation_s != 0) {
		if ( _len_activation_s % sizeof(*_tmp_activation_s) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_activation_s = (char*)malloc(_len_activation_s);
		if (_in_activation_s == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_activation_s, _len_activation_s, _tmp_activation_s, _len_activation_s)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_make_connected_layer(ms->ms_batch, ms->ms_inputs, ms->ms_outputs, _in_activation_s, _tmp_acti_length, ms->ms_batch_normalize, ms->ms_adam);

err:
	if (_in_activation_s) free(_in_activation_s);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_softmax_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_softmax_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_softmax_layer_t* ms = SGX_CAST(ms_ecall_make_softmax_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = ecall_make_softmax_layer(ms->ms_batch, ms->ms_inputs, ms->ms_groups, ms->ms_temperature, ms->ms_w, ms->ms_h, ms->ms_c, ms->ms_spatial, ms->ms_noloss);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_cost_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_cost_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_cost_layer_t* ms = SGX_CAST(ms_ecall_make_cost_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_cost_type = ms->ms_cost_type;
	int _tmp_cost_size = ms->ms_cost_size;
	size_t _len_cost_type = _tmp_cost_size;
	char* _in_cost_type = NULL;

	CHECK_UNIQUE_POINTER(_tmp_cost_type, _len_cost_type);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_cost_type != NULL && _len_cost_type != 0) {
		if ( _len_cost_type % sizeof(*_tmp_cost_type) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_cost_type = (char*)malloc(_len_cost_type);
		if (_in_cost_type == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_cost_type, _len_cost_type, _tmp_cost_type, _len_cost_type)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_make_cost_layer(ms->ms_batch, ms->ms_inputs, _in_cost_type, _tmp_cost_size, ms->ms_scale, ms->ms_ratio, ms->ms_noobject_scale, ms->ms_thresh);

err:
	if (_in_cost_type) free(_in_cost_type);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_maxpool_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_maxpool_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_maxpool_layer_t* ms = SGX_CAST(ms_ecall_make_maxpool_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = ecall_make_maxpool_layer(ms->ms_batch, ms->ms_h, ms->ms_w, ms->ms_c, ms->ms_size, ms->ms_stride, ms->ms_padding);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_dropout_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_dropout_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_dropout_layer_t* ms = SGX_CAST(ms_ecall_make_dropout_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_net_prev_output = ms->ms_net_prev_output;
	int _tmp_prev_size = ms->ms_prev_size;
	size_t _len_net_prev_output = _tmp_prev_size;
	float* _in_net_prev_output = NULL;
	float* _tmp_net_prev_delta = ms->ms_net_prev_delta;
	int _tmp_delta_size = ms->ms_delta_size;
	size_t _len_net_prev_delta = _tmp_delta_size;
	float* _in_net_prev_delta = NULL;

	CHECK_UNIQUE_POINTER(_tmp_net_prev_output, _len_net_prev_output);
	CHECK_UNIQUE_POINTER(_tmp_net_prev_delta, _len_net_prev_delta);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_net_prev_output != NULL && _len_net_prev_output != 0) {
		if ( _len_net_prev_output % sizeof(*_tmp_net_prev_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_net_prev_output = (float*)malloc(_len_net_prev_output);
		if (_in_net_prev_output == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_net_prev_output, _len_net_prev_output, _tmp_net_prev_output, _len_net_prev_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_net_prev_delta != NULL && _len_net_prev_delta != 0) {
		if ( _len_net_prev_delta % sizeof(*_tmp_net_prev_delta) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_net_prev_delta = (float*)malloc(_len_net_prev_delta);
		if (_in_net_prev_delta == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_net_prev_delta, _len_net_prev_delta, _tmp_net_prev_delta, _len_net_prev_delta)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_make_dropout_layer(ms->ms_batch, ms->ms_inputs, ms->ms_probability, ms->ms_w, ms->ms_h, ms->ms_c, _in_net_prev_output, _tmp_prev_size, _in_net_prev_delta, _tmp_delta_size);

err:
	if (_in_net_prev_output) free(_in_net_prev_output);
	if (_in_net_prev_delta) free(_in_net_prev_delta);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_make_avgpool_layer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_make_avgpool_layer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_make_avgpool_layer_t* ms = SGX_CAST(ms_ecall_make_avgpool_layer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = ecall_make_avgpool_layer(ms->ms_batch, ms->ms_w, ms->ms_h, ms->ms_c);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_allocate_workspace(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_allocate_workspace_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_allocate_workspace_t* ms = SGX_CAST(ms_ecall_allocate_workspace_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = ecall_allocate_workspace(ms->ms_workspace_size);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_transfer_weights(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_transfer_weights_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_transfer_weights_t* ms = SGX_CAST(ms_ecall_transfer_weights_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_vec = ms->ms_vec;
	int _tmp_length = ms->ms_length;
	size_t _len_vec = _tmp_length;
	float* _in_vec = NULL;

	CHECK_UNIQUE_POINTER(_tmp_vec, _len_vec);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_vec != NULL && _len_vec != 0) {
		if ( _len_vec % sizeof(*_tmp_vec) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_vec = (float*)malloc(_len_vec);
		if (_in_vec == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_vec, _len_vec, _tmp_vec, _len_vec)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_transfer_weights(_in_vec, _tmp_length, ms->ms_layer_i, ms->ms_type, ms->ms_additional);

err:
	if (_in_vec) free(_in_vec);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_net_output_return(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_net_output_return_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_net_output_return_t* ms = SGX_CAST(ms_ecall_net_output_return_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_net_output = ms->ms_net_output;
	int _tmp_length = ms->ms_length;
	size_t _len_net_output = _tmp_length;
	float* _in_net_output = NULL;
	uint8_t* _tmp_tag_buffer = ms->ms_tag_buffer;
	int _tmp_tag_length = ms->ms_tag_length;
	size_t _len_tag_buffer = _tmp_tag_length;
	uint8_t* _in_tag_buffer = NULL;

	CHECK_UNIQUE_POINTER(_tmp_net_output, _len_net_output);
	CHECK_UNIQUE_POINTER(_tmp_tag_buffer, _len_tag_buffer);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_net_output != NULL && _len_net_output != 0) {
		if ( _len_net_output % sizeof(*_tmp_net_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_net_output = (float*)malloc(_len_net_output);
		if (_in_net_output == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_net_output, _len_net_output, _tmp_net_output, _len_net_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_tag_buffer != NULL && _len_tag_buffer != 0) {
		if ( _len_tag_buffer % sizeof(*_tmp_tag_buffer) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_tag_buffer = (uint8_t*)malloc(_len_tag_buffer);
		if (_in_tag_buffer == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_tag_buffer, _len_tag_buffer, _tmp_tag_buffer, _len_tag_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_net_output_return(_in_net_output, _tmp_length, _in_tag_buffer, _tmp_tag_length);
	if (_in_net_output) {
		if (memcpy_s(_tmp_net_output, _len_net_output, _in_net_output, _len_net_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_tag_buffer) {
		if (memcpy_s(_tmp_tag_buffer, _len_tag_buffer, _in_tag_buffer, _len_tag_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_net_output) free(_in_net_output);
	if (_in_tag_buffer) free(_in_tag_buffer);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_forward_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_forward_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_forward_network_t* ms = SGX_CAST(ms_ecall_forward_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_net_input = ms->ms_net_input;
	int _tmp_l_inputs = ms->ms_l_inputs;
	size_t _len_net_input = _tmp_l_inputs;
	float* _in_net_input = NULL;
	uint8_t* _tmp_tag_buffer = ms->ms_tag_buffer;
	int _tmp_l_tag = ms->ms_l_tag;
	size_t _len_tag_buffer = _tmp_l_tag;
	uint8_t* _in_tag_buffer = NULL;

	CHECK_UNIQUE_POINTER(_tmp_net_input, _len_net_input);
	CHECK_UNIQUE_POINTER(_tmp_tag_buffer, _len_tag_buffer);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_net_input != NULL && _len_net_input != 0) {
		if ( _len_net_input % sizeof(*_tmp_net_input) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_net_input = (float*)malloc(_len_net_input);
		if (_in_net_input == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_net_input, _len_net_input, _tmp_net_input, _len_net_input)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_tag_buffer != NULL && _len_tag_buffer != 0) {
		if ( _len_tag_buffer % sizeof(*_tmp_tag_buffer) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_tag_buffer = (uint8_t*)malloc(_len_tag_buffer);
		if (_in_tag_buffer == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_tag_buffer, _len_tag_buffer, _tmp_tag_buffer, _len_tag_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_forward_network(_in_net_input, _tmp_l_inputs, _in_tag_buffer, _tmp_l_tag, ms->ms_net_train);

err:
	if (_in_net_input) free(_in_net_input);
	if (_in_tag_buffer) free(_in_tag_buffer);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_attest_session_token(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_attest_session_token_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_attest_session_token_t* ms = SGX_CAST(ms_ecall_attest_session_token_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_attestation_buffer = ms->ms_attestation_buffer;
	int _tmp_attest_size = ms->ms_attest_size;
	size_t _len_attestation_buffer = _tmp_attest_size;
	char* _in_attestation_buffer = NULL;
	uint8_t* _tmp_tag_buffer = ms->ms_tag_buffer;
	int _tmp_l_tag = ms->ms_l_tag;
	size_t _len_tag_buffer = _tmp_l_tag;
	uint8_t* _in_tag_buffer = NULL;

	CHECK_UNIQUE_POINTER(_tmp_attestation_buffer, _len_attestation_buffer);
	CHECK_UNIQUE_POINTER(_tmp_tag_buffer, _len_tag_buffer);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_attestation_buffer != NULL && _len_attestation_buffer != 0) {
		if ( _len_attestation_buffer % sizeof(*_tmp_attestation_buffer) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_attestation_buffer = (char*)malloc(_len_attestation_buffer);
		if (_in_attestation_buffer == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_attestation_buffer, _len_attestation_buffer, _tmp_attestation_buffer, _len_attestation_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_tag_buffer != NULL && _len_tag_buffer != 0) {
		if ( _len_tag_buffer % sizeof(*_tmp_tag_buffer) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_tag_buffer = (uint8_t*)malloc(_len_tag_buffer);
		if (_in_tag_buffer == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_tag_buffer, _len_tag_buffer, _tmp_tag_buffer, _len_tag_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_attest_session_token(ms->ms_session_token, _in_attestation_buffer, _tmp_attest_size, _in_tag_buffer, _tmp_l_tag);
	if (_in_attestation_buffer) {
		if (memcpy_s(_tmp_attestation_buffer, _len_attestation_buffer, _in_attestation_buffer, _len_attestation_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_tag_buffer) {
		if (memcpy_s(_tmp_tag_buffer, _len_tag_buffer, _in_tag_buffer, _len_tag_buffer)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_attestation_buffer) free(_in_attestation_buffer);
	if (_in_tag_buffer) free(_in_tag_buffer);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_decrypt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_decrypt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_decrypt_t* ms = SGX_CAST(ms_ecall_decrypt_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	uint32_t* _tmp_encrypt_input = ms->ms_encrypt_input;
	uint32_t _tmp_input_length = ms->ms_input_length;
	size_t _len_encrypt_input = _tmp_input_length;
	uint32_t* _in_encrypt_input = NULL;
	uint32_t* _tmp_plaintext_output = ms->ms_plaintext_output;
	uint32_t _tmp_output_length = ms->ms_output_length;
	size_t _len_plaintext_output = _tmp_output_length;
	uint32_t* _in_plaintext_output = NULL;
	uint8_t* _tmp_key = ms->ms_key;
	uint32_t _tmp_key_size = ms->ms_key_size;
	size_t _len_key = _tmp_key_size;
	uint8_t* _in_key = NULL;

	CHECK_UNIQUE_POINTER(_tmp_encrypt_input, _len_encrypt_input);
	CHECK_UNIQUE_POINTER(_tmp_plaintext_output, _len_plaintext_output);
	CHECK_UNIQUE_POINTER(_tmp_key, _len_key);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_encrypt_input != NULL && _len_encrypt_input != 0) {
		if ( _len_encrypt_input % sizeof(*_tmp_encrypt_input) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_encrypt_input = (uint32_t*)malloc(_len_encrypt_input);
		if (_in_encrypt_input == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_encrypt_input, _len_encrypt_input, _tmp_encrypt_input, _len_encrypt_input)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_plaintext_output != NULL && _len_plaintext_output != 0) {
		if ( _len_plaintext_output % sizeof(*_tmp_plaintext_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_plaintext_output = (uint32_t*)malloc(_len_plaintext_output)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_plaintext_output, 0, _len_plaintext_output);
	}
	if (_tmp_key != NULL && _len_key != 0) {
		if ( _len_key % sizeof(*_tmp_key) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_key = (uint8_t*)malloc(_len_key);
		if (_in_key == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_key, _len_key, _tmp_key, _len_key)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_decrypt(_in_encrypt_input, _tmp_input_length, _in_plaintext_output, _tmp_output_length, _in_key, _tmp_key_size);
	if (_in_plaintext_output) {
		if (memcpy_s(_tmp_plaintext_output, _len_plaintext_output, _in_plaintext_output, _len_plaintext_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_key) {
		if (memcpy_s(_tmp_key, _len_key, _in_key, _len_key)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_encrypt_input) free(_in_encrypt_input);
	if (_in_plaintext_output) free(_in_plaintext_output);
	if (_in_key) free(_in_key);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_encrypt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_encrypt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_encrypt_t* ms = SGX_CAST(ms_ecall_encrypt_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	uint32_t* _tmp_plaintext_input = ms->ms_plaintext_input;
	uint32_t _tmp_input_length = ms->ms_input_length;
	size_t _len_plaintext_input = _tmp_input_length;
	uint32_t* _in_plaintext_input = NULL;
	uint32_t* _tmp_encrypted_output = ms->ms_encrypted_output;
	uint32_t _tmp_output_length = ms->ms_output_length;
	size_t _len_encrypted_output = _tmp_output_length;
	uint32_t* _in_encrypted_output = NULL;
	uint8_t* _tmp_key = ms->ms_key;
	uint32_t _tmp_key_size = ms->ms_key_size;
	size_t _len_key = _tmp_key_size;
	uint8_t* _in_key = NULL;

	CHECK_UNIQUE_POINTER(_tmp_plaintext_input, _len_plaintext_input);
	CHECK_UNIQUE_POINTER(_tmp_encrypted_output, _len_encrypted_output);
	CHECK_UNIQUE_POINTER(_tmp_key, _len_key);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_plaintext_input != NULL && _len_plaintext_input != 0) {
		if ( _len_plaintext_input % sizeof(*_tmp_plaintext_input) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_plaintext_input = (uint32_t*)malloc(_len_plaintext_input);
		if (_in_plaintext_input == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_plaintext_input, _len_plaintext_input, _tmp_plaintext_input, _len_plaintext_input)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_encrypted_output != NULL && _len_encrypted_output != 0) {
		if ( _len_encrypted_output % sizeof(*_tmp_encrypted_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_encrypted_output = (uint32_t*)malloc(_len_encrypted_output)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_encrypted_output, 0, _len_encrypted_output);
	}
	if (_tmp_key != NULL && _len_key != 0) {
		if ( _len_key % sizeof(*_tmp_key) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_key = (uint8_t*)malloc(_len_key);
		if (_in_key == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_key, _len_key, _tmp_key, _len_key)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ms->ms_retval = ecall_encrypt(_in_plaintext_input, _tmp_input_length, _in_encrypted_output, _tmp_output_length, _in_key, _tmp_key_size);
	if (_in_encrypted_output) {
		if (memcpy_s(_tmp_encrypted_output, _len_encrypted_output, _in_encrypted_output, _len_encrypted_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_key) {
		if (memcpy_s(_tmp_key, _len_key, _in_key, _len_key)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_plaintext_input) free(_in_plaintext_input);
	if (_in_encrypted_output) free(_in_encrypted_output);
	if (_in_key) free(_in_key);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[15];
} g_ecall_table = {
	15,
	{
		{(void*)(uintptr_t)sgx_ecall_make_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_convolutional_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_connected_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_softmax_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_cost_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_maxpool_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_dropout_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_make_avgpool_layer, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_allocate_workspace, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_transfer_weights, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_net_output_return, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_forward_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_attest_session_token, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_decrypt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_encrypt, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[1][15];
} g_dyn_entry_table = {
	1,
	{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print(const char* str, float value)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_t));
	ocalloc_size -= sizeof(ms_ocall_print_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	ms->ms_value = value;
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

