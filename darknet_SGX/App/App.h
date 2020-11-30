#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "darknet.h"

/*
 * Global definitions
 */

extern sgx_enclave_id_t global_eid;
extern float *net_input_back;
extern float *net_delta_back;
extern float *net_output_back;
extern int global_start_index; 
extern uint8_t *net_tag_buffer;

# define TOKEN_FILENAME   "enclave.token"
# define ENCLAVE_FILENAME "enclave.signed.so"

#define MAX_DATA_SIZE 500
#define SGX_AESGCM_MAC_SIZE 16
#define SGX_AESGCM_IV_SIZE 12

#define DECRYPT_AND_VERIFY     1

#define ENCRYPT_OUTPUT_DATA    1

