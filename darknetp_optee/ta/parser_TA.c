#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknetp_ta.h"
#include "network_TA.h"
#include <parser_TA.h>
#include <blas_TA.h>
#include "math_TA.h"
#include "aes_TA.h"
#include "aes.h"

int aes_cbc_TA(char* xcrypt, float* gradient, int org_len, int mode)
{
    int result = 0;
    //convert float array to uint_8 one by one
    uint8_t *byte;
    //uint8_t array[org_len*4];
    uint8_t *array = (uint8_t *)calloc(org_len*4, sizeof(uint8_t));
    for(int z = 0; z < org_len; z++){
        byte = (uint8_t*)(&gradient[z]);
        for(int y = 0; y < 4; y++){
            array[z*4 + y] = byte[y];
        }
    }

    if(mode == 0)
    {
        //set ctx, iv, and key for aes
        int enc_len = (int)(org_len/4);
        struct AES_ctx ctx;
        uint8_t iv[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
        uint8_t key[16] = { (uint8_t)0x2b, (uint8_t)0x7e, (uint8_t)0x15, (uint8_t)0x16, (uint8_t)0x28, (uint8_t)0xae, (uint8_t)0xd2, (uint8_t)0xa6, (uint8_t)0xab, (uint8_t)0xf7, (uint8_t)0x15, (uint8_t)0x88, (uint8_t)0x09, (uint8_t)0xcf, (uint8_t)0x4f, (uint8_t)0x3c };

        //encryption
        AES_init_ctx_iv(&ctx, key, iv);
        for (int i = 0; i < enc_len; ++i)
        {
            if(strncmp(xcrypt, "encrypt", 2) == 0){
                AES_CBC_encrypt_buffer(&ctx, array + (i * 16), 16);
            }else if(strncmp(xcrypt, "decrypt", 2) == 0){
                AES_CBC_decrypt_buffer(&ctx, array + (i * 16), 16);
            }
        }
    }
    else
    {
        result = aes_gcm_entry(xcrypt, array, ta_tag_buffer, org_len*4);
        if(result != 0)
        {
            if(strncmp(xcrypt, "encrypt", 2) == 0){
                EMSG("Encryption failed: %d\n",result);
            }
            else{
                EMSG("Decryption failed: %d\n",result);
            }
        }
    }

    //convert uint8_t to float one by one
    for(int z = 0; z < org_len; z++){
        gradient[z] = *(float*)(&array[z*4]);
    }

    free(array);
    return result;
}

int aes_cbc_TA_token(char* xcrypt, char* attestation_buffer, int len)
{
    int result = aes_gcm_entry(xcrypt, (uint8_t *)attestation_buffer, ta_tag_buffer, len);
    if(result != 0)
    {
        if(strncmp(xcrypt, "encrypt", 2) == 0){
            EMSG("Encryption failed: %d\n", result);
        }
        else{
            EMSG("Decryption failed: %d\n", result);
        }
    }

    return result;
}

void transpose_matrix_TA(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}


void load_weights_TA(float *vec, int length, int layer_i, char type, int transpose)
{
    // decrypt
    float *tempvec = malloc(length*sizeof(float));
    copy_cpu_TA(length, vec, 1, tempvec, 1);
    //aes_cbc_TA("decrypt", tempvec, length);

    // copy
    layer_TA l = netta.layers[layer_i];

    if(type == 'b'){
        copy_cpu_TA(length, tempvec, 1, l.biases, 1);
    }
    else if(type == 'w'){
        copy_cpu_TA(length, tempvec, 1, l.weights, 1);
    }
    else if(type == 's'){
        copy_cpu_TA(length, tempvec, 1, l.scales, 1);
    }
    else if(type == 'm'){
        copy_cpu_TA(length, tempvec, 1, l.rolling_mean, 1);
    }
    else if(type == 'v'){
        copy_cpu_TA(length, tempvec, 1, l.rolling_variance, 1);
    }


    if(l.type == CONVOLUTIONAL_TA || l.type == DECONVOLUTIONAL_TA){
        if(l.flipped && type == 'w'){
            transpose_matrix_TA(l.weights, l.c*l.size*l.size, l.n);
        }
    }
    else if(l.type == CONNECTED_TA){
        if(transpose && type == 'w'){
            transpose_matrix_TA(l.weights, l.inputs, l.outputs);
        }
    }

    free(tempvec);
}

void save_weights_TA(float *weights_encrypted, int length, int layer_i, char type)
{
    layer_TA l = netta.layers[layer_i];

    if(type == 'b'){
        copy_cpu_TA(length, l.biases, 1, weights_encrypted, 1);
    }
    else if(type == 'w'){
        copy_cpu_TA(length, l.weights, 1, weights_encrypted, 1);
    }
    else if(type == 's'){
        copy_cpu_TA(length, l.scales, 1, weights_encrypted, 1);
    }
    else if(type == 'm'){
        copy_cpu_TA(length, l.rolling_mean, 1, weights_encrypted, 1);
    }
    else if(type == 'v'){
        copy_cpu_TA(length, l.rolling_variance, 1, weights_encrypted, 1);
    }

    //encryption
    aes_cbc_TA((char *)"encrypt", weights_encrypted, length, 1);
}
