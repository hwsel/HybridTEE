/*
 * aes-gcm-test.c
 */

#define AES_DEBUG

#include "aes.h"

const unsigned char t3_key[] = {
    0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c, 0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30, 0x83, 0x08
};
const unsigned char t3_iv[] = {
    0xca, 0xfe, 0xba, 0xbe, 0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88
};
const unsigned char t3_aad[] = {};
//const float array[] = {1.33, 4.67, 6.55, 3.88, 7.66, 0.76, 10.44, 2.11};

/*
const unsigned char t3_crypt[] = {
   0xea, 0x8f, 0x86, 0xd8, 0x7d, 0x83, 0xe7, 0x81 
};
*/

const unsigned char t3_tag[] = {
    0x4d, 0x5c, 0x2a, 0xf3, 0x27, 0xcd, 0x64, 0xa6, 0x2c, 0xf3, 0x5a, 0xbd, 0x2b, 0xa6, 0xfa, 0xb4
};


int aes_gcm_entry(char *xcrypt, uint8_t *gradient, uint8_t *tag_buffer, int size)
{
    int result;
    //unsigned char *t3_plain = (unsigned char *)malloc(sizeof(gradient));
    //memcpy((unsigned char *)t3_plain, array, sizeof(t3_plain));

    size = size * sizeof(uint8_t);
    unsigned char *t3_plain = (unsigned char *)gradient;
    //unsigned char* tag_buf = calloc(1,sizeof(t3_tag));
    unsigned char* tag_buf = (unsigned char *)tag_buffer;
  
    if(strncmp(xcrypt, "encrypt", 2) == 0)
    {
        unsigned char* crypt_buf = calloc(1,size);
        result = aes_gcm_ae(t3_key, sizeof(t3_key),
                            t3_iv, sizeof(t3_iv),
                            t3_plain, size,
                            t3_aad, sizeof(t3_aad),
                            crypt_buf, tag_buf);
	memcpy(gradient, (uint8_t *)crypt_buf, size);
	free(crypt_buf);
    }
    else if(strncmp(xcrypt, "decrypt", 2) == 0)
    {
        unsigned char* plain_buf = calloc(1,size);
        result = aes_gcm_ad(t3_key, sizeof(t3_key),
                            t3_iv, sizeof(t3_iv),
                            t3_plain, size,
                            t3_aad, sizeof(t3_aad),
                            tag_buf, plain_buf);
	memcpy(gradient, (uint8_t *)plain_buf, size);
	free(plain_buf);
    }
    else
    {
        printf("\n Incorrect argument for AES_GCM\n");
	result = -1;
    }

    return result;
}
