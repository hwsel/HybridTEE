#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include "darknetp_ta.h"

#include "convolutional_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "dropout_layer_TA.h"

#include "connected_layer_TA.h"
#include "softmax_layer_TA.h"
#include "cost_layer_TA.h"
#include "network_TA.h"

#include "activations_TA.h"
#include "darknet_TA.h"
#include "diffprivate_TA.h"
#include "parser_TA.h"
#include "math_TA.h"
#include "aes_TA.h"

#define LOOKUP_SIZE 4096

float *netta_truth;
int netnum = 0;
layer_TA softmax_lta;

uint32_t gGlobalBufferTracker = 8;
//bool TA_threshold_exceeded = false;

// Unique identifier for SGX - 36 characters for UUID + 10 characters for token
char UID[46] = "b8734d20-832e-43a9-bf9f-17f3e9e041e2";

TEE_Result TA_CreateEntryPoint(void)
{
    //DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    //DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("I'm Vincent, from secure world!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

static TEE_Result make_netowork_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    int n = params0[0];
    int time_steps = params0[1];
    int notruth = params0[2];
    int batch = params0[3];
    int subdivisions = params0[4];
    int random = params0[5];
    int adam = params0[6];
    int h = params0[7];
    int w = params0[8];
    int c = params0[9];
    int inputs = params0[10];
    int max_crop = params0[11];
    int min_crop = params0[12];
    int center = params0[13];
    int burn_in = params0[14];
    int max_batches = params0[15];

    float learning_rate = params1[0];
    float momentum = params1[1];
    float decay = params1[2];
    float B1 = params1[3];
    float B2 = params1[4];
    float eps = params1[5];
    float max_ratio = params1[6];
    float min_ratio = params1[7];
    float clip = params1[8];
    float angle = params1[9];
    float aspect = params1[10];
    float saturation = params1[11];
    float exposure = params1[12];
    float hue = params1[13];
    float power = params1[14];

    // Update TCB tracker
    gGlobalBufferTracker += (16 * sizeof(int));
    gGlobalBufferTracker += (15 * sizeof(float));

    make_network_TA(n, learning_rate, momentum, decay, time_steps, notruth, batch, subdivisions, random, adam, B1, B2, eps, h, w, c, inputs, max_crop, min_crop, max_ratio, min_ratio, center, clip, angle, aspect, saturation, exposure, hue, burn_in, power, max_batches);

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }
    */
    return TEE_SUCCESS;
}

static TEE_Result update_net_agrv_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INOUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    netta.workspace = params[1].memref.buffer;

    // Update TCB tracker
    gGlobalBufferTracker += (uint32_t)params[1].memref.size;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }
    */
    return TEE_SUCCESS;
}


static TEE_Result make_convolutional_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float params1 = params[1].value.a;
    char *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int n = params0[4];
    int groups = params0[5];
    int size = params0[6];
    int stride = params0[7];
    int padding = params0[8];
    int batch_normalize = params0[9];
    int binary = params0[10];
    int xnor = params0[11];
    int adam = params0[12];
    int flipped = params0[13];
    float dot = params1;
    char *acti = params2;

    ACTIVATION_TA activation = get_activation_TA(acti);

    gGlobalBufferTracker += (12 + 14 * sizeof(int) + 2 * sizeof(float));

    layer_TA lta = make_convolutional_layer_TA_new(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, adam, flipped, dot);
    netta.layers[netnum] = lta;
    netnum++;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}

static TEE_Result make_maxpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int size = params0[4];
    int stride = params0[5];
    int padding = params0[6];

    gGlobalBufferTracker += (4 + 7 * sizeof(int));

    layer_TA lta = make_maxpool_layer_TA(batch, h, w, c, size, stride, padding);
    netta.layers[netnum] = lta;
    netnum++;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}

static TEE_Result make_dropout_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    float *params2 = params[2].memref.buffer;
    float *params3 = params[3].memref.buffer;
    int buffersize = params[2].memref.size / sizeof(float);

    int *passint;
    passint = params0;
    int batch = passint[0];
    int inputs = passint[1];
    int w = passint[2];
    int h = passint[3];
    int c = passint[4];
    float probability = params1[0];

    float *net_prev_output = params2;
    float *net_prev_delta = params3;

    // Update TCB tracker
    gGlobalBufferTracker += (28 + (6*sizeof(int)) + sizeof(float));

    layer_TA lta = make_dropout_layer_TA_new(batch, inputs, probability, w, h, c, netnum);

    if(netnum == 0){
      for(int z=0; z<buffersize; z++){
        lta.output[z] = net_prev_output[z];
        lta.delta[z] = net_prev_delta[z];
      }
    }else{
        lta.output = netta.layers[netnum-1].output;
        lta.delta = netta.layers[netnum-1].delta;
    }

    netta.layers[netnum] = lta;
    netnum++;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);

    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}


static TEE_Result make_connected_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    passarg = params[0].memref.buffer;
    int batch = passarg[0];
    int inputs = passarg[1];
    int outputs = passarg[2];
    int batch_normalize = passarg[3];
    int adam = passarg[4];

    char *acti;
    acti = params[1].memref.buffer;
    ACTIVATION_TA activation = get_activation_TA(acti);

    // Update TCB tracker
    gGlobalBufferTracker += (8 + 5 * sizeof(int));

    layer_TA lta = make_connected_layer_TA_new(batch, inputs, outputs, activation, batch_normalize, adam);
    netta.layers[netnum] = lta;
    netnum++;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);

    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}

static TEE_Result make_softmax_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];
    int groups = params0[2];
    int w = params0[3];
    int h = params0[4];
    int c = params0[5];
    int spatial = params0[6];
    int noloss = params0[7];
    float temperature = params[1].value.a;

    // Update TCB tracker
    //gGlobalBufferTracker += (4 + (8*sizeof(int)) + sizeof(float));

    softmax_lta = make_softmax_layer_TA_new(batch, inputs, groups, temperature, w, h, c, spatial, noloss);
    //netta.layers[netnum] = lta;
    //netnum++;

    //EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }
    */
    return TEE_SUCCESS;
}

static TEE_Result make_cost_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];

    float *params1 = params[1].memref.buffer;
    float scale = params1[0];
    float ratio = params1[1];
    float noobject_scale = params1[2];
    float thresh = params1[3];

    char *cost_t;
    cost_t = params[2].memref.buffer;
    COST_TYPE_TA cost_type = get_cost_type_TA(cost_t);

    // Update TCB tracker
    gGlobalBufferTracker += (12 + (2*sizeof(int)) + (4*sizeof(float)));

    layer_TA lta = make_cost_layer_TA_new(batch, inputs, cost_type, scale, ratio, noobject_scale, thresh);
    netta.layers[netnum] = lta;
    netnum++;

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}


static TEE_Result transfer_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];
    int additional = params1[2];

    char type = params[2].value.a;

    // Update TCB tracker
    gGlobalBufferTracker += (8 + sizeof(char) + (3*sizeof(int)));

    load_weights_TA(vec, length, layer_i, type, additional);

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}

static TEE_Result save_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];

    char type = params[2].value.a;

    float *weights_encrypted = malloc(sizeof(float)*length);
    save_weights_TA(weights_encrypted, length, layer_i, type);

    for(int z=0; z<length; z++){
        vec[z] = weights_encrypted[z];
    }

    free(weights_encrypted);
    return TEE_SUCCESS;
}


static TEE_Result forward_network_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;

    netta.input = net_input;
    netta.train = net_train;

    forward_network_TA();
    /*if(TA_threshold_exceeded)
    {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    */
    return TEE_SUCCESS;
}

static TEE_Result forward_softmax_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    TEE_Result status = TEE_SUCCESS;
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    uint8_t *tag_buffer = params[1].memref.buffer;
    float *net_output = params[2].memref.buffer;

    memcpy(ta_tag_buffer, tag_buffer, 16);
    EMSG("\nDecrypt the partial results size = %d\n", softmax_lta.inputs);
    // Decrypt the partial results
    int result = aes_cbc_TA((char *)"decrypt", net_input, softmax_lta.inputs, 1);
    EMSG("\nDone...\n");
    if(result != 0)
    {
        EMSG("\n Decryption failed...!!!\n");
	status = TEE_ERROR_GENERIC;
    }
    else
    {
        //forward_network_TA();
        forward_softmax_layer_TA(net_input);

        for(int z = 0; z < softmax_lta.outputs * softmax_lta.batch; z++){
            net_output[z] = softmax_lta.output[z];
        }
    }

    free(ta_tag_buffer);
    return status;
}

static TEE_Result backward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    //float *ltaoutput_diff = diff_private(lta.output, lta.outputs*lta.batch, 4.0f, 4.0f);
    //float *ltadelta_diff = diff_private(lta.delta, lta.outputs*lta.batch, 4.0f, 4.0f);
    //IMSG("diff");


    float *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    float *buffersize = params[0].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params0[z] = ta_net_input[z];
        params1[z] = ta_net_delta[z];
    }

    //free(ltaoutput_diff);
    //free(ltadelta_diff);
    return TEE_SUCCESS;
}



static TEE_Result backward_network_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *ca_net_input = params[0].memref.buffer;
    float *ca_net_delta = params[1].memref.buffer;
    int net_train = params[2].value.a;

    netta.train = net_train;

    backward_network_TA(ca_net_input, ca_net_delta);

    return TEE_SUCCESS;
}

static TEE_Result update_network_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    update_args_TA a;
    a.batch = params0[0];
    a.adam = params0[1];
    a.t = params0[2];
    a.learning_rate = params1[0];
    a.momentum = params1[1];
    a.decay = params1[2];
    a.B1 = params1[3];
    a.B2 = params1[4];
    a.eps = params1[5];

    update_network_TA(a);

    free(ta_net_input);
    free(ta_net_delta);

    return TEE_SUCCESS;
}

static TEE_Result net_truth_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int size_truth = params[0].memref.size;
    float *params0 = params[0].memref.buffer;

    netta_truth = malloc(size_truth);
    for(int z=0; z<size_truth/sizeof(float); z++){
        netta_truth[z] = params0[z];
    }
    netta.truth = netta_truth;

    return TEE_SUCCESS;
}

static TEE_Result calc_network_loss_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);

    int *params0 = params[0].memref.buffer;
    int n = params0[0];
    int batch = params0[1];

    calc_network_loss_TA(n, batch);

    return TEE_SUCCESS;
}


static TEE_Result net_output_return_TA_params(uint32_t param_types,
                                              TEE_Param params[4])
{
    TEE_Result status = TEE_SUCCESS;
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    uint8_t *tagbuffer = params[1].memref.buffer;
    int tag_length = params[1].memref.size;

    // remove confidence scores
    //float * rm_conf[buffersize];
    //float maxconf; maxconf = -0.1f;
    //int maxidx; maxidx = 0;

    // Update TCB tracker
    //gGlobalBufferTracker += ((3*sizeof(int)) + (2* sizeof(float)) + 4);

    /*
    for(int z=0; z<buffersize; z++){
        if(ta_net_output[z] > maxconf){
            maxconf = ta_net_output[z];
            maxidx = z;
        }
        ta_net_output[z] = 0.0f;
    }
    ta_net_output[maxidx] = 1.0f;
    */

    ta_tag_buffer = (uint8_t *)calloc(1, tag_length);
    EMSG("\nEncrypt the partial results size = %d\n",buffersize);
    // Encrypt the partial result
    int result = aes_cbc_TA((char *)"encrypt", ta_net_output, buffersize, 1);
    EMSG("\nDone...\n");

    if(result == 0)
    {
        for(int z=0; z<buffersize; z++){
            params0[z] = ta_net_output[z];
        }
        for(int z=0; z<tag_length; z++){
            tagbuffer[z] = ta_tag_buffer[z];
        }
    }
    else
    {
        status = TEE_ERROR_GENERIC;
    }

    free(ta_net_output);

    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
	status = TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return status;
}

static TEE_Result allocate_workspace_TA_params(uint32_t param_types,
                                               TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);

    int *params0 = params[0].memref.buffer;
    uint32_t wzsize = params0[0];

    netta.workspace = (float *)calloc(wzsize, sizeof(float));
    gGlobalBufferTracker += wzsize;
    EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
    /*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
    {
        TA_threshold_exceeded = true;
        return TEE_ERROR_OUT_OF_MEMORY;
    }*/
    return TEE_SUCCESS;
}

static TEE_Result generate_secure_random_number(uint32_t param_types,
                                                TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_VALUE_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    TEE_Result status = TEE_SUCCESS;
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    //uint32_t *random_number = params[0].memref.buffer;

    //TEE_GenerateRandom(&ta_random_number, sizeof(ta_random_number));
    ta_random_number = (uint32_t)rand();
    EMSG("\nRandom number: %d\n",ta_random_number);
    //*random_number = ta_random_number;

    params[0].value.a = ta_random_number;

    return TEE_SUCCESS;
}

void getDecStr(char* str, uint8_t len, uint8_t total_count, uint32_t val)
{
    uint8_t i;
    for(i = 1; i <= len; i++)
    {
        str[len-i] = (char)((val % 10UL) + '0');
        val/=10;
    }
    for(; i <= total_count; i++)
    {
        str[i] = '0';
    }
    //str[i] = '\0';
}

int num_of_digits_int(int a)
{
    int x;
    for (x = 1; a >= 10; x++)
    {
       a = a / 10;
    }
    return x;
}

static TEE_Result verify_sgx_attestation_token(uint32_t param_types,
                                               TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INOUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    TEE_Result status = TEE_SUCCESS;
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    char *attestation_buffer = params[0].memref.buffer;
    int attest_size = params[0].memref.size;
    uint8_t *tag_buffer = params[1].memref.buffer;
    int tag_size = params[1].memref.size;

    ta_tag_buffer = (uint8_t *)calloc(1,tag_size);
    memcpy(ta_tag_buffer, tag_buffer, tag_size);

    EMSG("\nDecrypt the uuid token size = %d\n", attest_size);
    int result = aes_cbc_TA_token((char *)"decrypt", attestation_buffer, attest_size);
    EMSG("\nDone...\n");
    if(result != 0)
    {
        EMSG("\n Decryption failed...!!!\n");
        status = TEE_ERROR_GENERIC;
    }
    else
    {
        int total_count = 10;
	int num_of_digits = num_of_digits_int(ta_random_number);
	//int num_of_digits = 10;
        //char *token = malloc(sizeof(char) * num_of_digits);
	char token[10];
        
	getDecStr(token, num_of_digits, total_count, ta_random_number);
	EMSG("Expected token: %s", token);

	EMSG("Attestation string: %s\n", attestation_buffer);
	// Compare the UUID
        if(strncmp(attestation_buffer, UID, 36) != 0)
	{
            EMSG("\nUUID comparison failed...\n");
	    status = TEE_ERROR_GENERIC;
	}
	else
	{
            EMSG("\nUUID comparison passed...");
	}

	// Compare the session token
	if(strncmp(attestation_buffer+36, token, num_of_digits) != 0)
        {
            EMSG("\nToken comparison failed...\n");
	    status = TEE_ERROR_GENERIC;
        }
	else
        {
            EMSG("\nToken comparison passed...\n");
        }

	//free(token);
    }
    free(ta_tag_buffer);
    return status;
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */

    switch (cmd_id) {
        case MAKE_NETWORK_CMD:
        return make_netowork_TA_params(param_types, params);

        case WORKSPACE_NETWORK_CMD:
        return update_net_agrv_TA_params(param_types, params);

        case MAKE_CONV_CMD:
        return make_convolutional_layer_TA_params(param_types, params);

        case MAKE_MAX_CMD:
        return make_maxpool_layer_TA_params(param_types, params);

        case MAKE_DROP_CMD:
        return make_dropout_layer_TA_params(param_types, params);

        case MAKE_CONNECTED_CMD:
        return make_connected_layer_TA_params(param_types, params);

        case MAKE_SOFTMAX_CMD:
        return make_softmax_layer_TA_params(param_types, params);

        case MAKE_COST_CMD:
        return make_cost_layer_TA_params(param_types, params);

        case TRANS_WEI_CMD:
        return transfer_weights_TA_params(param_types, params);

        case SAVE_WEI_CMD:
            return save_weights_TA_params(param_types, params);

        case FORWARD_CMD:
        return forward_network_TA_params(param_types, params);

        case BACKWARD_CMD:
        return backward_network_TA_params(param_types, params);

        case BACKWARD_ADD_CMD:
        return backward_network_back_TA_params(param_types, params);

        case UPDATE_CMD:
        return update_network_TA_params(param_types, params);

        case NET_TRUTH_CMD:
        return net_truth_TA_params(param_types, params);

        case CALC_LOSS_CMD:
        return calc_network_loss_TA_params(param_types, params);

        case OUTPUT_RETURN_CMD:
        return net_output_return_TA_params(param_types, params);

        case ALLOCATE_WORKSPACE_CMD:
	return allocate_workspace_TA_params(param_types, params);

	case FORWARD_SOFTMAX_CMD:
        return forward_softmax_TA_params(param_types, params);

	case RAND_NUMBER_GENERATOR:
        return generate_secure_random_number(param_types, params);

	case VERIFY_ATTESTATION_TOKEN:
	return verify_sgx_attestation_token(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
