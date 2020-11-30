#include <err.h>
#include <stdio.h>
#include <string.h>

#include "darknet.h"
#include "activations.h"
#include "cost_layer.h"

#include "main.h"
#include "socket_api.h"
#include "utils.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* TEE resources */
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory workspaceSM;

float *net_input_back;
float *net_delta_back;
float *net_output_back;
float *net_result_back;
uint8_t *net_tag_buffer;
uint32_t session_token;

int sysCount = 0;
char state;

void debug_plot(char *filename, int num, float *tobeplot, int length)
{
    FILE * fp;
    int i;

    char strnum[10];
    sprintf(strnum, "%d", num);

    /* open the file for writing*/
    char *s1 = "debug_plot/";
    //char *s1 = "";
    char *s2 = ".txt";

    char *result = malloc(strlen(s1) + strlen(filename) + strlen(s2) + 1); // +1 for the null-terminator

    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, filename);
    strcat(result, strnum);
    strcat(result, s2);

    fp = fopen(result,"w");

    /* write lines of text into the file stream*/
    for(i = 0; i < length; i++){
        fprintf(fp, "%f\n",tobeplot[i]);
    }

    /* close the file*/
    fclose (fp);
    free(result);
}

void make_network_CA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[17];
    passint[0] = n;
    passint[1] = time_steps;
    passint[2] = notruth;
    passint[3] = batch;
    passint[4] = subdivisions;
    passint[5] = random;
    passint[6] = adam;
    passint[7] = h;
    passint[8] = w;
    passint[9] = c;
    passint[10] = inputs;
    passint[11] = max_crop;
    passint[12] = min_crop;
    passint[13] = center;
    passint[14] = burn_in;
    passint[15] = max_batches;

    float passfloat[15];
    passfloat[0] = learning_rate;
    passfloat[1] = momentum;
    passfloat[2] = decay;
    passfloat[3] = B1;
    passfloat[4] = B2;
    passfloat[5] = eps;
    passfloat[6] = max_ratio;
    passfloat[7] = min_ratio;
    passfloat[8] = clip;
    passfloat[9] = angle;
    passfloat[10] = aspect;
    passfloat[11] = saturation;
    passfloat[12] = exposure;
    passfloat[13] = hue;
    passfloat[14] = power;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passfloat;
    op.params[1].tmpref.size = sizeof(passfloat);

    res = TEEC_InvokeCommand(&sess, MAKE_NETWORK_CMD,
                             &op, &origin);


    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
	// Send encrypted weights over the network
	exit(0);
    }
    
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_NET) failed 0x%x origin 0x%x",
         res, origin);
}

/*
void update_net_agrv_CA_allocateSM(int workspace_size, float *workspace)
{
    uint32_t origin;
    TEEC_Result res;
    workspaceSM.size  = sizeof(float) * workspace_size;
    workspaceSM.flags = TEEC_MEM_INPUT | TEEC_MEM_OUTPUT;

    res = TEEC_AllocateSharedMemory(
                     &ctx,
                     &workspaceSM);
     if (res != TEEC_SUCCESS)
     errx(1, "TEEC_InvokeCommand(UPDATE_NET_ASM) failed 0x%x origin 0x%x", res, origin);
}
*/

void update_net_agrv_CA(int cond, int workspace_size, float *workspace)
{
    // forward condition
    if(cond == 0)
    {
        TEEC_Operation op;
        uint32_t origin;
        TEEC_Result res;

        workspaceSM.buffer = workspace;

        memset(&op, 0, sizeof(op));
        op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_MEMREF_PARTIAL_INOUT,
                                         TEEC_NONE, TEEC_NONE);

        op.params[0].value.a = cond;
        op.params[1].memref.parent = &workspaceSM;
        op.params[1].memref.offset = 0;
        op.params[1].memref.size   = sizeof(float) * workspace_size;

        res = TEEC_InvokeCommand(&sess, WORKSPACE_NETWORK_CMD,
                                 &op, &origin);

         if (res != TEEC_SUCCESS)
         errx(1, "TEEC_InvokeCommand(UPDATE_NET) failed 0x%x origin 0x%x",
              res, origin);
    }

    // backward condition
    if(cond == 1){
        float *wsbuffer = workspaceSM.buffer;
        for(int z=0; z<workspace_size; z++){
              workspace[z] = wsbuffer[z];
        }
    }
}

void allocate_workspace_CA(int workspace_size)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int params0[1];
    params0[0] = workspace_size;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(params0);

    res = TEEC_InvokeCommand(&sess, ALLOCATE_WORKSPACE_CMD,
                             &op, &origin);

    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(loss) failed 0x%x origin 0x%x",
         res, origin);
}

void make_convolutional_layer_CA(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[15];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = n;
    passint[5] = groups;
    passint[6] = size;
    passint[7] = stride;
    passint[8] = padding;
    passint[9] = batch_normalize;
    passint[10] = binary;
    passint[11] = xnor;
    passint[12] = adam;
    passint[13] = flipped;

    float passflo = dot;
    char *acti = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].value.a = passflo;

    op.params[2].tmpref.buffer = acti;
    op.params[2].tmpref.size = strlen(acti)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_CONV_CMD,
                             &op, &origin);

    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x",
         res, origin);
}

void make_maxpool_layer_CA(int batch, int h, int w, int c, int size, int stride, int padding)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[7];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = size;
    passint[5] = stride;
    passint[6] = padding;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_MAX_CMD,
                             &op, &origin);

    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAX) failed 0x%x origin 0x%x",
         res, origin);
}

void make_dropout_layer_CA(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, float *net_prev_delta)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[5];
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = w;
    passint[3] = h;
    passint[4] = c;
    float passflo[1];
    passflo[0] = probability;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(float)*1;

////////////////////////
    //debug_plot("net_prev_output", sysCount, net_prev_output, inputs*batch);
    //debug_plot("net_prev_delta", sysCount, net_prev_delta, inputs*batch);

    op.params[2].tmpref.buffer = net_prev_output;
    op.params[2].tmpref.size = sizeof(float)*inputs*batch;
    op.params[3].tmpref.buffer = net_prev_delta;
    op.params[3].tmpref.size = sizeof(float)*inputs*batch;
////////////////////////

    res = TEEC_InvokeCommand(&sess, MAKE_DROP_CMD,
                             &op, &origin);

    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(DROP) failed 0x%x origin 0x%x",
         res, origin);
}



void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passarg[5];
    passarg[0] = batch;
    passarg[1] = inputs;
    passarg[2] = outputs;
    passarg[3] = batch_normalize;
    passarg[4] = adam;

    char *actv = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passarg;
    op.params[0].tmpref.size = sizeof(passarg);

    op.params[1].tmpref.buffer = actv;
    op.params[1].tmpref.size = strlen(actv)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_CONNECTED_CMD,
                             &op, &origin);

    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FC) failed 0x%x origin 0x%x",
         res, origin);
}

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[8];
    float passflo = temperature;
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = groups;
    passint[3] = w;
    passint[4] = h;
    passint[5] = c;
    passint[6] = spatial;
    passint[7] = noloss;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_VALUE_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].value.a = passflo;

    res = TEEC_InvokeCommand(&sess, MAKE_SOFTMAX_CMD,
                             &op, &origin);
    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(SOFTMAX) failed 0x%x origin 0x%x",
         res, origin);
}

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[2];
    float passflo[4];
    char *passcost;

    passint[0] = batch;
    passint[1] = inputs;
    passflo[0] = scale;
    passflo[1] = ratio;
    passflo[2] = noobject_scale;
    passflo[3] = thresh;

    passcost = get_cost_string(cost_type);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    op.params[2].tmpref.buffer = passcost;
    op.params[2].tmpref.size = strlen(passcost)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_COST_CMD,
                             &op, &origin);
    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(COST) failed 0x%x origin 0x%x",
         res, origin);
}

void transfer_weights_CA(float *vec, int length, int layer_i, char type, int additional)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    
    int passint[3];
    passint[0] = length;
    passint[1] = layer_i;
    passint[2] = additional;
    
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);
    
    op.params[0].tmpref.buffer = vec;
    op.params[0].tmpref.size = sizeof(float)*length;
    
    op.params[1].tmpref.buffer = passint;
    op.params[1].tmpref.size = sizeof(passint);
    
    op.params[2].value.a = type;
    
    res = TEEC_InvokeCommand(&sess, TRANS_WEI_CMD,
                             &op, &origin);
    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(TRANS_WEI) failed 0x%x origin 0x%x",
             res, origin);
}

void save_weights_CA(float *vec, int length, int layer_i, char type)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    
    int passint[2];
    passint[0] = length;
    passint[1] = layer_i;
    
    float *weights_back = malloc(sizeof(float) * length);
    
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);
    
    op.params[0].tmpref.buffer = weights_back;
    op.params[0].tmpref.size = sizeof(float) * length;
    
    op.params[1].tmpref.buffer = passint;
    op.params[1].tmpref.size = sizeof(passint);
    
    op.params[2].value.a = type;
    
    res = TEEC_InvokeCommand(&sess, SAVE_WEI_CMD,
                             &op, &origin);
    
    for(int z=0; z<length; z++){
         vec[z] = weights_back[z];
    }
    
    free(weights_back);
    
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(SAVE_WEI) failed 0x%x origin 0x%x",
             res, origin);
}

void forward_network_CA(float *net_input, int l_inputs, int net_batch, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

     float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
     for(int z=0; z<l_inputs*net_batch; z++){
         params0[z] = net_input[z];
     }
     int params1 = net_train;

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * l_inputs*net_batch;
    op.params[1].value.a = params1;

    /////////  debug_plot  /////////
    //debug_plot("forward_net_input_", sysCount, params0, l_inputs*net_batch);

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD,
                             &op, &origin);
    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void forward_softmax_CA(float *net_input, int l_inputs, int net_batch, int l_outputs, float *net_output)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = net_input;
    op.params[0].tmpref.size = sizeof(float) * l_inputs * net_batch;
    op.params[1].tmpref.buffer = net_tag_buffer;
    op.params[1].tmpref.size = 16;
    op.params[2].tmpref.buffer = net_output;
    op.params[2].tmpref.size = sizeof(float) * l_outputs * net_batch;

    res = TEEC_InvokeCommand(&sess, FORWARD_SOFTMAX_CMD,
                             &op, &origin);
    /*if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }*/

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);
}

void backward_network_CA_addidion(int net_inputs, int net_batch)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input_back = malloc(sizeof(float) * net_inputs*net_batch);
  net_delta_back = malloc(sizeof(float) * net_inputs*net_batch);


  /////////  debug_plot  /////////
  //debug_plot("backward_net_input_back1_", sysCount, net_input_back, net_inputs*net_batch);
  //debug_plot("backward_net_delta_back1_", sysCount, net_delta_back, net_inputs*net_batch);


  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_OUTPUT,
                                   TEEC_NONE, TEEC_NONE);



   op.params[0].tmpref.buffer = net_input_back;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;
   op.params[1].tmpref.buffer = net_delta_back;
   op.params[1].tmpref.size = sizeof(float) * net_inputs*net_batch;

   res = TEEC_InvokeCommand(&sess, BACKWARD_ADD_CMD,
                            &op, &origin);

    /////////  debug_plot  /////////
    //debug_plot("backward_net_input_back2_", sysCount, net_input_back, net_inputs*net_batch);
    //debug_plot("backward_net_delta_back2_", sysCount, net_delta_back, net_inputs*net_batch);

   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(backward_add) failed 0x%x origin 0x%x",
        res, origin);
}



void backward_network_CA(float *net_input, int l_inputs, int net_batch, float *net_delta, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;


    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_NONE);

     float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
     float *params1 = malloc(sizeof(float)*l_inputs*net_batch);

     for(int z=0; z<l_inputs*net_batch; z++){
         params0[z] = net_input[z];
         params1[z] = net_delta[z];
     }

    op.params[0].tmpref.buffer = params0; // as lta.output
    op.params[0].tmpref.size = sizeof(float)*l_inputs*net_batch;
    op.params[1].tmpref.buffer = params1; // as n_delta
    op.params[1].tmpref.size = sizeof(float)*l_inputs*net_batch;
    op.params[2].value.a = net_train;

    res = TEEC_InvokeCommand(&sess, BACKWARD_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(backward) failed 0x%x origin 0x%x",
         res, origin);

     /////////  debug_plot  /////////
    //debug_plot("backward_net_input_", sysCount, params0, l_inputs*net_batch);
    //debug_plot("backward_net_delta_", sysCount, params1, l_inputs*net_batch);

   free(params0);
   free(params1);
}



void update_network_CA(update_args a)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    passint[0] = a.batch;
    passint[1] = a.adam;
    passint[2] = a.t;

    float passflo[6];
    passflo[0] = a.learning_rate;
    passflo[1] = a.momentum;
    passflo[2] = a.decay;
    passflo[3] = a.B1;
    passflo[4] = a.B2;
    passflo[5] = a.eps;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    res = TEEC_InvokeCommand(&sess, UPDATE_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);
}



void net_truth_CA(float *net_truth, int net_truths, int net_batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    // allocate memory for transmitting truth
    float *params0 = malloc(sizeof(float) * net_truths * net_batch);

    for(int z=0; z<net_truths*net_batch; z++){
        params0[z] = net_truth[z];
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float)*net_truths*net_batch;

    res = TEEC_InvokeCommand(&sess, NET_TRUTH_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(truth) failed 0x%x origin 0x%x",
         res, origin);

     /////////  debug_plot  /////////
     //debug_plot("backward_net_truth_", sysCount, params0, net_truths*net_batch);

    free(params0);
}

void calc_network_loss_CA(int n, int batch)
{
    sysCount++;
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int params0[2];
    params0[0] = n;
    params0[1] = batch;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(params0);

    res = TEEC_InvokeCommand(&sess, CALC_LOSS_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(loss) failed 0x%x origin 0x%x",
         res, origin);
}

void net_output_return_CA(int net_outputs, int net_batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    
    net_output_back = malloc(sizeof(float) * net_outputs * net_batch);
    //memset(net_tag_buffer, 0 , 16);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE, TEEC_NONE);
    
    op.params[0].tmpref.buffer = net_output_back;
    op.params[0].tmpref.size = sizeof(float) * net_outputs * net_batch;

    op.params[1].tmpref.buffer = net_tag_buffer;
    op.params[1].tmpref.size = 16;
    
    res = TEEC_InvokeCommand(&sess, OUTPUT_RETURN_CMD,
                             &op, &origin);
    
    float *tem = op.params[0].tmpref.buffer;
   
    if(res == TEEC_ERROR_OUT_OF_MEMORY)
    {
        printf("Memory threshold exceeded. Offload!!\n");
        // Send encrypted weights over the network
        exit(0);
    }

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);
}

int verify_attestation_token(char *attestation_buffer, int token_size)
{
    int result = 0;
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE,
                                     TEEC_NONE);

    op.params[0].tmpref.buffer = attestation_buffer;
    op.params[0].tmpref.size = token_size * sizeof(char);

    op.params[1].tmpref.buffer = net_tag_buffer;
    op.params[1].tmpref.size = 16;

    res = TEEC_InvokeCommand(&sess, VERIFY_ATTESTATION_TOKEN,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);
	result = -1;
    }
    return result;
}

void get_secure_randomnumber(void)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    session_token = malloc(sizeof(uint32_t));

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_OUTPUT,
                                     TEEC_NONE,
                                     TEEC_NONE, 
				     TEEC_NONE);

    //op.params[0].tmpref.buffer = session_token;
    //op.params[0].tmpref.size = sizeof(uint32_t);

    res = TEEC_InvokeCommand(&sess, RAND_NUMBER_GENERATOR,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);

    session_token = op.params[0].value.a;
}

int remote_attestation(void)
{
    clock_t time1 = clock();
    clock_t time2;
    float t1;
    // Get a random number as session token from TA
    get_secure_randomnumber();
    printf("\nValue of session token: %d\n", session_token);

    char token[80];
    strcpy(token, "data/darknet/session_token.data");
    fprintf(stderr, "Saving partial outputs to %s\n", token);
    FILE *fp = fopen(token, "wb");
    if(!fp) file_error(token);

    fwrite(&session_token, sizeof(uint32_t), 1, fp);
    fclose(fp);

    // Send this token to SGX
    printf("\nSending session token to SGX\n");
    int status = 0;
    status = Socket_Client(token, NULL, 8000, 1);
    if(status != 0)
    {
        printf("\nFile sending failed...\n");
        exit(1);
    }
    printf("Done.....\n");

    t1 = sec(clock()-time1);
    // Wait for partial results from SGX
    printf("\nWait for encrypted UUID+token from SGX\n");
    status = Socket_Server("data/darknet/token_uuid_sgx.data", "data/darknet/token_size.txt", "data/darknet/token_tag_sgx.data", "data/darknet/token_tag_size.txt", &time2, 8000, 0);
    if(status != 0)
    {
        printf("\nFile receiving failed...\n");
        exit(1);
    }
    printf("Done....\n");

    // Verify the attested UUID
    char *tag_buffer = "data/darknet/token_tag_sgx.data";
    //char *tag_size = "data/darknet/token_tag_size.txt";
    char *uuid_token = "data/darknet/token_uuid_sgx.data";
    char *uuid_token_size = "data/darknet/token_size.txt";

    fprintf(stderr, "Loading token from %s...", uuid_token);
    FILE *fp1 = fopen(uuid_token_size, "rb");
    FILE *fp1_1 = fopen(uuid_token, "rb");
    if(fp1 == NULL)
    {
        file_error(uuid_token_size);
    }
    if(fp1_1 == NULL)
    {
        file_error(uuid_token);
    }

    int token_size;
    fread(&token_size, sizeof(int), 1, fp1);
    token_size = 46;
    printf("\nRead token with size: %d\n", token_size);
    
    // Allocate buffer for token
    char *attestation_buffer = malloc(sizeof(char) * token_size);
    fread(attestation_buffer, sizeof(char), token_size, fp1_1);

    net_tag_buffer = (uint8_t *)calloc(1,16);    // 16 byte tag for encryption
    FILE *fp0 = fopen(tag_buffer, "rb");
    if(fp0 == NULL)
    {
        file_error(tag_buffer);
    }
    fread(net_tag_buffer, 1, 16, fp0);
    
    fclose(fp0);
    fclose(fp1);
    fclose(fp1_1);

    status = verify_attestation_token(attestation_buffer, token_size);
    if(status != 0)
    {
        printf("\nRemote attestation verification failed....\n");
    }
    else
    {
        printf("\nRemote attestation verification successful....\n");
    }

    fprintf(stderr, "\nAttestation process completed in %f seconds.\n", t1 + sec(clock()-time2));
    
    free(net_tag_buffer);
    free(attestation_buffer);
    return status;
}

void prepare_tee_session()
{
    TEEC_UUID uuid = TA_DARKNETP_UUID;
    uint32_t origin;
    TEEC_Result res;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, origin);
}

void terminate_tee_session()
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}

int main(int argc, char **argv)
{

    printf("Prepare session with the TA\n");
    prepare_tee_session();

    // Remote attestation
    int result = remote_attestation();

    if(result == 0)
    {
        printf("Begin darknet\n");
        darknet_main(argc, argv);
    }
    else
    {
        printf("Remote attestation failed....!!!\n");
    }

    terminate_tee_session();
    return 0;
}
