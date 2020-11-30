#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"
#include "stdlib.h"
#include <string.h>
#include <stdio.h>

#include "darknetp_ta.h"
#include "activations_TA.h"
#include "aes_TA.h"
#include "blas_TA.h"
#include "darknet_TA.h"
#include "gemm_TA.h"
#include "im2col_TA.h"
#include "math_TA.h"
#include "utils_TA.h"
#include "Enclave.h"
#include "aes.h"
#include <math.h>
#include <float.h>
#include <string.h>

#define ENCRYPT_OUTPUT_DATA  1
/*
 *@brief      Performs a simple addition in the enclave
 *
 * @param      a      The first input for our simple addition
 * @param      b      The second input for our simple addition
 *
 * @return     Truthy if addition successful, falsy otherwise.
 */

network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;
float *netta_truth;
int netnum = 0;
int ta_start_index = 1;

float gGlobalBufferCounter;
uint8_t *ta_tag_buffer;

// Unique identifier for SGX - 36 characters for UUID + 10 characters for token
char UID[46] = "b8734d20-832e-43a9-bf9f-17f3e9e041e2";

sgx_status_t decrypt_data(uint8_t *encrypt_In, size_t len, uint8_t *decrypt_Out, size_t lenOut)
{
    //emit_debug((char *) p_dst);
   
    return SGX_SUCCESS;
}

int ecall_make_network(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches, int start_index)
{
    netta.n = n;

    netta.seen = (size_t *)calloc(1, sizeof(size_t));
    netta.layers = (layer_TA *)calloc(netta.n, sizeof(layer_TA));
    netta.t    = (int *)calloc(1, sizeof(int));
    netta.cost = (float *)calloc(1, sizeof(float));

    netta.learning_rate = learning_rate;
    netta.momentum = momentum;
    netta.decay = decay;
    netta.time_steps = time_steps;
    netta.notruth = notruth;
    netta.batch = batch;
    netta.subdivisions = subdivisions;
    netta.random = random;
    netta.adam = adam;
    netta.B1 = B1;
    netta.B2 = B2;
    netta.eps = eps;
    netta.h = h;
    netta.w = w;
    netta.c = c;
    netta.inputs = inputs;
    netta.max_crop = max_crop;
    netta.min_crop = min_crop;
    netta.max_ratio = max_ratio;
    netta.min_ratio = min_ratio;
    netta.center = center;
    netta.clip = clip;
    netta.angle = angle;
    netta.aspect = aspect;
    netta.saturation = saturation;
    netta.exposure = exposure;
    netta.hue = hue;
    netta.burn_in = burn_in;
    netta.power = power;
    netta.max_batches = max_batches;

    ta_start_index = start_index;
    ocall_print("\nEnclave start index: \n",(float)ta_start_index);

    gGlobalBufferCounter += (sizeof(size_t) + sizeof(int) + sizeof(float) + (netta.n * sizeof(layer_TA)));
    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);
    return 0; 
}

void binarize_weights_TA(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void swap_binary_TA(layer_TA *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}

void binarize_cpu_TA(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void add_bias_TA(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            register int b_val = (b*n + i)*size;
	    #pragma omp parallel for
            for(j = 0; j < size; ++j){
                //output[(b*n + i)*size + j] += biases[i];
		output[b_val + j] += biases[i];
            }
        }
    }
}

void scale_bias_TA(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            register int b_val = (b*n + i)*size;
	    #pragma omp parallel for
            for(j = 0; j < size; ++j){
                //output[(b*n + i)*size + j] *= scales[i];
		output[b_val + j] *= scales[i];
            }
        }
    }
}

void forward_batchnorm_layer_enclave(layer_TA l, network_TA net)
{
    if(l.type == BATCHNORM_TA) copy_cpu_TA(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu_TA(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu_TA(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu_TA(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu_TA(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu_TA(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu_TA(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu_TA(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu_TA(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
        copy_cpu_TA(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu_TA(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias_TA(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias_TA(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void forward_convolutional_layer_enclave(layer_TA l, network_TA net)
{
    int i, j;

    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights_TA(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary_TA(&l);
        binarize_cpu_TA(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_TA(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
	    //ocall_print("\n\nFirst output value\n\n",5555.0);
            gemm_TA(0,0,m,n,k,1,a,k,b,n,1,c,n);
	    
        }
    }

    gGlobalBufferCounter += (5*sizeof(int) + 16);

    if(l.batch_normalize){
        forward_batchnorm_layer_enclave(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary_TA(&l);

    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);
    /* 
    ocall_print("\nConv layer outputs",555.0);
    ocall_print("\n",555.0);
    for(int p = 0; p < 500; p++)
        ocall_print(" ",l.output[p]);
    ocall_print("\n\n",555.0);
    */
    
}

int ecall_make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, char *activation_s, int acti_length, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
    int i;
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));
    l.type = CONVOLUTIONAL_TA;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = (float *)calloc(c/groups*n*size*size, sizeof(float));
    //l.weight_updates = (float *)calloc(c/groups*n*size*size, sizeof(float));

    l.biases = (float *)calloc(n, sizeof(float));
    //l.bias_updates = (float *)calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    ACTIVATION_TA activation = get_activation_TA(activation_s);
    // float scale = 1./sqrt(size*size*c);
    float scale = ta_sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    /*
    for(i = 0; i < l.nweights; ++i) {
        l.weights[i] = scale*rand_normal_TA(0,1);
    }
    */

    int out_w = (l.w + 2*l.pad - l.size) / l.stride + 1;
    int out_h = (l.h + 2*l.pad - l.size) / l.stride + 1;
    
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (float *)calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = (float *)calloc(l.batch*l.outputs, sizeof(float));

    l.forward_TA = forward_convolutional_layer_enclave;
    //l.backward_TA = backward_convolutional_layer_TA_new;
    //l.update_TA = update_convolutional_layer_TA_new;
    l.backward_TA = NULL;
    l.update_TA = NULL;
    if(binary){
        l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
        l.cweights = (char *)calloc(l.nweights, sizeof(char));
        l.scales = (float *)calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
        l.binary_input = (float *)calloc(l.inputs*l.batch, sizeof(float));
    }

    gGlobalBufferCounter += (((c/groups*n*size*size) + n + (2*l.batch*l.outputs)) * sizeof(float));

    if(batch_normalize){
        l.scales = (float *)calloc(n, sizeof(float));
        //l.scale_updates = (float *)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float *)calloc(n, sizeof(float));
        l.variance = (float *)calloc(n, sizeof(float));

        //l.mean_delta = (float *)calloc(n, sizeof(float));
        //l.variance_delta = (float *)calloc(n, sizeof(float));

        l.rolling_mean = (float *)calloc(n, sizeof(float));
        l.rolling_variance = (float *)calloc(n, sizeof(float));
        l.x = (float *)calloc(l.batch*l.outputs, sizeof(float));
        //l.x_norm = (float *)calloc(l.batch*l.outputs, sizeof(float));
	
	gGlobalBufferCounter += ((5*n + l.batch*l.outputs) * sizeof(float));
    }
    if(adam){
        l.m = (float *)calloc(l.nweights, sizeof(float));
        l.v = (float *)calloc(l.nweights, sizeof(float));
        l.bias_m = (float *)calloc(n, sizeof(float));
        l.scale_m = (float *)calloc(n, sizeof(float));
        l.bias_v = (float *)calloc(n, sizeof(float));
        l.scale_v = (float *)calloc(n, sizeof(float));
    }

    l.workspace_size = (size_t)(l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float));
    l.activation = activation;

    l.flipped = flipped;
    l.dot = dot;

    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);

//IMSG("conv_TA%4d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    /*
    if(l.workspace_size)
    {
        netta.workspace = (float *)calloc(1,l.workspace_size);
    }*/

    netta.layers[netnum] = l;
    netnum++;
    return 0;
}

void forward_connected_layer_enclave(layer_TA l, network_TA net)
{
    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;

    gemm_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if(l.batch_normalize){
        forward_batchnorm_layer_enclave(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
}

int ecall_make_connected_layer(int batch, int inputs, int outputs, char *activation_s, int acti_length, int batch_normalize, int adam)
{
    int i;
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));
    l.learning_rate_scale = 1;
    l.type = CONNECTED_TA;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = (float *)calloc(batch*outputs, sizeof(float));
    l.delta = (float *)calloc(batch*outputs, sizeof(float));

    l.weight_updates = (float *)calloc(inputs*outputs, sizeof(float));
    l.bias_updates = (float *)calloc(outputs, sizeof(float));

    l.weights = (float *)calloc(outputs*inputs, sizeof(float));
    l.biases = (float *)calloc(outputs, sizeof(float));

    l.forward_TA = forward_connected_layer_enclave;
    //l.backward_TA = backward_connected_layer_TA_new;
    //l.update_TA = update_connected_layer_TA_new;
    l.backward_TA = NULL;
    l.update_TA = NULL;

    ACTIVATION_TA activation = get_activation_TA(activation_s);

    //float scale = 1./sqrt(inputs);
    float scale = ta_sqrt(2./inputs);
    /*
    for(i = 0; i < outputs*inputs; ++i){
        //l.weight_updates[i] = 1.0f;
        l.weights[i] = scale * rand_uniform_TA(-1, 1);
    }
    */

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = (float *)calloc(l.inputs*l.outputs, sizeof(float));
        l.v = (float *)calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = (float *)calloc(l.outputs, sizeof(float));
        l.scale_m = (float *)calloc(l.outputs, sizeof(float));
        l.bias_v = (float *)calloc(l.outputs, sizeof(float));
        l.scale_v = (float *)calloc(l.outputs, sizeof(float));
    }

    if(batch_normalize){
        l.scales = (float *)calloc(outputs, sizeof(float));
        l.scale_updates = (float *)calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float *)calloc(outputs, sizeof(float));
        l.mean_delta = (float *)calloc(outputs, sizeof(float));
        l.variance = (float *)calloc(outputs, sizeof(float));
        l.variance_delta = (float *)calloc(outputs, sizeof(float));

        l.rolling_mean = (float *)calloc(outputs, sizeof(float));
        l.rolling_variance = (float *)calloc(outputs, sizeof(float));

        l.x = (float *)calloc(batch*outputs, sizeof(float));
        l.x_norm = (float *)calloc(batch*outputs, sizeof(float));
    }

    l.activation = activation;
    //IMSG("connected_TA                         %4d  ->  %4d\n", inputs, outputs);
  
    netta.layers[netnum] = l;
    netnum++;
    return 0;
}

void forward_softmax_layer_enclave(const layer_TA l, network_TA net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu_TA(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu_TA(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu_TA(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array_TA(l.loss, l.batch*l.inputs);
    }
}

int ecall_make_softmax_layer(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    assert(0); // Should not create the softmax layer in SGX
    assert(inputs%groups == 0);
    //IMSG("softmax_TA                                     %4d\n",  inputs);
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));

    l.type = SOFTMAX_TA;
    l.batch = batch;
    l.groups = groups;

    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float *)calloc(inputs*batch, sizeof(float));
    l.output = (float *)calloc(inputs*batch, sizeof(float));
    l.delta = (float *)calloc(inputs*batch, sizeof(float));
    l.cost = (float *)calloc(1, sizeof(float));

    l.temperature = temperature;
    l.w = w;
    l.h = h;
    l.c = c;
    l.spatial = spatial;
    l.noloss = noloss;

    l.forward_TA = forward_softmax_layer_enclave;
    //l.backward_TA = backward_softmax_layer_TA;
    l.backward_TA = NULL;

    netta.layers[netnum] = l;
    netnum++;
    return 0;
}

COST_TYPE_TA get_cost_type_TA(char *s)
{   
    if (strcmp(s, "seg")==0) return SEG_TA;
    if (strcmp(s, "sse")==0) return SSE_TA;
    if (strcmp(s, "masked")==0) return MASKED_TA;
    if (strcmp(s, "smooth")==0) return SMOOTH_TA;
    if (strcmp(s, "L1")==0) return L1_TA;
    if (strcmp(s, "wgan")==0) return WGAN_TA;
    //snprintf("Couldn't find cost type %s, going with SSE\n", s);
    return SSE_TA;
}

void forward_cost_layer_enclave(layer_TA l, network_TA net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED_TA){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM_TA) net.input[i] = SECRET_NUM_TA;
        }
    }
    if(l.cost_type == SMOOTH_TA){
        smooth_l1_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1_TA){
        l1_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array_TA(l.output, l.batch*l.inputs);
}

int ecall_make_cost_layer(int batch, int inputs, char *cost_type, int cost_size, float scale, float ratio, float noobject_scale, float thresh)
{
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));
    l.type = COST_TA;  
                            
    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    COST_TYPE_TA cost_type_ta = get_cost_type_TA(cost_type);
    l.cost_type = cost_type_ta;
    l.delta = (float *)calloc(inputs*batch, sizeof(float));
    l.output = (float *)calloc(inputs*batch, sizeof(float));
    l.cost = (float *)calloc(1, sizeof(float));

    l.scale = scale;
    l.ratio = ratio;
    l.noobject_scale = noobject_scale;
    l.thresh = thresh;

    l.forward_TA = forward_cost_layer_enclave;
    //l.backward_TA = backward_cost_layer_TA;
    l.backward_TA = NULL;

    netta.layers[netnum] = l;
    netnum++;
    return 0;
}

void forward_maxpool_layer_enclave(const layer_TA l, network_TA net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -first_aim_money;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -first_aim_money;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }

    gGlobalBufferCounter += (12*sizeof(int) + 2*sizeof(float));
    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);
    /*
    ocall_print("\nMaxpool layer outputs",555.0);
    ocall_print("\n",555.0);
    for(int p = 0; p < 500; p++)
        ocall_print(" ",l.output[p]);
    ocall_print("\n\n",555.0);
    */
}

int ecall_make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));

    l.type = MAXPOOL_TA;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int *)calloc(output_size, sizeof(int));
    l.output =  (float *)calloc(output_size, sizeof(float));
    l.delta =   (float *)calloc(output_size, sizeof(float));
    l.forward_TA = forward_maxpool_layer_enclave;
    //l.backward_TA = backward_maxpool_layer_TA_new;
    l.backward_TA = NULL;

    gGlobalBufferCounter += (output_size * (sizeof(int) + 2*sizeof(float)));
    //IMSG("max_TA       %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);

    netta.layers[netnum] = l;
    netnum++;
    return 0;   
}

void forward_dropout_layer_enclave(layer_TA l, network_TA net)
{
    int i;
    if (!net.train) return;

    float *pter;
    if(l.netnum == 0){
        for(i = 0; i < l.batch * l.inputs; ++i){
            l.output[i] = net.input[i];
        }

        pter = l.output;
    }else{
        pter = net.input;
    }

    for(i = 0; i < l.batch * l.inputs; ++i){
        //printf("i = %d; total = %d\n",i, l.batch * l.inputs);
        float r = rand_uniform_TA(0, 1);
        l.rand[i] = r;
        if(r < l.probability)   pter[i] = 0;
        else    pter[i] *= l.scale;
    }
}

int ecall_make_dropout_layer(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, int prev_size, float *net_prev_delta, int delta_size)
{
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));
    int size = prev_size/sizeof(float);

    l.type = DROPOUT_TA;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float *)calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);

    l.netnum = netnum;

    l.output = (float *)malloc(sizeof(float) * inputs*batch);
    l.delta = (float *)malloc(sizeof(float) * inputs*batch); 

    l.forward_TA = forward_dropout_layer_enclave;
    //l.backward_TA = backward_dropout_layer_TA_new;
    l.backward_TA = NULL;
    l.w = w;
    l.h = h;
    l.c = c;

    //char prob[20];
    //ftoa(probability,prob,3);
    //IMSG("dropout_TA    p = %s               %4d  ->  %4d\n", prob, inputs, inputs);

    if(netnum == 0){
      for(int z=0; z<size; z++){             // Check and update
        l.output[z] = net_prev_output[z];
        l.delta[z] = net_prev_delta[z];
      }
    }else{
        l.output = netta.layers[netnum-1].output;
        l.delta = netta.layers[netnum-1].delta;
    }

    netta.layers[netnum] = l;
    netnum++;
    return 0;
}

void forward_avgpool_layer_enclave(const layer_TA l, network_TA net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }

    /*
    ocall_print("\nAvg pool layer outputs",555.0);
    ocall_print("\n",555.0);
    for(int i = 0; i < l.batch * l.c; i++)
        ocall_print(" ",l.output[i]);
    ocall_print("\n\n",555.0);
    */
}

int ecall_make_avgpool_layer(int batch, int w, int h, int c)
{
    layer_TA l;
    //memset((void *)&l, 0, sizeof(layer_TA));
    l.type = AVGPOOL_TA;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  (float *)calloc(output_size, sizeof(float));
    l.delta =   (float *)calloc(output_size, sizeof(float));
    l.forward_TA = forward_avgpool_layer_enclave;
    l.backward_TA = NULL;

    gGlobalBufferCounter += (2*output_size*sizeof(float));
    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);

    netta.layers[netnum] = l;
    netnum++;
    return 0;  
}

int ecall_allocate_workspace(int workspace_size)
{
    netta.workspace = (float *)calloc(1, workspace_size);
    gGlobalBufferCounter += workspace_size;
    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
    //ocall_print("\n",-1.0);
    return 0;
}

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
                ocall_print("Encryption failed\n",(float)result);
	    }
	    else{
                ocall_print("Decryption failed\n",(float)result);    
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

int aes_cbc_TA_token(char* xcrypt, char* UID, int len)
{
    int result = aes_gcm_entry(xcrypt, (uint8_t *)UID, ta_tag_buffer, len);
    if(result != 0)
    {
        if(strncmp(xcrypt, "encrypt", 2) == 0){
            ocall_print("Encryption failed\n",(float)result);
        }
        else{
            ocall_print("Decryption failed\n",(float)result);
        }
    }
    
    return result;
}

void transpose_matrix_TA(float *a, int rows, int cols)
{
    float *transpose = (float *)calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

int ecall_transfer_weights(float *vec, int length, int layer_i, char type, int additional)
{
    int size = length/sizeof(float);
    // decrypt
    float *tempvec = (float *)calloc(1,length);
    if(tempvec == NULL)
    {
        ocall_print("Unable to allocate memory",-1.0);
    	return 1;
    }
    copy_cpu_TA(size, vec, 1, tempvec, 1);
    //int result = aes_cbc_TA((char *)"decrypt", tempvec, size, 0);

    layer_TA l = netta.layers[layer_i - ta_start_index];

    if(type == 'b'){
        copy_cpu_TA(size, tempvec, 1, l.biases, 1);
    }
    else if(type == 'w'){
        copy_cpu_TA(size, tempvec, 1, l.weights, 1);
    }
    else if(type == 's'){
        copy_cpu_TA(size, tempvec, 1, l.scales, 1);
    }
    else if(type == 'm'){
        copy_cpu_TA(size, tempvec, 1, l.rolling_mean, 1);
    }
    else if(type == 'v'){
        copy_cpu_TA(size, tempvec, 1, l.rolling_variance, 1);
    }


    if(l.type == CONVOLUTIONAL_TA || l.type == DECONVOLUTIONAL_TA){
        if(l.flipped && type == 'w'){
            transpose_matrix_TA(l.weights, l.c*l.size*l.size, l.n);
        }
    }
    else if(l.type == CONNECTED_TA){
        if(additional && type == 'w'){
            transpose_matrix_TA(l.weights, l.inputs, l.outputs);
        }
    }

    free(tempvec);
    return 0;
}

int ecall_net_output_return(float *net_output, int length, uint8_t *tag_buffer, int tag_length)
{
    // remove confidence scores
    //float * rm_conf[length];
    //float maxconf; maxconf = -0.1f;
    //int maxidx; maxidx = 0;
    int size = length / sizeof(float);

    /*
    for(int z=0; z<size; z++){
        if(ta_net_output[z] > maxconf){
            maxconf = ta_net_output[z];
            maxidx = z;
        }
        ta_net_output[z] = 0.0f;
    }
    ta_net_output[maxidx] = 1.0f;
    */

    memset(ta_tag_buffer, 0, tag_length);
    ocall_print("Encrypt the partial results\n",0.0);
    // Encrypt the partial results
    int result = aes_cbc_TA((char *)"encrypt", ta_net_output, size, 1);
    ocall_print("Done...\n",0.0);

    if(result == 0)
    {
        for(int z=0; z<size; z++){
            net_output[z] = ta_net_output[z];
        }

	for(int z=0; z<tag_length; z++){
            tag_buffer[z] = ta_tag_buffer[z];
        }
    }

    free(ta_net_output);
    free(ta_tag_buffer);
    return result;
}

void calc_network_cost_enclave()
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < netta.n; ++i){
        if(netta.layers[i].cost){
            sum += netta.layers[i].cost[0];
            ++count;
        }
    }
    *netta.cost = sum/count;
    err_sum += *netta.cost;
}

int ecall_forward_network(float *net_input, int l_inputs, uint8_t *tag_buffer, int l_tag, int net_train)
{
    netta.input = net_input;
    netta.train = net_train;

    // Decrypt the input data
    // Allocate tag memory
    ta_tag_buffer = (uint8_t *)calloc(1, l_tag);
    memcpy(ta_tag_buffer, tag_buffer, l_tag);
    ocall_print("Decrypt the partial results\n",0.0);
    int result = aes_cbc_TA((char *)"decrypt", net_input, (l_inputs/sizeof(float)), 1);
    ocall_print("Done...\n",0.0);

    /*
    ocall_print("\nDecrypted output\n",555.0);
    for(int j = 0; j < 50; j++)
    {
        ocall_print(" ",net_input[j]);
    }
    ocall_print("\n",555.0);
    */

    roundnum++;
    int i;
    for(i = 0; (i < netta.n) && (result == 0); ++i){
        netta.index = i;
        layer_TA l = netta.layers[i];

        if(l.delta){
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward_TA(l, netta);

         netta.input = l.output;

        if(l.truth) {
            netta.truth = l.output;
        }
        //output of the network (for predict)
        //if(l.type == SOFTMAX_TA){
	if(i == netta.n - 1){
            ta_net_output = (float *)malloc(sizeof(float)*l.outputs*1);

	    gGlobalBufferCounter += sizeof(float)*l.outputs;
	    //ocall_print("Memory usage in MB = ", (float)gGlobalBufferCounter/(1024*1024));
            //ocall_print("\n",-1.0);
            for(int z=0; z<l.outputs*1; z++){
                ta_net_output[z] = l.output[z];
            }
        }
    }

    //calc_network_cost_enclave();
    return result;
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

void str_concat(char* str1, char* str2, uint8_t len)
{
    for(int i = 0; i < len; i++)
    {
        str1[i] = str2[i];
    }
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

int ecall_attest_session_token(int session_token, char *attestation_buffer, int attest_size, uint8_t *tag_buffer, int l_tag)
{
    int total_count = 10;
    int size_uuid = 36;
    char *str = &UID[size_uuid];
    int num_of_digits = num_of_digits_int(session_token);
    //char token[10];
    //snprintf(str, sizeof(str), "%d", session_token);
  
    //ocall_print("\nSession token: ",session_token); 
    // Concatenate the session token to the UUID
    getDecStr(str, num_of_digits, total_count, session_token);
    //getDecStr(token, num_of_digits, total_count, session_token);
    //str_concat(str, token, total_count);

    ocall_print("Session token + UUID\n",-1);
    ocall_print(UID,-555.0);

    // Encrypt session token + UUID
    ta_tag_buffer = (uint8_t *)calloc(1, l_tag);
    ocall_print("Encrypt and hash the session token + uuid \n",0.0);
    int result = aes_cbc_TA_token((char *)"encrypt", UID, sizeof(UID));
    ocall_print("Done...\n",0.0);

    if(result == 0)
    {
        memcpy(attestation_buffer, UID, sizeof(UID));
        for(int z=0; z<l_tag; z++){
            tag_buffer[z] = ta_tag_buffer[z];
        }
    }
    else
    {
        ocall_print(" Encryption failed \n",0.0);
        result = -1;
    }

    free(ta_tag_buffer);
    return result;
}

int ecall_decrypt(uint32_t *encrypt_input, uint32_t input_length, uint32_t *plaintext_output, uint32_t output_length, uint8_t *key, uint32_t key_size)
{
    sgx_status_t status = decrypt_data((uint8_t *)encrypt_input, input_length, (uint8_t *)plaintext_output, output_length);
    if(status != SGX_SUCCESS)
    {
        ocall_print("Decryption failed...!!!",1.0);
        return 1;
    }
    return 0;
}

int ecall_encrypt(uint32_t *plaintext_input, uint32_t input_length, uint32_t *encrypted_output, uint32_t output_length, uint8_t *key, uint32_t key_size)
{
    return 0;
}

