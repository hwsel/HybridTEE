#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "Enclave_u.h"
#include "App.h"
#include "darknet.h"
//#include "activation_layer.h"
#include "activations.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "dropout_layer.h"
#include "list.h"
#include "maxpool_layer.h"
#include "option_list.h"
#include "parser.h"
#include "softmax_layer.h"
#include "avgpool_layer.h"
#include "utils.h"
#include "socket_api.h"

//extern sgx_enclave_id_t global_eid;
int count_global = 0;
int global_dp = 0;

typedef struct{
    char *type;
    list *options;
}section;

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

static size_t get_workspace_size(layer l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int index)
{
    int i;
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = CONVOLUTIONAL;

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
    l.weight_updates = (float *)calloc(c/groups*n*size*size, sizeof(float));

    l.biases = (float *)calloc(n, sizeof(float));
    l.bias_updates = (float *)calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) {
        l.weights[i] = scale*rand_normal();
        //printf("rand_normal()=%f\n",rand_normal());
    }

    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (float *)calloc(l.batch*l.outputs, sizeof(float));

        l.delta  = (float *)calloc(l.batch*l.outputs, sizeof(float));

    //l.forward = forward_convolutional_layer;
    //l.backward = backward_convolutional_layer;
    //l.update = update_convolutional_layer;
    l.forward = NULL;
    l.backward = NULL;
    l.update = NULL;
    if(binary){
        l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
        l.cweights = (char *)calloc(l.nweights, sizeof(char));
        l.scales = (float *)calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
        l.binary_input = (float *)calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = (float *)calloc(n, sizeof(float));
        l.scale_updates = (float *)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float *)calloc(n, sizeof(float));
        l.variance = (float *)calloc(n, sizeof(float));

        l.mean_delta = (float *)calloc(n, sizeof(float));
        l.variance_delta = (float *)calloc(n, sizeof(float));

        l.rolling_mean = (float *)calloc(n, sizeof(float));
        l.rolling_variance = (float *)calloc(n, sizeof(float));
        l.x = (float *)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float *)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = (float *)calloc(l.nweights, sizeof(float));
        l.v = (float *)calloc(l.nweights, sizeof(float));
        l.bias_m = (float *)calloc(n, sizeof(float));
        l.scale_m = (float *)calloc(n, sizeof(float));
        l.bias_v = (float *)calloc(n, sizeof(float));
        l.scale_v = (float *)calloc(n, sizeof(float));
    }
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    if(index >= global_start_index)
    {
        fprintf(stderr, "conv_TA%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
    }
    return l;
}

int debug_num = 0;

/*
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
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
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }



    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);

    debug_num++;
}*/

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, (char *)"filters",1);
    int size = option_find_int(options, (char *)"size",1);
    int stride = option_find_int(options, (char *)"stride",1);
    int pad = option_find_int_quiet(options, (char *)"pad",0);
    int padding = option_find_int_quiet(options, (char *)"padding",0);
    int groups = option_find_int_quiet(options, (char *)"groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, (char *)"activation", (char *)"logistic");
    ACTIVATION activation = get_activation(activation_s);
    int acti_len = strlen(activation_s) + 1;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, (char *)"batch_normalize", 0);
    int binary = option_find_int_quiet(options, (char *)"binary", 0);
    int xnor = option_find_int_quiet(options, (char *)"xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam, params.index);
    layer.flipped = option_find_int_quiet(options, (char *)"flipped", 0);
    layer.dot = option_find_float_quiet(options, (char *)"dot", 0);

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
	int ptr;
        ecall_make_convolutional_layer(global_eid, &ptr,batch,h,w,c,n,groups,size,stride,padding,activation_s,acti_len,batch_normalize, binary, xnor, params.net->adam, layer.flipped, layer.dot);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }
    return layer;
}

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam, int index)
{
    int i;
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

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

    //l.forward = forward_connected_layer;
    //l.backward = backward_connected_layer;
    //l.update = update_connected_layer;
    l.forward = NULL;
    l.backward = NULL;
    l.update = NULL;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

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

    if(index >= global_start_index)
    {
        fprintf(stderr, "connected_TA                         %4d  ->  %4d\n", inputs, outputs);
    }
    return l;
}

/*
void forward_connected_layer(layer l, network net)
{

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);


    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}*/

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, (char *)"output",1);
    char *activation_s = option_find_str(options, (char *)"activation", (char *)"logistic");
    ACTIVATION activation = get_activation(activation_s);
    int acti_len = strlen(activation_s) + 1;
    int batch_normalize = option_find_int_quiet(options, (char *)"batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam, params.index);

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
	int ptr;
        ecall_make_connected_layer(global_eid, &ptr, params.batch, params.inputs, output, activation_s, acti_len, batch_normalize, params.net->adam);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }
    return l;
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);

    if(0)
    {
        fprintf(stderr, "softmax_TA                                     %4d\n",  inputs);
    }
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float *)calloc(inputs*batch, sizeof(float));
    l.output = (float *)calloc(inputs*batch, sizeof(float));
    l.delta = (float *)calloc(inputs*batch, sizeof(float));
    l.cost = (float *)calloc(1, sizeof(float));

    //l.forward = forward_softmax_layer;
    //l.backward = backward_softmax_layer;
    l.forward = NULL;
    l.backward = NULL;

    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, (char *)"groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, (char *)"temperature", 1);
    char *tree_file = option_find_str(options, (char *)"tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, (char *)"spatial", 0);
    l.noloss =  option_find_int_quiet(options, (char *)"noloss", 0);

    // Do not create the softmax layer in SGX. Final prediction will be done locally on the trustzone device
    if(0)
    {
        int ptr;
        ecall_make_softmax_layer(global_eid, &ptr, params.batch, params.inputs, groups, l.temperature, l.w, l.h, l.c, l.spatial, l.noloss);
    }

    return l;
}


COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, (char *)"seg")==0) return SEG;
    if (strcmp(s, (char *)"sse")==0) return SSE;
    if (strcmp(s, (char *)"masked")==0) return MASKED;
    if (strcmp(s, (char *)"smooth")==0) return SMOOTH;
    if (strcmp(s, (char *)"L1")==0) return L1;
    if (strcmp(s, (char *)"wgan")==0) return WGAN;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return (char *)"seg";
        case SSE:
            return (char *)"sse";
        case MASKED:
            return (char *)"masked";
        case SMOOTH:
            return (char *)"smooth";
        case L1:
            return (char *)"L1";
        case WGAN:
            return (char *)"wgan";
    }
    return (char *)"sse";
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale, int index)
{
    if(index >= global_start_index)
    {
        fprintf(stderr, "cost_TA                                        %4d\n",  inputs);
    }
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = (float *)calloc(inputs*batch, sizeof(float));
    l.output = (float *)calloc(inputs*batch, sizeof(float));
    l.cost = (float *)calloc(1, sizeof(float));

    //l.forward = forward_cost_layer;
    //l.backward = backward_cost_layer;
    l.forward = NULL;
    l.backward = NULL;
    return l;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, (char *)"type", (char *)"sse");
    COST_TYPE type = get_cost_type(type_s);
    int cost_size = strlen(type_s) + 1;
    float scale = option_find_float_quiet(options, (char *)"scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale, params.index);
    layer.ratio =  option_find_float_quiet(options, (char *)"ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, (char *)"noobj", 1);
    layer.thresh =  option_find_float_quiet(options, (char *)"thresh",0);

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
        int ptr;
        ecall_make_cost_layer(global_eid, &ptr, params.batch, params.inputs, type_s, cost_size, scale, layer.ratio, layer.noobject_scale, layer.thresh);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }

    return layer;
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int index)
{
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = MAXPOOL;
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
    //l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
    l.forward = NULL;
    l.backward = NULL;    

    if(index >= global_start_index)
    {
        fprintf(stderr, "max_TA       %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    return l;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, (char *)"stride",1);
    int size = option_find_int(options, (char *)"size",stride);
    int padding = option_find_int_quiet(options, (char *)"padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding,params.index);

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
        int ptr;
        ecall_make_maxpool_layer(global_eid, &ptr, batch,h,w,c,size,stride,padding);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }

    return layer;
}

dropout_layer make_dropout_layer(int batch, int inputs, float probability, int index)
{
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float *)calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    //l.forward = forward_dropout_layer;
    //l.backward = backward_dropout_layer;
    l.forward = NULL;
    l.backward = NULL;

    if(index >= global_start_index)
    {
        fprintf(stderr, "dropout_TA    p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    }
    return l;
}

dropout_layer parse_dropout(list *options, size_params params, float *net_prev_output, float *net_prev_delta)
{
    float probability = option_find_float(options, (char *)"probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability, params.index);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;

    layer.output = net_prev_output;
    layer.delta = net_prev_delta;

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
        int ptr;
        ecall_make_dropout_layer(global_eid, &ptr, params.batch, params.inputs, probability, params.w, params.h, params.c, net_prev_output, sizeof(float)*params.inputs*params.batch, net_prev_delta, sizeof(float)*params.inputs*params.batch);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }

    return layer;
}

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c, int index)
{
    if(index >= global_start_index)
    {
        fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    }
    layer l;
    memset((void *)&l, 0, sizeof(layer));
    l.type = AVGPOOL;
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
    l.forward = NULL;
    l.backward = NULL;
    return l;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c,params.index);

    if(params.index >= global_start_index)
    {
        clock_t time = clock();
        int ptr;
        ecall_make_avgpool_layer(global_eid, &ptr, batch, w, h, c);
        fprintf(stderr, "\nLayer processed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, (char *)"random")==0) return RANDOM;
    if (strcmp(s, (char *)"poly")==0) return POLY;
    if (strcmp(s, (char *)"constant")==0) return CONSTANT;
    if (strcmp(s, (char *)"step")==0) return STEP;
    if (strcmp(s, (char *)"exp")==0) return EXP;
    if (strcmp(s, (char *)"sigmoid")==0) return SIG;
    if (strcmp(s, (char *)"steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, (char *)"batch", 1);
    //if(state == 'p'){
    net->batch = 1;
    //}
    net->learning_rate = option_find_float(options, (char *)"learning_rate", .001);
    net->momentum = option_find_float(options, (char *)"momentum", .9);
    net->decay = option_find_float(options, (char *)"decay", .0001);
    int subdivs = option_find_int(options, (char *)"subdivisions",1);
    net->time_steps = option_find_int_quiet(options, (char *)"time_steps",1);
    net->notruth = option_find_int_quiet(options, (char *)"notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, (char *)"random", 0);

    net->adam = option_find_int_quiet(options, (char *)"adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, (char *)"B1", .9);
        net->B2 = option_find_float(options, (char *)"B2", .999);
        net->eps = option_find_float(options, (char *)"eps", .0000001);
    }

    net->h = option_find_int_quiet(options, (char *)"height",0);
    net->w = option_find_int_quiet(options, (char *)"width",0);
    net->c = option_find_int_quiet(options, (char *)"channels",0);
    net->inputs = option_find_int_quiet(options, (char *)"inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, (char *)"max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, (char *)"min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, (char *)"max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, (char *)"min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, (char *)"center",0);
    net->clip = option_find_float_quiet(options, (char *)"clip", 0);

    net->angle = option_find_float_quiet(options, (char *)"angle", 0);
    net->aspect = option_find_float_quiet(options, (char *)"aspect", 1);
    net->saturation = option_find_float_quiet(options, (char *)"saturation", 1);
    net->exposure = option_find_float_quiet(options, (char *)"exposure", 1);
    net->hue = option_find_float_quiet(options, (char *)"hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, (char *)"policy", (char *)"constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, (char *)"burn_in", 0);
    net->power = option_find_float_quiet(options, (char *)"power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, (char *)"step", 1);
        net->scale = option_find_float(options, (char *)"scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, (char *)"steps");
        char *p = option_find(options, (char *)"scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = (int *)calloc(n, sizeof(int));
        float *scales = (float *)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, (char *)"gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, (char *)"gamma", 1);
        net->step = option_find_int(options, (char *)"step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, (char *)"max_batches", 0);

    int ptr;
    int num_layers = net->n - global_start_index - 1; // Reduce the total count of layers in Enclave to N-3
    ecall_make_network(global_eid, &ptr, num_layers, net->learning_rate, net->momentum, net->decay, net->time_steps, net->notruth, net->batch, net->subdivisions, net->random, net->adam, net->B1, net->B2, net->eps, net->h, net->w, net->c, net->inputs, net->max_crop, net->min_crop, net->max_ratio, net->min_ratio, net->center, net->clip, net->angle, net->aspect, net->saturation, net->exposure, net->hue, net->burn_in, net->power, net->max_batches, global_start_index);
}

int is_network(section *s)
{
    return (strcmp(s->type, (char *)"[net]")==0
            || strcmp(s->type, (char *)"[network]")==0);
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);

    //net->gpu_index = gpu_index;
    net->gpu_index = -1;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");

    while(n){
        params.index = count;
	if((count >= global_start_index) && (count < net->n - 1)) // Skip the first and last layer
        {
            fprintf(stderr, "%5d ", count);
	}
	s = (section *)n->val;
        options = s->options;
        layer l;
	memset((void *)&l, 0, sizeof(layer));
        LAYER_TYPE lt = string_to_layer_type(s->type);

        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params, net->layers[count-1].output, net->layers[count-1].delta);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
	}else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, (char *)"truth", 0);
        l.onlyforward = option_find_int_quiet(options, (char *)"onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, (char *)"stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, (char *)"dontsave", 0);
        l.dontload = option_find_int_quiet(options, (char *)"dontload", 0);
        l.numload = option_find_int_quiet(options, (char *)"numload", 0);
        l.dontloadscales = option_find_int_quiet(options, (char *)"dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, (char *)"learning_rate", 1);
        l.smooth = option_find_float_quiet(options, (char *)"smooth", 0);
        option_unused(options);

        net->layers[count] = l;

        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        count_global = count;

        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }

    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = (float *)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float *)calloc(net->truths*net->batch, sizeof(float));
    if(workspace_size){
	int ptr;
        net->workspace = (float *)calloc(1, workspace_size);
	ecall_allocate_workspace(global_eid, &ptr, workspace_size);
        //netta.workspace = net->workspace;
    }

    return net;
}


list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section *)malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

/*
void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights_comm(layer l, int i)
{
    int num = l.nweights;
    save_weights_CA(l.biases, l.n, i, 'b');
    if (l.batch_normalize){
        save_weights_CA(l.scales, l.n, i, 's');
        save_weights_CA(l.rolling_mean, l.n, i, 'm');
        save_weights_CA(l.rolling_variance, l.n, i, 'v');
    }
    save_weights_CA(l.weights, num, i, 'w');
}


void save_connected_weights(layer l, FILE *fp)
{
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_connected_weights_comm(layer l, int i)
{
    int num = l.nweights;
    save_weights_CA(l.biases, l.outputs, i, 'b');
    save_weights_CA(l.weights, l.outputs*l.inputs, i, 'w');
    if (l.batch_normalize){
        save_weights_CA(l.scales, l.outputs, i, 's');
        save_weights_CA(l.rolling_mean, l.outputs, i, 'm');
        save_weights_CA(l.rolling_variance, l.outputs, i, 'v');
    }
}
*/


void transpose_matrix(float *a, int rows, int cols)
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


void load_connected_weights_comm(layer l, FILE *fp, int i, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
   
    int ptr; 
    if(i >= global_start_index)
    {
        ecall_transfer_weights(global_eid, &ptr, l.biases, l.outputs*sizeof(float), i, 'b', 0);
        ecall_transfer_weights(global_eid, &ptr, l.weights, l.outputs*l.inputs*sizeof(float), i, 'w', transpose);
    }

    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        
	if(i >= global_start_index)
        {
            ecall_transfer_weights(global_eid, &ptr, l.scales, l.outputs*sizeof(float), i, 's', 0);
            ecall_transfer_weights(global_eid, &ptr, l.rolling_mean, l.outputs*sizeof(float), i, 'm', 0);
            ecall_transfer_weights(global_eid, &ptr, l.rolling_variance, l.outputs*sizeof(float), i, 'v', 0);
	}
    }
}

void load_convolutional_weights_comm(layer l, FILE *fp, int i)
{
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    int ptr;

    fread(l.biases, sizeof(float), l.n, fp);
    if(i >= global_start_index)
    {
        ecall_transfer_weights(global_eid, &ptr, l.biases, l.n*sizeof(float), i, 'b', 0);
    }

    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
       
        if(i >= global_start_index)
        {	
            ecall_transfer_weights(global_eid, &ptr, l.scales, l.n*sizeof(float), i, 's', 0);
            ecall_transfer_weights(global_eid, &ptr, l.rolling_mean, l.n*sizeof(float), i, 'm', 0);
            ecall_transfer_weights(global_eid, &ptr, l.rolling_variance, l.n*sizeof(float), i, 'v', 0);
        }
    }
    
    fread(l.weights, sizeof(float), num, fp);
    if(i >= global_start_index)
    {
        ecall_transfer_weights(global_eid, &ptr, l.weights, num*sizeof(float), i, 'w', 0);
    }
}


void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
   
    clock_t time = clock();
    for(i = start; i < net->n && i < cutoff; ++i){
        
        // load weights of the NW side
        layer l = net->layers[i];
        //int layerTA_i = i;

        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights_comm(l, fp, i);
        }
        if(l.type == CONNECTED){
            load_connected_weights_comm(l, fp, i, transpose);
        }
    }

    fprintf(stderr, "Done!\n");
    fprintf(stderr, "Weights loaded in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n-1); // Load weights for layers from start index to N-1 th layer
}
