
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "utils.h"
#include "Enclave_u.h"
#include "App.h"

//#include "crop_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
//#include "activation_layer.h"
#include "maxpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "parser.h"
#include "socket_api.h"

//extern sgx_enclave_id_t global_eid;
//extern float *net_input_back;
//extern float *net_delta_back;
//extern float *net_output_back;

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return (char *)"convolutional";
        case ACTIVE:
            return (char *)"activation";
        case DECONVOLUTIONAL:
            return (char *)"deconvolutional";
        case CONNECTED:
            return (char *)"connected";
        case MAXPOOL:
            return (char *)"maxpool";
        case SOFTMAX:
            return (char *)"softmax";
        case DROPOUT:
            return (char *)"dropout";
        case COST:
            return (char *)"cost";
        case BATCHNORM:
            return (char *)"batchnorm";
        default:
            break;
    }
    return (char *)"none";
}

network *make_network(int n)
{
    network *net = (network *)calloc(1, sizeof(network));
    net->n = n;
    net->layers = (layer *)calloc(net->n, sizeof(layer));
    net->seen = (size_t *)calloc(1, sizeof(size_t));
    net->t    = (int *)calloc(1, sizeof(int));
    net->cost = (float *)calloc(1, sizeof(float));
    return net;
}

int workspaceBOO(network net)
{
   int workspace_size = 0;
   for(int i = 0; i < net.n; ++i)
   {
        if(CONVOLUTIONAL == net.layers[i].type)
        {
            layer l = net.layers[i];
            if(l.workspace_size > workspace_size){
                  workspace_size = l.workspace_size;
            }

        }
   }

   return workspace_size;
}

int wssize = -1;

void forward_network(network *netp, float *inputs)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int ptr;
    network net = *netp;
    layer l = net.layers[global_start_index];

    ret = ecall_forward_network(global_eid, &ptr, inputs, l.inputs*l.batch*sizeof(float), net_tag_buffer, 16, net.train);
    if(ptr != 0)
    {
        printf("\n\n Error occurred in forward network : %d\n", ptr);
	exit(1);
    }
    //calc_network_cost(netp);
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            //printf("i=%d, cost=%f\n",i,net.layers[i].cost[0]);
            ++count;
        }
    }
    *net.cost = sum/count;
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

/*
void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}
*/

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    clock_t time = clock();
    forward_network(net, input);
    fprintf(stderr, "\nForward propagation completed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    
    float *out;
    layer l = net->layers[net->n - 2];
    // Allocate memory for output returned from the TA
    net_output_back = (float *)malloc(sizeof(float) * l.outputs * l.batch);
    memset(net_tag_buffer, 0, 16);

    //ecall to return output
    int ptr;
    ecall_net_output_return(global_eid, &ptr, net_output_back, l.outputs * sizeof(float) * l.batch, net_tag_buffer, 16);
    out = net_output_back;
    *net = orig;
    return out;
}


int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = (float *)calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}


float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
    free(net);
}


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);
}
