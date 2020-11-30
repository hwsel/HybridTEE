#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"
#include "math_TA.h"

#include "darknetp_ta.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;
uint8_t *ta_tag_buffer;
uint32_t ta_random_number;

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
    netta.n = n;

    // Update TCB tracker
    gGlobalBufferTracker += (2*sizeof(float) + sizeof(int) + 12);
    gGlobalBufferTracker += sizeof(netta);

    EMSG("No of layers in TA = %d\n",netta.n);
    netta.seen = calloc(1, sizeof(size_t));
    netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));

    gGlobalBufferTracker += (sizeof(size_t) + sizeof(int) + sizeof(float) + (netta.n * sizeof(layer_TA)));

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

    //netta.truth = net->truth; ////// ing network.c train_network
}

void add_noise_TA(layer_TA *l)
{
    int row = l->out_h;
    int column = l->out_w;
    int channels = l->out_c;
    int i,j,k;
    int noise_perc = 10;
    int noise_check;
    //int m = l->n/l->groups;
    //int k = l->size*l->size*l->c/l->groups;
    //int n = l->out_w*l->out_h;

    //srand((unsigned)time(NULL));
    //printf("\nValues = %d %d %d\n",row, column, channels);
    for(i = 0; i < channels; i++)
    {
        for(j = 0; j < row; j++)
        {
            for(k = 0; k < column; k++)
            {
                int col_index = (i * row + j) * column + k;
                //printf("%.2f ", l->output[col_index]);
                noise_check = rand() % ((noise_perc+1)-1) + 1;
                if( noise_check == noise_perc )
                {
                    int val = rand() % 255;
                    if(val >= 127)
                        l->output[col_index] = 1;
                    else
                        l->output[col_index] = 0;
                }
            }
        }
    }
}

void forward_network_TA()
{
    roundnum++;
    int i;
    for(i = 0; i < netta.n; ++i){
        netta.index = i;
        layer_TA l = netta.layers[i];

	if(l.type == SOFTMAX_TA)
	{
	    // Seperately handled for softmax layer
            continue;
	}

        if(l.delta){
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward_TA(l, netta);

#if ADD_NOISE_AT_TCB_CUTOFF
        // Add noise to the partition layer
        if(i == 2)
        {
            add_noise_TA(&l);
        }
#endif

        netta.input = l.output;

        if(l.truth) {
            netta.truth = l.output;
        }

	// Save output of last processed layer
	if(i == netta.n - 1){
            ta_net_output = calloc(l.outputs*l.batch,sizeof(float));
	    ta_tag_buffer = (uint8_t *)calloc(1, 16);

            // Update TCB tracker
            gGlobalBufferTracker += (sizeof(float)*l.outputs*l.batch);

            for(int z=0; z<l.outputs*l.batch; z++){
                ta_net_output[z] = l.output[z];
            }
        }

	EMSG("TCB usage = %d Bytes\n", gGlobalBufferTracker);
	/*if(gGlobalBufferTracker > TA_THRESHOLD_LIMIT)
        {
            TA_threshold_exceeded = true;
	    break;	// Threshold exceeded. Stop further processing
        }*/
    }

    //if(!TA_threshold_exceeded)
    //{
    calc_network_cost_TA();
    //}
}


void update_network_TA(update_args_TA a)
{
    int i;
    for(i = 0; i < netta.n; ++i){
        layer_TA l = netta.layers[i];
        if(l.update_TA){
            l.update_TA(l, a);
        }
    }
}


void calc_network_cost_TA()
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


void calc_network_loss_TA(int n, int batch)
{
    float loss = (float)err_sum/(n*batch);

    if(avg_loss == -1) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    char loss_char[20];
    char avg_loss_char[20];
    ftoa(loss, loss_char, 5);
    ftoa(avg_loss, avg_loss_char, 5);
    IMSG("loss = %s, avg loss = %s from the TA\n",loss_char, avg_loss_char);
    err_sum = 0;
}



void backward_network_TA(float *ca_net_input, float *ca_net_delta)
{
    int i;

    for(i = netta.n-1; i >= 0; --i){
        layer_TA l = netta.layers[i];

        if(l.stopbackward) break;
        if(i == 0){
            // ta_net_input malloc so not destroy before addition backward
            ta_net_input = malloc(sizeof(float)*l.inputs*l.batch);
            ta_net_delta = malloc(sizeof(float)*l.inputs*l.batch);

            for(int z=0; z<l.inputs*l.batch; z++){
                ta_net_input[z] = ca_net_input[z];
                ta_net_delta[z] = ca_net_delta[z];
            }

            netta.input = ta_net_input;
            netta.delta = ta_net_delta;
        }else{
            layer_TA prev = netta.layers[i-1];
            netta.input = prev.output;
            netta.delta = prev.delta;
        }

        netta.index = i;
        l.backward_TA(l, netta);

        if((l.type == DROPOUT_TA) && (i == 0)){
            for(int z=0; z<l.inputs*l.batch; z++){
                ta_net_input[z] = l.output[z];
                ta_net_delta[z] = l.delta[z];
            }
            //netta.input = l.output;
            //netta.delta = l.delta;
        }
    }
    //backward_network_back_TA_params(netta.input, netta.delta, netta.layers[0].inputs, netta.batch);
}
