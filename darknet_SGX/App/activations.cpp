#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return (char *)"logistic";
        case LOGGY:
            return (char *)"loggy";
        case RELU:
            return (char *)"relu";
        case ELU:
            return (char *)"elu";
        case SELU:
            return (char *)"selu";
        case RELIE:
            return (char *)"relie";
        case RAMP:
            return (char *)"ramp";
        case LINEAR:
            return (char *)"linear";
        case TANH:
            return (char *)"tanh";
        case PLSE:
            return (char *)"plse";
        case LEAKY:
            return (char *)"leaky";
        case STAIR:
            return (char *)"stair";
        case HARDTAN:
            return (char *)"hardtan";
        case LHTAN:
            return (char *)"lhtan";
        default:
            break;
    }
    return (char *)"relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, (char *)"logistic")==0) return LOGISTIC;
    if (strcmp(s, (char *)"loggy")==0) return LOGGY;
    if (strcmp(s, (char *)"relu")==0) return RELU;
    if (strcmp(s, (char *)"elu")==0) return ELU;
    if (strcmp(s, (char *)"selu")==0) return SELU;
    if (strcmp(s, (char *)"relie")==0) return RELIE;
    if (strcmp(s, (char *)"plse")==0) return PLSE;
    if (strcmp(s, (char *)"hardtan")==0) return HARDTAN;
    if (strcmp(s, (char *)"lhtan")==0) return LHTAN;
    if (strcmp(s, (char *)"linear")==0) return LINEAR;
    if (strcmp(s, (char *)"ramp")==0) return RAMP;
    if (strcmp(s, (char *)"leaky")==0) return LEAKY;
    if (strcmp(s, (char *)"tanh")==0) return TANH;
    if (strcmp(s, (char *)"stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

