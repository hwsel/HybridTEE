// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


void compare_networks(network *n1, network *n2, data d);
char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);

//float network_accuracy_multi(network *net, data d, int n);
void print_network(network *net);
void calc_network_cost(network *net);

#endif

