#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "Enclave_u.h"
#include "sgx_urts.h"
#include "App.h"
#include "darknet.h"
#include "utils.h"
#include "socket_api.h"

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid;
float *net_input_back;
float *net_delta_back;
float *net_output_back;
uint8_t *net_tag_buffer;
char *net_attestation_buffer;
uint32_t session_token;

int global_start_index;

// OCall implementations
void ocall_print(const char* str, float value) {
    printf("%s\n", str);
    printf("%.2f", value);
}

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret) {
    printf("SGX error code: %x\n", ret);
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

/*
void print_weights(char *cfgfile, char *weightfile, int n)
{
    network *net = load_network(cfgfile, weightfile, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for(i = 0; i < l.n; ++i){
        //printf("[");
        for(j = 0; j < l.size*l.size*l.c; ++j){
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i*l.size*l.size*l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}
*/

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, char *tag, char *tag_size, int top, clock_t time)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    layer l = net->layers[global_start_index];
    layer l1 = net->layers[net->n - 2];
    printf("\nTotal input size = %d\n\n",l.inputs*l.batch*sizeof(float));
    float *input_values = (float *)calloc(1, l.inputs*l.batch*sizeof(float));
    srand(2222222);

    //list *options = read_data_cfg(datacfg);

    //char *name_list = option_find_str(options, (char *)"names", 0);
    //if(!name_list) name_list = option_find_str(options, (char *)"labels", (char *)"data/labels.list");
    //if(top == 0) top = option_find_int(options, (char *)"top", 1);

    int i = 0;
    //char **names = get_labels(name_list);
    //clock_t time;
    //int *indexes = (int *)calloc(top, sizeof(int));
    //char buff[256];
    //char *input = buff;
    while(1){
	/*
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
	*/

        //image im = load_image_color(input, 0, 0);
        //image r = letterbox_image(im, net->w, net->h);
        //image r = resize_min(im, 320);
        //printf("%d %d\n", r.w, r.h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        //float *X = r.data;
        fprintf(stderr, "Loading partial outputs from %s...", filename);
        fflush(stdout);
        FILE *fp = fopen(filename, "rb");
        if(fp == NULL) 
	{
            file_error(filename);
	}

        fread(input_values, sizeof(float), l.inputs * l.batch, fp);	

	net_tag_buffer = (uint8_t *)calloc(1,16);    // 16 byte tag for encryption
	FILE *fp0 = fopen(tag, "rb");
        if(fp0 == NULL)
        {
            file_error(tag);
        }
        fread(net_tag_buffer, 1, 16, fp0);

	/*
	printf("\n\nPartial Inputs from network\n");
        for(i = 0; i < 500; i++)
        {
            printf("%.2f ", input_values[i]);
        }
	*/

	
        //time=clock();
	printf("\n Run forward propagation.....\n");
        float *partial_results = network_predict(net, input_values);
        //if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);

	//top_k(predictions, l1.outputs, top, indexes);

	
	/*
	printf("\n\nPartial results\n");
        for(i = 0; i < 500; i++)
        {
            printf("%.2f ",partial_results[i]);
        }
        printf("\n\n");
        */

	int total_size = l1.outputs * l1.batch;
        char output_file[100];
	char output_tag[80];
	printf("\nTotal size sgx : %d\n", total_size);
        strcpy(output_file, "partial_outputs_sgx.data");
        fprintf(stderr, "Saving partial outputs to %s\n", output_file);
	strcpy(output_tag, "tag_sgx.data");
        fprintf(stderr, "Saving generated GCM tag to %s\n", output_tag);
        FILE *fp1 = fopen(output_file, "wb");
	FILE *fp1_1 = fopen(output_tag, "wb");
        if(fp1 == NULL)
	{	
            file_error(output_file);
	}
	if(fp1_1 == NULL)
        {
            file_error(output_tag);
        }

        fwrite(partial_results, sizeof(float), total_size, fp1);
        fwrite(net_tag_buffer, 1, 16, fp1_1);

        free(net_output_back);
	free(input_values);
	free(net_tag_buffer);

	/*
	printf("\n\nPredicted results\n");
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
	    //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
	*/

        //if(r.data != im.data) free_image(r);
        //free_image(im);
	fclose(fp);
	fclose(fp1);
        fclose(fp0);
	fclose(fp1_1);

        /*	
	// Return the encrypted partial results back to trustzone
	int status = 0;
        status = Socket_Client(output_file);
        if(status != 0)
        {
            printf("\nUnable to send output file!!\n");
            exit(1);
        }

	// Return the generated GCM tag back to trustzone
        status = 0;
        status = Socket_Client(output_tag);
        if(status != 0)
        {
            printf("\nUnable to send tag file!!\n");
            exit(1);
        }
	*/

	int status = 0;
        status = Socket_Client(output_file, output_tag, 5000);
        if(status != 0)
        {
            printf("\nUnable to send output file!!\n");
            exit(1);
        }

        fprintf(stderr, "\nTotal execution time SGX: %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);	

        if (filename) break;
    }
}

void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int status = 0;
    int top = find_int_arg(argc, argv, (char *)"-t", 0);
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *filesize = (argc > 7) ? argv[7]: 0;
    char *tag = (argc > 8) ? argv[8]: 0;
    char *tag_size = (argc > 9) ? argv[9]: 0;
    clock_t time;

    //Get the start layer index for SGX
    global_start_index = find_int_arg(argc, argv, (char *)"-st", 1);
    global_start_index = 2;
    printf("SGX layer start index: %d\n",global_start_index);

    status = Socket_Server(filename, filesize, tag, tag_size, &time, 5000, 0);
    if(status != 0)
    {
        printf("\nUnable to receive file!!\n");
        exit(1);
    }
    

    /*
    status = Socket_Server(filename, filesize, NULL, NULL, &time, 1);
    if(status != 0)
    {
        printf("\nUnable to receive file!!\n");
	exit(1);
    }

    status = Socket_Server(tag, tag_size, NULL, NULL, &time, 1);
    if(status != 0)
    {
        printf("\nUnable to receive file!!\n");
        exit(1);
    }
    */

    if(0==strcmp(argv[2], (char *)"predict")) {
        //state = 'p';
        predict_classifier(data, cfg, weights, filename, tag, tag_size, top, time);
    }
    else
    {
        printf("Incorrect arguments\n");
    }
}

int remote_attestation(void)
{
    int result = 0;
    int status = 0;
    clock_t time = clock();
    //clock_t time2;
    
    // Wait for partial results from SGX
    printf("\nWait for encrypted UUID from SGX\n");
    status = Socket_Server((char *)"data/darknet/session_token.data", (char *)"data/darknet/token_size.txt", NULL, NULL, &time, 8000, 1); // Run in attestation mode
    if(status != 0)
    {
        printf("\nFile receiving failed...\n");
        exit(1);
    }
    printf("Done....\n");

    char *session_token_file = (char *)"data/darknet/session_token.data";
    fprintf(stderr, "Loading session token from %s...", session_token_file);
    FILE *fp = fopen(session_token_file, "rb");
    if(fp == NULL)
    {
        file_error(session_token_file);
    }

    fread(&session_token, sizeof(uint32_t), 1, fp);
    fclose(fp);

    net_tag_buffer = (uint8_t *)calloc(1,16);
    // Allocate memory for attestation
    net_attestation_buffer = (char *)calloc(sizeof(char), 46);

    printf("\nSession token value: %d",session_token);
    // Encrypt UUID with token for attestation
    printf("\n Attest the session token with UUID\n");
    int ptr;
    ecall_attest_session_token(global_eid, &ptr, session_token, net_attestation_buffer, sizeof(char) * 46, net_tag_buffer, 16);

    if(ptr != 0)
    {
	// Encryption failed in the Enclave
        result = -1;
    }
    else
    {
        // Send the encrypted attestation buffer back to trustzone	
	char output_token[80];
	char output_tag[80];
        strcpy(output_token, "uuid_token_sgx.data");
        fprintf(stderr, "Saving uuid token to %s\n", output_token);
        strcpy(output_tag, "tag_token_sgx.data");
        fprintf(stderr, "Saving generated GCM tag to %s\n", output_tag);
        FILE *fp1 = fopen(output_token, "wb");
        FILE *fp1_1 = fopen(output_tag, "wb");
        if(fp1 == NULL)
        {
            file_error(output_token);
        }
        if(fp1_1 == NULL)
        {
            file_error(output_tag);
        }

        fwrite(net_attestation_buffer, sizeof(char), 46, fp1);
        fwrite(net_tag_buffer, 1, 16, fp1_1);
        
	fclose(fp1);
	fclose(fp1_1);

	printf("\nSend the encrypted token and tag back to trustzone\n");
        
	/*
	// Return the encrypted token back to trustzone
        status = Socket_Client(output_token);
        if(status != 0)
        {
            printf("\nUnable to send output file!!\n");
            exit(1);
        }

        // Return the generated GCM tag back to trustzone
        status = 0;
        status = Socket_Client(output_tag);
        if(status != 0)
        {
            printf("\nUnable to send tag file!!\n");
            exit(1);
        }
        */

	// Return the encrypted token back to trustzone
        status = Socket_Client(output_token, output_tag, 8000);
        if(status != 0)
        {
            printf("\nUnable to send output file!!\n");
            exit(1);
        }

	fprintf(stderr, "\nAttestation process completed in %f seconds.\n", (float)(clock()-time)/CLOCKS_PER_SEC);
    }
    free(net_tag_buffer);
    free(net_attestation_buffer);
    return result;
}

//int main(int argc, char const *argv[]) {
int main(int argc, char **argv) {
    global_eid = 0;
    if (initialize_enclave() < 0) {
        std::cout << "Fail to initialize enclave." << std::endl;
        return 1;
    }
    int ptr;
  
    printf("\nRemote attestation\n");
    int result = remote_attestation();

    if(result == 0)
    {
        printf("Begin darknet\n");
        run_classifier(argc, argv); 
    }
    else
    {
        printf("\nRemote attestation process failed...\n");
    }
    
    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);

    std::cout << "\nInfo: SampleEnclave successfully returned." << std::endl;

    return 0;
}
