
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<sys/resource.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<netinet/in.h>
#include<signal.h>
#include<time.h>

#define min(X,Y) (((X) < (Y)) ? (X) : (Y))
#define max(X,Y) (((X) > (Y)) ? (X) : (Y))

//int find_int_arg(int argc, char **argv, char *arg, int def);
//void del_arg(int argc, char **argv, int index);

// Client definitions
int Socket_Client(char *filename, char *filename1, int port, int mode);
short SocketCreate_Client(void);
int SocketConnect(int hSocket, int port);
//int SocketSend(int hSocket,char* Rqst,short lenRqst);
//int SocketReceive(int hSocket,char* Rsp,short RvcSize);
bool sendfile(int sock, FILE *f);
bool sendlong(int sock, long value);
bool senddata(int sock, void *buf, int buflen);

// Server definitions
int Socket_Server(char *filename, char *filesize, char *tag, char *tag_size, clock_t *time, int port, int mode);
int BindCreatedSocket(int hSocket, int port);
bool readdata(int sock, void *buf, int buflen);
bool readlong(int sock, long *value);
bool readfile(int sock, FILE *f, FILE *f1);
