// Reference for socket programming code is  AticleWorld tutorial 
// https://aticleworld.com/socket-programming-in-c-using-tcpip/
#include "socket_api.h"
#include <sys/socket.h>

short SocketCreate(void)
{
    short hSocket;
    printf("Create the socket\n");
    hSocket = socket(AF_INET, SOCK_STREAM, 0);
    return hSocket;
}
    
int BindCreatedSocket(int hSocket, int port)
{
    int iRetval=-1;
    //int ClientPort = 8000;
    int ClientPort = port;
    struct sockaddr_in  remote= {0};
    /* Internet address family */
    remote.sin_family = AF_INET;
    /* Any incoming interface */
    remote.sin_addr.s_addr = htonl(INADDR_ANY);
    //remote.sin_addr.s_addr = inet_addr("169.254.232.180"); // Client address
    remote.sin_family = AF_INET;
    remote.sin_port = htons(ClientPort); /* Local port */
    iRetval = bind(hSocket,(struct sockaddr *)&remote,sizeof(remote));
    return iRetval;
}

bool readdata(int sock, void *buf, int buflen)
{
    unsigned char *pbuf = (unsigned char *) buf;
    int total_bytes_rcvd = 0;

    while (buflen > 0)
    {
        int num = recv(sock, pbuf, buflen, 0);
        //if (num < 0)
        //{
            //if (WSAGetLastError() == WSAEWOULDBLOCK)
            //{
                // optional: use select() to check for timeout to fail the read
               // continue;
            //}
            //return false;
        //}
        if (num <= 0)
            return false;

        pbuf += num;
        buflen -= num;
	total_bytes_rcvd += num;
        printf("Total Bytes Received : %d\n",total_bytes_rcvd);
    }

    return true;
}

bool readlong(int sock, long *value)
{
    if (!readdata(sock, value, sizeof(value)))
        return false;
    *value = ntohl(*value);
    return true;
}

bool readfile(int sock, FILE *f, FILE *f1)
{
    long filesize;
    if (!readlong(sock, &filesize))
        return false;
    if (filesize > 0)
    {
        char buffer[1024];
	// Write the total file size in bytes to a file
	printf("Filesize = %ld\n",filesize);
	//size_t written_size = fwrite(&filesize, 1, sizeof(filesize), f1);
	fprintf(f1, "%ld",filesize);
        do
        {
            int num = min(filesize, sizeof(buffer));
            if (!readdata(sock, buffer, num))
                return false;
            int offset = 0;
            do
            {
                size_t written = fwrite(&buffer[offset], 1, num-offset, f);
                if (written < 1)
                    return false;
                offset += written;
            }
            while (offset < num);
            filesize -= num;
        }
        while (filesize > 0);
    }
    return true;
}

int Socket_Server(char *filename, char *filesize, char *tag, char *tag_size, clock_t *time, int port, int mode)
{
    int socket_desc, sock, clientLen, read_size;
    struct sockaddr_in server, client;
    int count = 0;
    //char client_message[200]= {0};
    //char message[100] = {0};
    //const char *pMessage = "hello aticleworld.com";
    //char *filename = argv[1];
    //char *filesize = argv[2];
    if(filename == NULL || filesize == NULL || ((mode == 0) && (tag == NULL || tag_size == NULL)))
    {
        printf("Invalid filename\n");
	return 1;
    }
    
    //Create socket
    socket_desc = SocketCreate();
    if (socket_desc == -1)
    {
        printf("Could not create socket");
        return 1;
    }
    printf("Socket created\n");

    int reuse = 1;
    if (setsockopt(socket_desc, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
        perror("setsockopt(SO_REUSEADDR) failed");

    if (setsockopt(socket_desc, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
        perror("setsockopt(SO_REUSEPORT) failed");

    /*
    struct linger lin;
    lin.l_onoff = 1;
    lin.l_linger = 1;
    if(setsockopt(socket_desc, SOL_SOCKET, SO_LINGER, (const char *)&lin, sizeof(struct linger)) < 0)
        perror("setsockopt(SO_LINGER) failed");
    */

    //Bind
    if( BindCreatedSocket(socket_desc, port) < 0)
    {
        //print the error message
        perror("bind failed.");
        return 1;
    }
    printf("bind done\n");
    //Listen
    if(listen(socket_desc, 5) != 0)
    {
        //print the error message
        perror("listen failed");
        return 1;
    }
    //Accept and incoming connection
    while(1)
    {
        printf("Waiting for incoming connections...\n");
        clientLen = sizeof(struct sockaddr_in);
        //accept connection from an incoming client
        sock = accept(socket_desc,(struct sockaddr *)&client,(socklen_t*)&clientLen);
        if (sock < 0)
        {
            perror("accept failed");
            return 1;
        }
        printf("Connection accepted\n");
        
	if(mode == 0)
        {
            FILE *fp;
            FILE *fp1;
            FILE *fp2;
            FILE *fp3;
            fp = fopen(filename, "wb");
            fp1 = fopen(filesize, "wb");
            fp2 = fopen(tag, "wb");
            fp3 = fopen(tag_size, "wb");
            *time = clock();
            if(fp == NULL || fp1 == NULL || fp2 == NULL || fp3 == NULL)
            {
                printf("ERROR, could not open file\n");
                return 1;
            }

            if(readfile(sock, fp, fp1) == false)
            {
                printf("ERROR, could not read the data\n");
                return 1;
            }

            if(readfile(sock, fp2, fp3) == false)
            {
                printf("ERROR, could not read the data\n");
                return 1;
            }

            close(sock);
            fclose(fp);
            fclose(fp1);
            fclose(fp2);
            fclose(fp3);
            sleep(1);
        }
        else
        {   
            FILE *fp;
            FILE *fp1;
            fp = fopen(filename, "wb");
            fp1 = fopen(filesize, "wb");
            if(fp == NULL || fp1 == NULL)
            {   
                printf("ERROR, could not open file\n");
                return 1;
            }
            if(readfile(sock, fp, fp1) == false)
            {
                printf("ERROR, could not read the data\n");
                return 1;
            }

            close(sock);
            fclose(fp);
            fclose(fp1);
            sleep(1);
        }
        break;
    }
    return 0;
}

