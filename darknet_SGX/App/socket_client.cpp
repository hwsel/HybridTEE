// Reference for socket programming code is  AticleWorld tutorial 
// https://aticleworld.com/socket-programming-in-c-using-tcpip/

#include "socket_api.h"

#define MAX 1024

//Create a Socket for server communication
short SocketCreate_Client(void)
{
    short hSocket;
    printf("Create the socket\n");
    hSocket = socket(AF_INET, SOCK_STREAM, 0);
    return hSocket;
}

//try to connect with server
int SocketConnect(int hSocket, int port)
{
    int iRetval=-1;
    //int ServerPort = 8000;
    int ServerPort = port;
    struct sockaddr_in remote= {0};
    remote.sin_addr.s_addr = inet_addr("169.254.232.180"); //Server address
    remote.sin_family = AF_INET;
    remote.sin_port = htons(ServerPort);
    iRetval = connect(hSocket,(struct sockaddr *)&remote,sizeof(struct sockaddr_in));
    return iRetval;
}

/*
// Send the data to the server and set the timeout of 20 seconds
int SocketSend(int hSocket, FILE *fp)
{
    char buff[MAX];
    int bytes_sent = -1;
    int total_bytes_sent = 0;
    struct timeval tv;
    tv.tv_sec = 360;  // 360 Secs Timeout
    tv.tv_usec = 0;
    if(setsockopt(hSocket,SOL_SOCKET,SO_SNDTIMEO,(char *)&tv,sizeof(tv)) < 0)
    {
        printf("Time Out\n");
        return -1;
    }

    while(fgets(buff, MAX, fp) != NULL)
    {
        bytes_sent = send(hSocket, buff, sizeof(buff), 0);
        total_bytes_sent += bytes_sent;
	printf("Total Bytes Sent : %d\n",total_bytes_sent);
    }
    return bytes_sent;
}
*/

bool senddata(int sock, void *buf, int buflen)
{
    unsigned char *pbuf = (unsigned char *)buf;
    int total_bytes_sent = 0;

    while (buflen > 0)
    {
        int num = send(sock, pbuf, buflen, 0);
        if (num < 0)
        {
            //if (WSAGetLastError() == WSAEWOULDBLOCK)
            //{
                // optional: use select() to check for timeout to fail the send
                //continue;
            //}
            return false;
        }

        pbuf += num;
        buflen -= num;
	total_bytes_sent += num;
        printf("Total Bytes Sent : %d\n",total_bytes_sent);
    }
    return true;
}

bool sendlong(int sock, long value)
{
    value = htonl(value);
    return senddata(sock, &value, sizeof(value));
}

bool sendfile(int sock, FILE *f)
{
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    rewind(f);
    if (filesize == EOF)
        return false;
    if (!sendlong(sock, filesize))
        return false;
    if (filesize > 0)
    {
        char buffer[MAX];
        do
        {
            size_t num = min(filesize, sizeof(buffer));
            num = fread(buffer, 1, num, f);
            if (num < 1)
                return false;
            if (!senddata(sock, buffer, num))
                return false;
            filesize -= num;
        }
        while (filesize > 0);
    }
    return true;
}

int Socket_Client(char *filename, char *filename1, int port)
{
    int hSocket;
    //struct sockaddr_in server;
    //char SendToServer[100] = {0};
    //char server_reply[200] = {0};
    //char *filename = argv[1];

    if(filename == NULL || filename1 == NULL)
    {
        printf("\nIncorrect filename\n\n");
	return 1;
    }

    // Read the input file
    FILE *fp;
    fp = fopen(filename,"rb");
    if( fp == NULL )
    {
        printf("Error IN Opening File .. \n");
        return 1;
    }

    FILE *fp1;
    fp1 = fopen(filename1,"rb");
    if( fp1 == NULL )
    {
        printf("Error IN Opening File .. \n");
        return 1;
    }

    
    hSocket = SocketCreate_Client();
    if(hSocket == -1)
    {
        printf("Could not create socket\n");
        return 1;
    }
    printf("Socket is created\n");


    int reuse = 1;
    if (setsockopt(hSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
        perror("setsockopt(SO_REUSEADDR) failed");

    if (setsockopt(hSocket, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
        perror("setsockopt(SO_REUSEPORT) failed");

    /*
    linger lin;
    lin.l_onoff = 1;
    lin.l_linger = 0;
    if(setsockopt(hSocket, SOL_SOCKET, SO_LINGER, (const char *)&lin, sizeof(linger)) < 0)
        perror("setsockopt(SO_LINGER) failed");
    */

    //Connect to remote server
    if (SocketConnect(hSocket, port) < 0)
    {
        perror("connect failed.\n");
        return 1;
    }
    printf("Sucessfully conected with server\n");
    //printf("Enter the Message: ");
    //gets(SendToServer);

    // First send the filesize to the server, then send the data
    if(sendfile(hSocket, fp) == false)
    {
        printf("File Send failed\n");
	return 1;
    }

    // First send the filesize to the server, then send the data
    if(sendfile(hSocket, fp1) == false)
    {
        printf("File Send failed\n");
        return 1;
    }
    
    // Close the connection
    close(hSocket);
    shutdown(hSocket,0);
    shutdown(hSocket,1);
    shutdown(hSocket,2);
    fclose (fp);
    fclose (fp1);
    sleep(1);
    return 0;
}
