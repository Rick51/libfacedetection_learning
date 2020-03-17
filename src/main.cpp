#include "cnn.h"
#include "head.h"

#if defined DEBUG

int load_bin(void *buf, unsigned char size, unsigned int width, unsigned int height, const char* filename)
{
    FILE *fp = NULL;
    int read_number = 0;
    if(!buf || !filename) return ERR;
    fp = fopen(filename,"rb");
    if(!fp) return ERR;
    read_number = fread(buf, size, width * height, fp);
    printf("read %d pixels(%d bytes) from %s\n", read_number,read_number * size, filename);
    fclose(fp);
    return OK;
}

int write_bin(void *buf, unsigned char size, unsigned int width, unsigned int height, const char* filename)
{
    FILE *fp = NULL;
    int write_number = 0;
    int i, j;
    if(!buf) return ERR;
    fp = fopen(filename, "wb");
    if(!fp) return ERR;

    write_number = fwrite(buf, size, width * height, fp);
    printf("write %d pixels(%d bytes) to %s\n", write_number, write_number * size, filename);
    fclose(fp);
    return OK;
}
#endif

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        printf("user should give width and height of the demo image.\n");
        exit(1);
    }
    int ret = ERR;  
    FRAME oriImg = {0};                                         //原始图片
    int DEMO_ABGR_BIN_READ_ORIG_X;
    int DEMO_ABGR_BIN_READ_ORIG_Y;                             
    sscanf(argv[1],"%d", &DEMO_ABGR_BIN_READ_ORIG_X);           //原始图片的宽
    sscanf(argv[2],"%d", &DEMO_ABGR_BIN_READ_ORIG_Y);           //原始图片的长
    
    FRAME padImg = {0};                                         //添加黑边的图片
    int top_padding = 0;                                        //添加的上方黑边的长度大小
    int left_padding = 0;                                       //添加的左方黑边的长度大小

    FRAME scaleImg = {0};                                       //添加了黑边并缩放至320X240大小的图片
    
    int faceNum = 0;                                            //最终有多少张人脸
    RECT facesInfo[MAX_FACENUM];                                //人脸信息数组，保存人脸的x y w h信息                  
    memset(facesInfo, 0, sizeof(RECT)*MAX_FACENUM); 
    int i = 0;                                                  //~

    /* 读取输入，DEMO_ABGR_BIN_READ_ORIG_X , DEMO_ABGR_BIN_READ_ORIG_Y 为原图真实长宽,这里demo图片为500X671大小。*/
    RGB_IMAGE ori_image[DEMO_ABGR_BIN_READ_ORIG_X * DEMO_ABGR_BIN_READ_ORIG_Y];
    if(ERR == load_bin((void*)ori_image, sizeof(RGB_IMAGE), DEMO_ABGR_BIN_READ_ORIG_X, DEMO_ABGR_BIN_READ_ORIG_Y, DEMO_ABGR_BIN_READ))
    {
        printf("load input image failed.\n");
        return ret;
    }

    oriImg.pAddr[0] = (unsigned char*)ori_image;
    oriImg.width = DEMO_ABGR_BIN_READ_ORIG_X;
    oriImg.height = DEMO_ABGR_BIN_READ_ORIG_Y;
    oriImg.pStride[0] = oriImg.width << 2;

    /* 计算padding的方向和padding 的大小 */
    prePadding(&oriImg, &padImg, &top_padding, &left_padding);
    
    padImg.pAddr[0] = NULL;
    padImg.pAddr[0] = (unsigned char*)calloc(padImg.width * padImg.height, 4);
    if(!padImg.pAddr[0])
    {
        printf("malloc for padding image failed.\n");
        return ret;
    }

    /* padding black pixels */
    if (OK != processPadding(&oriImg, &padImg, top_padding, left_padding))
    {
        printf("padding zero to input image failed.\n");
        goto output1;
    }

#if defined DEBUG
    printf("after padding, image's width is %d, height is %d\n",padImg.width, padImg.height);
    write_bin((void*)padImg.pAddr[0], sizeof(RGB_IMAGE), padImg.width, padImg.height, DEMO_ABGR_BIN_PADDING_WRITE);    
#endif    

    /* 按照比例缩放至320X240大小传给人脸检测库 */
    scaleImg.width = DEMO_ABGR_BIN_READ_X;
    scaleImg.height = DEMO_ABGR_BIN_READ_Y;
    scaleImg.pStride[0] = scaleImg.width << 2;
    scaleImg.pAddr[0] = NULL;
    scaleImg.pAddr[0] = (unsigned char*)calloc(DEMO_ABGR_BIN_READ_X * DEMO_ABGR_BIN_READ_Y, 4);
    if (!scaleImg.pAddr[0])
    {
        printf("malloc for scale image failed.\n");
        goto output1;
    }

    if(OK != resizeBL(&padImg, &scaleImg))
    {
        printf("resize padding image to 320X240 image failed.\n");
        goto output2;
    }

#if defined DEBUG
    printf("after scaleing, image's width is %d, height is %d\n",scaleImg.width, scaleImg.height);
    write_bin((void*)scaleImg.pAddr[0], sizeof(RGB_IMAGE), scaleImg.width, scaleImg.height, DEMO_ABGR_BIN_WRITE);    
#endif


    faceNum = faceDetectProcess(&scaleImg, facesInfo);    
    if(0 == faceNum)
    {
        printf("no faces detected.\n");
        goto output2;
    }
    ret = OK;

#if defined DEBUG
    printf("face number is %d\n", faceNum);
    printf("face info is:\n");
    for (i = 0; i < faceNum; i++)
    {
        facesInfo[i].x = ((facesInfo[i].x * padImg.width) / float(DEMO_ABGR_BIN_READ_X)) - left_padding;
        facesInfo[i].y = ((facesInfo[i].y * padImg.height) / float(DEMO_ABGR_BIN_READ_Y)) - top_padding;
        facesInfo[i].width = (facesInfo[i].width * padImg.width) / float(DEMO_ABGR_BIN_READ_X);
        facesInfo[i].height = (facesInfo[i].height * padImg.height) / float(DEMO_ABGR_BIN_READ_Y);
        if ((facesInfo[i].x + facesInfo[i].width >= DEMO_ABGR_BIN_READ_ORIG_X) || (facesInfo[i].y + facesInfo[i].height >= DEMO_ABGR_BIN_READ_ORIG_Y))
        {
            printf("face[%d] is illegal.\n",i);
        }
        else
        {
            printf("face[%d]: x1: %d, y1: %d, x2: %d, y2: %d\n",i, facesInfo[i].x, facesInfo[i].y, facesInfo[i].x + facesInfo[i].width ,facesInfo[i].y + facesInfo[i].height);           
        }
        
    }           
#endif

output2:
    if(scaleImg.pAddr[0])
    {
        free(scaleImg.pAddr[0]);
        scaleImg.pAddr[0] = NULL;
    }

output1:
    if(padImg.pAddr[0])
    {
        free(padImg.pAddr[0]);
        padImg.pAddr[0] = NULL;
    }

    return ret;
}