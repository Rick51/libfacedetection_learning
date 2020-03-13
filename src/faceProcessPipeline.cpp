#include "cnn.h"
#include "head.h"

#define FACE_DETECT_BUFFER_SIZE 0x20000


int facedetect(FRAME *pOrgImg, RECT* facesInfo)
{

    if (NULL == pOrgImg)
    {
        return 0;
    }

    int width = pOrgImg->width;
    int height = pOrgImg->height;
    int stride = pOrgImg->pStride[0];

    unsigned char *srcpixels = NULL;
    srcpixels = pOrgImg->pAddr[0];
    
    int * pResults = NULL; 
    unsigned char * pBuffer = (unsigned char *)malloc(FACE_DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return 0;
    }
    pResults = facedetect_cnn(pBuffer, (unsigned char*)srcpixels, width, height, stride);

    int faceNum = pResults ? *pResults : 0;
    for(int i = 0; i < faceNum; i++)
    {
        short * p = ((short*)(pResults+1))+142*i;

        facesInfo[i].x = p[0];
        facesInfo[i].y = p[1];
        facesInfo[i].width = p[2];
        facesInfo[i].height = p[3];
        facesInfo[i].score = p[4];  
    }
    if(pBuffer)
    {
        free(pBuffer);
        pBuffer = NULL;
    }
    return faceNum;
}

int faceDetectProcess(FRAME *pOrgImg, RECT *facesInfo)
{
    int faceNum = facedetect(pOrgImg, facesInfo);
    return faceNum;
}
