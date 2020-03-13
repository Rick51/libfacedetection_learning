#include "head.h"


/*******************************************************************************
Function:       prePadding
Description:    get padding direction and size
Input:          pSrcImg         original image
                pDstImg         calculate new width and height after padding
                top_padding     padding by vertical direction with top_padding size
                left_padding    padding by horizon direction with left_padding size
Output:         NA
Return:         NA
*******************************************************************************/
void prePadding(FRAME *pSrcImg, FRAME *pDstImg, int *top_padding, int *left_padding)
{

    pDstImg->width = pSrcImg->width;
    pDstImg->height = pSrcImg->height;
    if (pSrcImg->width * DEMO_ABGR_BIN_READ_Y < pSrcImg->height * DEMO_ABGR_BIN_READ_X)
    {
        /* padding 0 to Width */
        pDstImg->width = pSrcImg->height * DEMO_ABGR_BIN_READ_X / DEMO_ABGR_BIN_READ_Y;
        *left_padding = (pDstImg->width - pSrcImg->width) / 2;
    }
    else if(pSrcImg->width * DEMO_ABGR_BIN_READ_Y > pSrcImg->height * DEMO_ABGR_BIN_READ_X)
    {
        /* padding 0 to height */
        pDstImg->height = pSrcImg->width * DEMO_ABGR_BIN_READ_Y / DEMO_ABGR_BIN_READ_X;
        *top_padding = (pDstImg->height - pSrcImg->height) / 2;
    }
    pDstImg->pStride[0] = pDstImg->width << 2;
}

/*******************************************************************************
Function:       processPadding
Description:    padding black pixels to original image
Input:          pSrcImg         original image
                pDstImg         save the padded image
                top_padding     padding by vertical direction with top_padding size
                left_padding    padding by horizon direction with left_padding size
Output:         NA
Return:         NA
*******************************************************************************/
int processPadding(FRAME *pSrcImg, FRAME *pDstImg, int top_padding, int left_padding)
{
    if ((!pSrcImg) || (!pDstImg))
    {
        return ERR;
    }

    unsigned int u32DstX = 0; 
    unsigned int u32DstY = 0;
    unsigned int u32SrcX = 0;
    unsigned int u32SrcY = 0;

    unsigned int ori_width = pSrcImg->width;
    unsigned int ori_height = pSrcImg->height;
    unsigned int dst_width = pDstImg->width;
    unsigned int dst_height = pDstImg->height;

    unsigned char *srcpixels = NULL;
    unsigned char *dstpixels = NULL;
    srcpixels = pSrcImg->pAddr[0];
    dstpixels = pDstImg->pAddr[0];

    unsigned int ch = 0;
    if ((left_padding) && (!top_padding))
    {
        for (u32DstY = 0; u32DstY < dst_height; ++u32DstY)
        {
            for (u32DstX = left_padding; u32DstX < left_padding + ori_width; ++u32DstX)
            {
                u32SrcX = u32DstX - left_padding;
                for (ch = 0; ch < 3; ++ch)
                {
                    dstpixels[u32DstY * pDstImg->pStride[0] + u32DstX * 4 + ch] = 
                    srcpixels[u32DstY * pSrcImg->pStride[0] + u32SrcX * 4 + ch];
                }
            }
        }
    }
    else if ((!left_padding) && (top_padding))
    {
        for (u32DstY = top_padding; u32DstY < top_padding + ori_height; ++u32DstY)
        {
            for (u32DstX = 0; u32DstX < dst_width; ++u32DstX)
            {
                u32SrcY = u32DstY - top_padding;
                for (ch = 0; ch < 3; ++ch)
                {
                    dstpixels[u32DstY * pDstImg->pStride[0] + u32DstX * 4 + ch] = 
                    srcpixels[u32SrcY * pSrcImg->pStride[0] + u32DstX * 4 + ch];
                }
            }
        }        
    }
    else if((!left_padding) && (!top_padding))
    {
        printf("original input is already 320X240, we do nothing except copy.\n");
        for (u32DstY = 0; u32DstY < dst_height; ++u32DstY)
        {
            for (u32DstX = 0; u32DstX < dst_width; ++u32DstX)
            {
                for (ch = 0; ch < 3; ++ch)
                {
                    dstpixels[u32DstY * pDstImg->pStride[0] + u32DstX * 4 + ch] = 
                    srcpixels[u32DstY * pSrcImg->pStride[0] + u32DstX * 4 + ch];
                }
            }
        } 
    }
    else
    {
        printf("sth wrong.\n");
        return ERR;
    }
    return OK;
}

/*******************************************************************************
Function:       resizeBL
Description:    resize image with bilinear interpolation, dst image size should
                be less than RESIZE_MAX_DST_SIZE
Input:          pSrcImg image before resize
                pDstImg image after resize, buffer should be alloc before call this 
Output:         pDstImg image after resize
Return:         0, success
                -1,failed
*******************************************************************************/
int resizeBL(FRAME *pSrcImg, FRAME *pDstImg)
{
    if ((!pSrcImg) || (!pDstImg) || (pDstImg->width > RESIZE_MAX_DST_SIZE) || (pDstImg->height > RESIZE_MAX_DST_SIZE))
    {
        return ERR;
    }

    unsigned char *pDst = pDstImg->pAddr[0];
    unsigned int u32DstX = 0;                               //目的图片像素索引
    unsigned int u32DstY = 0;
    unsigned int u32SrcX[RESIZE_MAX_DST_SIZE] = {0};        //对应源像素位置整数部分
    unsigned int u32SrcY = 0;
    unsigned short u16HorVal[RESIZE_MAX_DST_SIZE] = {0};    //对应源图片位置小数部分
    unsigned short u16VerVal = 0;
    unsigned int u32Tmp = 0;
    unsigned short u16Dst0 = 0;
    unsigned short u16Dst1 = 0;
    unsigned int u32Idx0 = 0;
    unsigned int u32Idx1 = 0;
    unsigned int u32Idx2 = 0;
    unsigned int u32XBound = 0;
    unsigned int u32YBound = 0;
    
    unsigned short u16ScaleHor = (pSrcImg->width << RESIZE_PRECI_LEVEL) / pDstImg->width;
    unsigned short u16ScaleVer = (pSrcImg->height << RESIZE_PRECI_LEVEL) / pDstImg->height;

    unsigned char *pSrcTmp0 = NULL;                         //源图像一行像素值
    unsigned char *pSrcTmp1 = NULL;
    unsigned char u8ChnId = 0;

    u32XBound = (pSrcImg->width - 1) << 2;
    u32YBound = pSrcImg->height - 1;    

    /* 预先计算目标像素对应在源图像的水平位置 */
    for (u32DstX = 0; u32DstX < pDstImg->width; ++u32DstX)
    {
        u32Tmp = u32DstX * u16ScaleHor;
        u32SrcX[u32DstX] = (u32Tmp >> RESIZE_PRECI_LEVEL) << 2;
        u16HorVal[u32DstX] = (u32Tmp & RESIZE_PRECI_MASK) * (u32SrcX[u32DstX] < u32XBound);
    }

    for (u32DstY = 0; u32DstY < pDstImg->height; ++u32DstY)
    {
        u32Tmp = u32DstY * u16ScaleVer;
        u32SrcY = u32Tmp >> RESIZE_PRECI_LEVEL;
        u16VerVal = (u32Tmp & RESIZE_PRECI_MASK) * (u32SrcY < u32YBound);

        pSrcTmp0 = pSrcImg->pAddr[0] + u32SrcY * pSrcImg->pStride[0];
        pSrcTmp1 = pSrcTmp0 + pSrcImg->pStride[0] * (u32SrcY < u32YBound);
        
        for (u32DstX = 0; u32DstX < pDstImg->width; ++u32DstX)
        {
            u32Idx0 = u32SrcX[u32DstX];
            u32Idx1 = u32Idx0 + 4 * (u32Idx0 < u32XBound);
            u32Idx2 = u32DstX << 2;
            for (u8ChnId = 0; u8ChnId < 3; ++u8ChnId)
            {
                /* 水平行1插值 & 水平行2插值 & 列插值 */
                u16Dst0 = pSrcTmp0[u32Idx0] + ((unsigned int)u16HorVal[u32DstX] * 
                    ((unsigned int)pSrcTmp0[u32Idx1] - pSrcTmp0[u32Idx0]) >> RESIZE_PRECI_LEVEL);
                u16Dst1 = pSrcTmp1[u32Idx0] + ((unsigned int)u16HorVal[u32DstX] * 
                    ((unsigned int)pSrcTmp1[u32Idx1] - pSrcTmp1[u32Idx0]) >> RESIZE_PRECI_LEVEL);
                pDst[u32Idx2] = u16Dst0 + ((unsigned int)u16VerVal * 
                    ((unsigned int)u16Dst1 - u16Dst0) >> RESIZE_PRECI_LEVEL);

                ++u32Idx0;
                ++u32Idx1;
                ++u32Idx2;
            }
        }
        pDst += pDstImg->pStride[0];
    }
    return OK;
}