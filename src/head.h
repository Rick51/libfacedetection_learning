#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEBUG
#define OK 0
#define ERR -1

#define DEMO_ABGR_BIN_READ              "./demo.bin"
#define DEMO_ABGR_BIN_WRITE             "./scaleimage.bin"
#define DEMO_ABGR_BIN_PADDING_WRITE     "./padding.bin"
#define DEMO_ABGR_BIN_FACE_WRITE        "./face.bin"

#define MAX_FACENUM 256

#define DEMO_ABGR_BIN_READ_X 320                    /* 人脸检测模型固定输入宽度 */
#define DEMO_ABGR_BIN_READ_Y 240                    /* 人脸检测模型固定输入高度 */
#define FACE_IMG_MIN_SIZE 24                        /* 检出人脸最小长度阈值 */

//RESIZE CONFIG
#define RESIZE_MAX_DST_SIZE 400
#define RESIZE_PRECI_LEVEL 7
#define RESIZE_PRECI_MULTI (1 << RESIZE_PRECI_LEVEL)
#define RESIZE_PRECI_MASK ((1 << RESIZE_PRECI_LEVEL) - 1)

//MATH
#define FLOAT_DELTA     1e-6
#define FLOAT_ISZERO(f) (FLOAT_DELTA >= (f) && -FLOAT_DELTA <= (f))
#ifndef MAX
    #define MAX(a, b)       ((a) < (b) ? (b) : (a))
#endif
#ifndef MIN
    #define MIN(a, b)       ((a) <= (b) ? (a) : (b))
#endif
#define PI              3.14159265358979323846
#define PI_2            1.57079632679489661923
#define IN_RANGE(x, lb, ub) (((x) >= (lb)) && ((x) <= (ub)))
#define CLIPI(x, a, b)   ((x) = ((x) + ((int)(x) < (int)(a)) * ((int)(a) - (int)(x)) + \
    ((int)(x) > (int)(b)) * ((int)(b) - (int)(x))))
#define CLIPU(x, a, b) ((x) = ((x) + ((unsigned int)(x) < (unsigned int)(a)) * \
        ((unsigned int)(a) - (unsigned int)(x)) + ((unsigned int)(x) > (unsigned int)(b)) * \
        ((unsigned int)(b) - (unsigned int)(x))))
#define CLIPF(x, a, b)  ((x) = ((x) + ((float)(x) < (float)(a)) * ((float)(a) - (float)(x)) + \
    ((float)(x) > (float)(b)) * ((float)(b) - (float)(x))))
#define U32_DIFF(u1, u2)    ((int) (u1) - (int) (u2))
#define ALIGN(base, align) ((((base) + (align) - 1) & ~((align) - 1)) - align)

typedef struct FRAME_S
{
    unsigned char *pAddr[3];    //data address of all channel, for packed format only channel 0 used
    unsigned int pStride[3];    //stride of one row, for packed format only stride 0 used
    unsigned int width;         //frame width
    unsigned int height;        //frame height
}FRAME;

typedef struct RGB_IMAGE_S
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
}RGB_IMAGE;

typedef struct RECT_S
{
    int x;                          /**< 矩形起点X坐标，取值范围 */
    int y;                          /**< 矩形起点Y坐标，取值范围 */
    unsigned int width;             /**< 矩形宽度，取值范围 */
    unsigned int height;            /**< 矩形高度，取值范围 */
    float score;
}RECT;


void prePadding(FRAME *pSrcImg, FRAME *pDstImg, int *top_padding, int *left_padding);
int processPadding(FRAME *pSrcImg, FRAME *pDstImg, int top_padding, int left_padding);
int resizeBL(FRAME *pSrcImg, FRAME *pDstImg);
int faceDetectProcess(FRAME *pOrgImg, RECT *facesInfo);










