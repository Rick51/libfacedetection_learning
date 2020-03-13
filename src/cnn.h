#pragma once

//#define _ENABLE_AVX2 //Please enable it if X64 CPU
//#define _ENABLE_NEON //Please enable it if ARM CPU




//DO NOT EDIT the following code if you don't really understand it.

#if defined(_ENABLE_AVX2)
#include <immintrin.h>
#endif

#if defined(_ENABLE_NEON)
#include "arm_neon.h"
#define _ENABLE_INT8_CONV
#endif

#if defined(_ENABLE_AVX2) 
#define _MALLOC_ALIGN 256
#else
#define _MALLOC_ALIGN 128
#endif

#if defined(_ENABLE_AVX2) && defined(_ENABLE_NEON)
#error Cannot enable the two of SSE2 AVX and NEON at the same time.
#endif


#if defined(_OPENMP)
#include <omp.h>
#endif

#include <string.h>
#include <vector>
#include <iostream>

using namespace std;

void* myAlloc(size_t size);
void myFree_(void* ptr);

#define myFree(ptr) (myFree_(*(ptr)), *(ptr)=0);
#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifndef SUM
#  define SUM(a,b)  ((a) + (b))
#endif

#ifndef ABS
#  define ABS(a,b)  ((a)-(b) > 0 ? (a)-(b) : (b)-(a))
#endif

typedef struct ObjectRect_
{
    float score;
    int xmin;
    int ymin;
    int width;
    int height;
}ObjectRect;

typedef struct FaceInfo_
{
    float yaw;
    float pitch;
    float roll;
    float lefteye_x;
    float righteye_x;
    float nose_x;
    float leftmouth_x;
    float rightmouth_x;
    float lefteye_y;
    float righteye_y;
    float nose_y;
    float leftmouth_y;
    float rightmouth_y;
    float backhead;
    float frontface;
}FaceInfo;

typedef struct PriorboxParam_
{
    int *min_size;
    int *max_size;
    float *aspect_ratio;
    bool flip;
    bool clip;
    float offset;
    //float step;
    //float step_h;
    //float step_w;
    //float img_h;
    //float img_w;
}PriorboxParam;


class CDataBlob
{
public:
    float * data_float;
    signed char * data_int8;
    int width;
    int height;
    int channels;
    int floatChannelStepInByte;
    int int8ChannelStepInByte;
    float int8float_scale;
    bool int8_data_valid;
public:
    CDataBlob() {
        data_float = 0;
        data_int8 = 0;
        width = 0;
        height = 0;
        channels = 0;
        floatChannelStepInByte = 0;
        int8ChannelStepInByte = 0;
        int8float_scale = 1.0f;
        int8_data_valid = false;
    }
    CDataBlob(int w, int h, int c)
    {
        data_float = 0;
        data_int8 = 0;
        create(w, h, c);
    }
    ~CDataBlob()
    {
        setNULL();
    }

    void setNULL()
    {
        if (data_float)
            myFree(&data_float);
        if (data_int8)
            myFree(&data_int8);
        width = height = channels = floatChannelStepInByte = int8ChannelStepInByte = 0;
        int8float_scale = 1.0f;
        int8_data_valid = false;
    }
    bool create(int w, int h, int c)
    {
        setNULL();

        width = w;
        height = h;
        channels = c;
        //alloc space for float array
        int remBytes = (sizeof(float)* channels) % (_MALLOC_ALIGN / 8);
        if (remBytes == 0)
            floatChannelStepInByte = channels * sizeof(float);
        else
            floatChannelStepInByte = (channels * sizeof(float)) + (_MALLOC_ALIGN / 8) - remBytes;
        data_float = (float*)myAlloc(width * height * floatChannelStepInByte);

        //alloc space for int8 array
        remBytes = (sizeof(char)* channels) % (_MALLOC_ALIGN / 8);
        if (remBytes == 0)
            int8ChannelStepInByte = channels * sizeof(char);
        else
            int8ChannelStepInByte = (channels * sizeof(char)) + (_MALLOC_ALIGN / 8) - remBytes;
        data_int8 = (signed char*)myAlloc(width * height * int8ChannelStepInByte);

        if (data_float == NULL)
        {
            cerr << "Cannot alloc memeory for float data blob: " 
                << width  << "*"
                << height << "*"
                << channels << endl;
            return false;
        }

        if (data_int8 == NULL)
        {
            cerr << "Cannot alloc memeory for uint8 data blob: "
                << width << "*"
                << height << "*"
                << channels << endl;
            return false;
        }
        
        for (int r = 0; r < this->height; r++)
        {
            for (int c = 0; c < this->width; c++)
            {
                int pixel_end = this->floatChannelStepInByte / sizeof(float);
                float * pF = (float*)(this->data_float + (r * this->width + c) * this->floatChannelStepInByte/sizeof(float));
                for (int ch = this->channels; ch < pixel_end; ch++)
                    pF[ch] = 0;

                pixel_end = this->int8ChannelStepInByte / sizeof(char);
                char * pI = (char*)(this->data_int8 + (r * this->width + c) * this->int8ChannelStepInByte/sizeof(char));
                for (int ch = this->channels; ch < pixel_end; ch++)
                    pI[ch] = 0;
            }
        }
        
        return true;
    }

    bool setInt8DataFromCaffeFormat(signed char * pData, int dataWidth, int dataHeight, int dataChannels)
    {
        if (pData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }
        if (dataWidth != this->width ||
            dataHeight != this->height ||
            dataChannels != this->channels)
        {
            cerr << "The dim of the data can not match that of the Blob." << endl;
            return false;
        }
        
        for(int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
            {
                signed char * p = (this->data_int8 + (width * row + col) * int8ChannelStepInByte /sizeof(char));
                for (int ch = 0; ch < channels; ch++)
                {
                    p[ch] = pData[ch * height * width + row * width + col];
                }
            }
        return true;
    }
    bool setFloatDataFromCaffeFormat(float * pData, int dataWidth, int dataHeight, int dataChannels)
    {
        if (pData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }
        if (dataWidth != this->width ||
            dataHeight != this->height ||
            dataChannels != this->channels)
        {
            cerr << "The dim of the data can not match that of the Blob." << endl;
            return false;
        }
        
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
            {
                float * p = (this->data_float + (width * row + col) * floatChannelStepInByte / sizeof(float));
                for (int ch = 0; ch < channels; ch++)
                {
                    p[ch] = pData[ch * height * width + row * width + col];
                }
            }
        return true;
    }

    bool setDataFromImage(const unsigned char * imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep,
        float * pChannelMean, float * pChannelScale)
    {
        if (imgData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }
        if (pChannelMean == NULL)
        {
            cerr << "The mean values is null." << endl;
            return false;
        }
        if (pChannelScale == NULL)
        {
            cerr << "The scale values is null." << endl;
            return false;
        }

        create(imgWidth, imgHeight, 3);

        for (int r = 0; r < imgHeight; r++)
        {
            for (int c = 0; c < imgWidth; c++)
            {
                const unsigned char * pImgData = imgData + imgWidthStep * r + imgChannels * c;
                float * pBlobData = this->data_float + (this->width * r + c) * this->floatChannelStepInByte /sizeof(float);
                //for (int ch = 0; ch < imgChannels; ch++)
                //    pBlobData[ch] = ((float)(pImgData[ch] - pChannelMean[ch])) / pChannelScale[ch];

                //rgb -> bgr
                pBlobData[0] = ((float)(pImgData[2] - pChannelMean[2])) / pChannelScale[2];
                pBlobData[1] = ((float)(pImgData[1] - pChannelMean[1])) / pChannelScale[1];
                pBlobData[2] = ((float)(pImgData[0] - pChannelMean[0])) / pChannelScale[0];
            }
        }
        return true;
    }
    /*针对第一层卷积是3X3S2P1的格式，特殊加速处理*/
    /*思想是将输入图像按照特定方式展开，方便与conv的权重做点乘的时候不需要重复读写（因为S=2，K=3造成的）重叠的数据*/
    bool setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char * imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep,
        float * pChannelMean, float * pScale)
    {
        if (imgData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }
        if (pChannelMean == NULL)
        {
            cerr << "The mean values is null." << endl;
            return false;
        }
        
        if (imgChannels != 3 && imgChannels != 4)
        {
            cerr << "The input image must be a 3-channel RGB or 4-channel RGBA image." << endl;
            return false;
        }
        create((imgWidth+1)/2, (imgHeight+1)/2, 27);
        memset(data_float, 0, width * height * floatChannelStepInByte);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int r = 0; r < this->height; r++)
        {
            for (int c = 0; c < this->width; c++)
            {
                float * pData = this->data_float + (r * this->width + c) * this->floatChannelStepInByte / sizeof(float);
                for (int fy = -1; fy <= 1; fy++)
                {
                    int srcy = r * 2 + fy;
                    
                    if (srcy < 0 || srcy >= imgHeight) //out of the range of the image
                        continue;

                    for (int fx = -1; fx <= 1; fx++)
                    {
                        int srcx = c * 2 + fx;

                        if (srcx < 0 || srcx >= imgWidth) //out of the range of the image
                            continue;

                        const unsigned char * pImgData = imgData + imgWidthStep * srcy + imgChannels * srcx;

                        int output_channel_offset = ((fy + 1) * 3 + fx + 1) * 3; //3x3 filters, 3-channel image
                        // rgb->bgr here
                        pData[output_channel_offset] = ((float)(pImgData[2] - pChannelMean[2]))/pScale[2];
                        pData[output_channel_offset+1] = ((float)(pImgData[1] - pChannelMean[1]))/pScale[1];
                        pData[output_channel_offset+2] = ((float)(pImgData[0] - pChannelMean[0]))/pScale[0];
                    }
                }
            }
        }
        return true;
    }
    float getElementFloat(int x, int y, int channel)
    {
        if (this->data_float)
        {
            if (x >= 0 && x < this->width &&
                y >= 0 && y < this->height &&
                channel >= 0 && channel < this->channels)
            {
                float * p = (float*)(this->data_float + (y*this->width + x)*this->floatChannelStepInByte / sizeof(float));
                return p[channel];
            }
         }
        
        return 0.f;
    }
    int getElementint8(int x, int y, int channel)
    {
        if (this->data_int8 && this->int8_data_valid)
        {
            if (x >= 0 && x < this->width &&
                y >= 0 && y < this->height &&
                channel >= 0 && channel < this->channels)
            {
                signed char * p = this->data_int8 + (y*this->width + x)*this->int8ChannelStepInByte/sizeof(char);
                return p[channel];
            }
        }
        
        return 0;
    }

    friend ostream &operator<<(ostream &output, const CDataBlob &dataBlob)
    {
        output << "DataBlob Size (Width, Height, Channel) = (" 
            << dataBlob.width
            << ", " << dataBlob.height 
            << ", " << dataBlob.channels
            << ")" << endl;
        for (int ch = 0; ch < dataBlob.channels; ch++)
        {
            output << "Channel " << ch << ": " << endl;

            for (int row = 0; row < dataBlob.height; row++)
            {
                output << "(";
                for (int col = 0; col < dataBlob.width; col++)
                {
                    float * p = (dataBlob.data_float + (dataBlob.width * row + col) * dataBlob.floatChannelStepInByte/sizeof(float));
                    output << p[ch];
                    if (col != dataBlob.width - 1)
                        output << ", ";
                }
                output << ")" << endl;
            }
        }

        return output;
    }
};


typedef struct ConvInfo{
    int group;
    int pad;
    int stride;
    int width;
    int height;
    int channels;
    int num;
}ConvInfo;

typedef struct BiasInfo
{
    int width;
    int height;
    int channels;
    int num;
}BiasInfo;

typedef struct PreluInfo 
{
    int width;
    int height;
    int channels;
    int num;
}PreluInfo;

typedef struct IpInfo
{
    int width;
    int height;
    int channels;
    int num;
}IpInfo;




class Filters {
public:
    vector<CDataBlob *> filters;
    int group;
    int pad;
    int stride;
    float weight_scale; //scale for filters: weight_int8 = weight_float32 * weight_scale
    float data_scale; // scale for inputs: bottom_blob_int8 = bottom_blob_float32 * data_scale
};

class Bias {
public:
    vector<CDataBlob *> bias_term;
};

class Ips {
public:
    vector<CDataBlob *> ips;
};

class Prelus {
public:
    vector<CDataBlob *> prelus;
};


bool convolution(CDataBlob *inputData, const Filters* filters, CDataBlob *outputData);
bool addbias(CDataBlob *inputData, const Bias * bias);
bool prelu(CDataBlob *inputData, const Prelus * prelus);
bool maxpooling2x2S2(const CDataBlob *inputData, CDataBlob *outputData);
bool maxpooling3x3S2(const CDataBlob *inputData, CDataBlob *outputData);
bool avepooling3x3P1S1(const CDataBlob *inputData, CDataBlob *outputData);
bool innerproduct(const CDataBlob * inputData, Ips * ips, CDataBlob * outputData);
bool concat2(const CDataBlob *inputData1, const CDataBlob *inputData2, CDataBlob *outputData);
bool concat4(const CDataBlob *inputData1, const CDataBlob *inputData2, const CDataBlob *inputData3, const CDataBlob *inputData4, CDataBlob *outputData);
bool concat7(const CDataBlob *inputData1, const CDataBlob *inputData2, const CDataBlob *inputData3, const CDataBlob *inputData4, const CDataBlob *inputData5, const CDataBlob *inputData6, const CDataBlob *inputData7, CDataBlob *outputData);
bool scale(CDataBlob * dataBlob, float scale);
bool relu(const CDataBlob *inputOutputData);
bool priorbox(const CDataBlob * featureData, const CDataBlob * imageData, int num_sizes, float * pWinSizes, CDataBlob * outputData);
bool priorbox_caffe(const CDataBlob * featureData, const CDataBlob * imageData, PriorboxParam * priorbox , int lengthOfMinsizes, int lengthOfMaxsizes, int lengthOfRatiosizes, CDataBlob * outputData);
bool normalize(CDataBlob * inputOutputData, float * pScale);
bool blob2vector(const CDataBlob * inputData, CDataBlob * outputData, bool isFloat);
bool detection_output(const CDataBlob * priorbox, const CDataBlob * loc, const CDataBlob * conf, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, CDataBlob * outputData);
bool detectionout(const CDataBlob * prior_data, const CDataBlob * loc_data, const CDataBlob * conf_data, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, int num_classes, CDataBlob * outputData);
/* the input data for softmax must be a vector, the data stored in a multi-channel blob with size 1x1 */
//bool softmax1vector(CDataBlob *inputOutputData, int numclass);
bool softmax1vector2class(const CDataBlob *inputOutputData);
vector<CDataBlob *>  splitCDataBlobToGroupChannel(CDataBlob * inputData, int group, bool isFloat);


// NET STRUCT
//yufacedetect_net(buffer memory for storing face detection results, !!its size must be 0x20000 Bytes!!
int * facedetect_cnn(unsigned char * result_buffer, unsigned char * rgb_image_data, int width, int height, int step);
vector<ObjectRect> objectdetect(unsigned char * rgbImageData, int width, int height, int step); 
//landmark detect net
float * landmarkdetect_cnn(float * result_buffer, unsigned char * rgb_image_data, int width, int height, int step);
FaceInfo FaceInfoDetectCnn(unsigned char * rgbImageData, int with, int height, int step);
//multi class detect net
int* mobilenet_detect(unsigned char *result_buffer, unsigned char * rgb_image_data, int width, int height, int step);
//RFB net
int * facedetect_RFB(unsigned char * result_buffer, unsigned char * rgb_image_data, int width, int height, int step);
vector<ObjectRect> faceDetectRFB(unsigned char* rgbImageData, int width, int height, int step);