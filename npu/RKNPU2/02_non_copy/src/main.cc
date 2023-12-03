#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"
#include "string.h"

using namespace cv;

static int rknn_GetTopN(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount,
                        uint32_t topNum) {
    uint32_t i, j;
    uint32_t top_count = outputCount > topNum ? topNum : outputCount;

    for (i = 0; i < topNum; ++i) {
        pfMaxProb[i] = -FLT_MAX;
        pMaxClass[i] = -1;
    }

    for (j = 0; j < top_count; j++) {
        for (i = 0; i < outputCount; i++) {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) ||
                (i == *(pMaxClass + 3)) || (i == *(pMaxClass + 4))) {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j)) {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

int main(int argc, char *argv[]) {
    char *model_path = argv[1];
    char *image_path = argv[2];

    rknn_context context;
    rknn_init(&context, model_path, 0, 0, NULL);

    cv::Mat img = cv::imread(image_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    /*调用rknn_query接口查询输入输出tensor属性*/
    rknn_tensor_attr input_attr[1], output_attr[1];
    memset(input_attr, 0, sizeof(rknn_tensor_attr));
    memset(output_attr, 0, sizeof(rknn_tensor_attr));
    rknn_query(context, RKNN_QUERY_INPUT_ATTR, input_attr, sizeof(input_attr));
    rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, output_attr, sizeof(output_attr));

    /*调用rknn_create_mem接口申请输入和输出数据内存*/
    rknn_tensor_mem *input_mem[1], *output_mem[1];
    input_mem[0] = rknn_create_mem(context, input_attr[0].size_with_stride);
    output_mem[0] = rknn_create_mem(context, output_attr[0].n_elems * sizeof(float));

    unsigned char *input_data = img.data;
    memcpy(input_mem[0]->virt_addr, input_data, input_attr[0].size_with_stride);

    /*调用rknn_set_io_mem让NPU使用上面申请到的内存*/
    input_attr[0].type = RKNN_TENSOR_UINT8;
    output_attr[0].type = RKNN_TENSOR_FLOAT32;
    rknn_set_io_mem(context, input_mem[0], input_attr);
    rknn_set_io_mem(context, output_mem[0], output_attr);

    /*调用rknn_run进行模型推理*/
    rknn_run(context, NULL);

    // Get top 5
    uint32_t topNum = 5;
    uint32_t MaxClass[topNum];
    float fMaxProb[topNum];
    float *buffer = (float *)output_mem[0]->virt_addr;
    uint32_t sz = output_attr[0].n_elems;
    int top_count = sz > topNum ? topNum : sz;

    rknn_GetTopN(buffer, fMaxProb, MaxClass, sz, topNum);

    printf("---- Top%d ----\n", top_count);
    for (int j = 0; j < top_count; j++) {
        printf("%8.6f - %d\n", fMaxProb[j], MaxClass[j]);
    }

    /*调用rknn_destory_mem接口销毁申请的内存*/
    rknn_destroy_mem(context, input_mem[0]);
    rknn_destroy_mem(context, output_mem[0]);

    /*调用rknn_destory销毁context*/
    rknn_destroy(context);
    /* code */
    return 0;
}
