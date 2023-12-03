#include <stdio.h>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>

using namespace cv;

static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

#define MAX_TOP_NUM 20
  if (topNum > MAX_TOP_NUM)
    return 0;

  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
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


int main(int argc, char *argv[])
{
  /*要求程序传入的第一个参数为RKNN模型、第二个参数为要推理的图片*/
  char *model_path = argv[1];
  char *image_path = argv[2];

  /*调用rknn_init接口将RKNNM哦性的运行环境和相关信息赋予到context变量中*/
  rknn_context context;
  rknn_init(&context, model_path, 0, 0, NULL);

  /*使用opencv读取要推理的图像数据*/
  cv::Mat img = cv::imread(image_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  /*调用rknn_query接口查询tensor输入输出个数*/
  rknn_input_output_num io_num;
  rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  /*调用rknn_inputs_set接口设置输入数据*/
  rknn_input input[1];
  memset(input, 0, sizeof(rknn_input));
  input[0].index = 0;
  input[0].buf = img.data;
  input[0].size = img.rows * img.cols * img.channels() *sizeof(uint8_t);
  input[0].pass_through = 0;
  input[0].type = RKNN_TENSOR_UINT8;
  input[0].fmt = RKNN_TENSOR_NHWC;
  rknn_inputs_set(context, 1, input);

  /*调用rknn_run接口进行模型推理了*/
  rknn_run(context, NULL);

  /*调用rknn_outputs_get接口获取模型推理结果*/
  rknn_output output[1];
  memset(output, 0 ,sizeof(rknn_output));
  output[0].index = 0;
  output[0].is_prealloc = 0;
  output[0].want_float = 1;
  rknn_outputs_get(context, 1, output, NULL);


   // Post Process
  for (int i = 0; i < io_num.n_output; i++) 
  {
    uint32_t MaxClass[5];
    float    fMaxProb[5];
    float*   buffer = (float*)output[i].buf;
    uint32_t sz     = output[i].size / 4;

    rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);

    printf(" --- Top5 ---\n");
    for (int i = 0; i < 5; i++) {
      printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
  }

  /*调用rknn_outputs_release接口释放推理输出相关的资源*/
  rknn_outputs_release(context, 1, output);

  /*调用rknn_destory 接口销毁context变量*/
  rknn_destroy(context);
  /* code */
  return 0;
}
