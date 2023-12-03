#include <stdio.h>
#include <string.h>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <sys/time.h>

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

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char *argv[])
{
  char *model_path  = argv[1];
  char *image_path  = argv[2];
  
  rknn_context context;
  rknn_init(&context, model_path, 0, 0,NULL);

  rknn_input_output_num io_num;
  rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_input_output_num));

  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  cv::cvtColor(img,img,cv::COLOR_BGR2RGB);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size = img.cols * img.rows * img.channels();
  inputs[0].pass_through = 0;
  inputs[0].buf = img.data;

  rknn_inputs_set(context, io_num.n_input, inputs);

    // Run
  printf("Begin perf ...\n");
  
  int64_t start_us  = getCurrentTimeUs();
  int ret               = rknn_run(context, NULL);
  int64_t elapse_us = getCurrentTimeUs() - start_us;
  if (ret < 0) {
    printf("rknn run error %d\n", ret);
    return -1;
  }
  printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", 0, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);

  rknn_output outputs[1];
  memset(outputs, 0, sizeof(outputs));
  outputs[0].index = 0;
  outputs[0].is_prealloc = 0;
  outputs[0].want_float = 1;
  rknn_outputs_get(context, io_num.n_output, outputs, NULL);

    // Post Process
  for (int i = 0; i < io_num.n_output; i++) 
  {
    uint32_t MaxClass[5];
    float    fMaxProb[5];
    float*   buffer = (float*)outputs[i].buf;
    uint32_t sz     = outputs[i].size / 4;

    rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);

    printf(" --- Top5 ---\n");
    for (int i = 0; i < 5; i++) {
      printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
  }

  rknn_outputs_release(context, io_num.n_output, outputs);
  rknn_destroy(context);
  return 0;
}
