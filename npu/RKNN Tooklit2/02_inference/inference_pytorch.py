from rknn.api import RKNN
import cv2
import numpy as np


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[123.675,116.28,103.53]],
        std_values=[[58.395, 58.395, 58.395]],
        target_platform="rk3588"
    )

    rknn.load_pytorch(model="./resnet18.pt", input_size_list=[[1, 3, 224, 224]])

    rknn.build(
        do_quantization=True,
        dataset="dataset.txt",
    )

    rknn.export_rknn(export_path="resnet18.rknn")

    # 调用init_runtime接口初始化运行时环境
    rknn.init_runtime(
        target="rk3588",  # target 表示RKNN模型运行平台
        target_sub_class=None,
        device_id=None,
        perf_debug=False,  # perf_debug 设置为True，可以打开性能评估的debug模式
        eval_mem=False,  # eval_mem设置为True，表示使能内存评估模式
        async_mode=False,  # 表示是否使能异步模式
        core_mask=RKNN.NPU_CORE_AUTO,  # 可以设置运行时NPU的核心
    )

    # 使用opencv获取要推理的图片数据
    img = cv2.imread(
        filename="./space_shuttle_224.jpg", # 表示要读取的赌片路径
        flags=None
    )
    # cvtColor 继续宁数据格式转化
    cv2.cvtColor(
        src=img,  # 表示要转换的数据
        code=cv2.COLOR_BGR2RGB,  # code表示转换码
    )

    # 调用inference接口进行推理测试
    outputs = rknn.inference(
        inputs=[img],  # inputs表示要推理的数据
        data_format="nhwc", #data_format表示要推理的数据模式
    )

    # 对outputs进行后处理
    show_outputs(softmax(np.array(outputs[0][0])))

    rknn.release()