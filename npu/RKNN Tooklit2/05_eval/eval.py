from rknn.api import RKNN

if __name__ == '__main__':
    rknn = RKNN()

    # 使用load_rknn接口导入rknn模型
    rknn.load_rknn(path="./resnet18.rknn")

    # 使用init_runtime接口初始化运行时环境
    rknn.init_runtime(
        target="rk3588",
        perf_debug=True, # perf_debug表示是否开启性能评估的debug模式
        eval_mem=True,  #  eval_mem表示是否是能内存评估
    )

    # 使用eval_perf接口进行性能评估
    # rknn.eval_perf(
    #     inputs=["space_shuttle_224.jpg"],  # inputs表示要测试的图片
    #     data_format=None,  # data_format表示要推理的数据模式
    #     is_print=True, # is_print 表示使能打印性能信息
    # )
    # 使用eval_memory接口进行内存评估
    rknn.eval_memory(
        is_print=True,  # 表示使能打印内存评估信息
    )
    rknn.release()