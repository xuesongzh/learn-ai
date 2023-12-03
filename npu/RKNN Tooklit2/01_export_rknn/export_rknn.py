from rknn.api import RKNN

if __name__ == '__main__':
    rknn = RKNN(verbose=True, verbose_file="log.txt")

    # 调用config接口配置要生成的RKNN模型
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],  # mean_values表示预处理要减去的均值化参数
        std_values=[[58.395, 58.395, 58.395]],  # std_values 表示预处理要除的标准化参数
        quantized_dtype="asymmetric_quantized-8",  # quantized_dtype表示量化类型
        quantized_algorithm='normal',  # quantized_algorithm表示量化的算法
        quantized_method='channel',  # quantized_method 表示量化的方式
        quant_img_RGB2BGR=False,  #
        target_platform="rk3588", # target_platform表示RKNN模型的运行平台
        float_dtype="float16",  # float_dtype表示RKNNM哦性中的默认浮点数类型
        optimization_level=3, # optimization_level表示模型优化等级
        custom_string="this is my rknn model ",  # custom_string表示添加的自定义字符串信息
        remove_weight=False,  # remove_weight表示生成一个去除权重信息的从模型
        compress_weight=False, # compress_weight表示压缩模型权重，可以减小模型体积
        inputs_yuv_fmt=None, # 表示RKNN模型输入数据的YUV格式
        single_core_mode=False,  #表示构建的RKNN模型运行在单核心模式，只适用于RK3588
    )
    # 添加load_xxx接口进行常用深度学习模型的导入
    rknn.load_pytorch(
        model="./resnet18.pt",  # model表示加载模型的地址
        input_size_list=[[1, 3, 224, 224]],  # input_size_list表示模型输入节点对应图片的尺寸和通道数
    )

    # 使用build接口来构建RKNN模型
    rknn.build(
        do_quantization=True,  # do_quantization 表示是否对RKNN模型进行量化操作，
        dataset="dataset.txt",  # dataset 表示要量化的图片
        rknn_batch_size=-1  #
    )

    # 调用export_rknn接口导出RKNN模型
    rknn.export_rknn(
        export_path="resnet18.rknn" # export_path表示到处的RKNN模型路径
    )
    rknn.release()
