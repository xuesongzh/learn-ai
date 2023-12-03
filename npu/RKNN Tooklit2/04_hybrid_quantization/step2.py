from rknn.api import RKNN

if __name__ == '__main__':
    rknn = RKNN()

    # 调用hybrid_quantization_step2接口进行混合量化的第二个步骤
    rknn.hybrid_quantization_step2(
        model_input="resnet18.model",  # model_input表示第一部生成的模型文件
        data_input="resnet18.data",  # data_input表示第一步生成的配置文件
        model_quantization_cfg="resnet18.quantization.cfg",  # model_quantization_cfg表示第一步生成的量化配置文件
    )

    # 调用量化精度分析接口【评估RKNN模型
    rknn.accuracy_analysis(
        inputs=["space_shuttle_224.jpg"],
        output_dir="./snapshot",
        target="rk3588"
    )

    # 调用RKNN模型导出接口导出RKNN模型
    rknn.export_rknn(export_path="./resnet18.rknn")
    rknn.release()