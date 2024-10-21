BENCHMARKS = {
    # Softmax runs fine
    "softmax": (
        "build/benchmarks/test_fwd_softmax/dnnmark_test_fwd_softmax",
        "-config config_example/softmax_config.dnnmark -mmap mmap.bin",
    ),
    # Alexnet runs with 0 shaderActiveTicks
    # "alexnet": (
    #     "build/benchmarks/test_alexnet/dnnmark_test_alexnet",
    #     "-config config_example/alexnet.dnnmark -mmap mmap.bin",
    # ),
    # BN runs fine
    "bn": (
        "build/benchmarks/test_fwd_bn/dnnmark_test_fwd_bn",
        "-config config_example/bn_config.dnnmark -mmap mmap.bin",
    ),
    # Bypass runs fine
    "bypass": (
        "build/benchmarks/test_fwd_bypass/dnnmark_test_fwd_bypass",
        "-config config_example/bypass_config.dnnmark -mmap mmap.bin",
    ),
    # Activation runs fine
    "activation": (
        "build/benchmarks/test_fwd_activation/dnnmark_test_fwd_activation",
        "-config config_example/activation_config.dnnmark -mmap mmap.bin",
    ),
    # Composed runs with 0 shaderActiveTicks
    # "composed_model": (
    #     "build/benchmarks/test_fwd_composed_model/dnnmark_test_fwd_composed_model",
    #     "-config config_example/composed_model_config.dnnmark -mmap mmap.bin",
    # ),
    # Conv runs with 0 shaderActiveTicks
    # "conv": (
    #     "build/benchmarks/test_fwd_conv/dnnmark_test_fwd_conv",
    #     "-config config_example/conv_config.dnnmark -mmap mmap.bin",
    # ),
    # Dropout runs with 0 shaderActiveTicks
    # "dropout": (
    #     "build/benchmarks/test_fwd_dropout/dnnmark_test_fwd_dropout",
    #     "-config config_example/dropout_config.dnnmark -mmap mmap.bin",
    # ),
    # FC dies with <Signals.SIGABRT: 6>
    # "fc": (
    #     "build/benchmarks/test_fwd_fc/dnnmark_test_fwd_fc",
    #     "-config config_example/fc_config.dnnmark -mmap mmap.bin",
    # ),
    # LRN runs with 0 shaderActiveTicks
    # "lrn": (
    #     "build/benchmarks/test_fwd_lrn/dnnmark_test_fwd_lrn",
    #     "-config config_example/lrn_config.dnnmark -mmap mmap.bin",
    # ),
    # Pool runs fine
    "pool": (
        "build/benchmarks/test_fwd_pool/dnnmark_test_fwd_pool",
        "-config config_example/pool_config.dnnmark -mmap mmap.bin",
    ),
    # VGG runs with 0 shaderActiveTicks
    # "VGG": (
    #     "build/benchmarks/test_VGG/dnnmark_test_VGG",
    #     "-config config_example/VGG.dnnmark -mmap mmap.bin",
    # ),
}
