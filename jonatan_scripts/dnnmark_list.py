BENCHMARKS = {
    "softmax": (
        "build/benchmarks/test_fwd_softmax/dnnmark_test_fwd_softmax",
        "-config config_example/softmax_config.dnnmark -mmap mmap.bin",
    ),
    "alexnet": (
        "build/benchmarks/test_alexnet/dnnmark_test_alexnet",
        "-config config_example/alexnet.dnnmark -mmap mmap.bin",
    ),
    "bn": (
        "build/benchmarks/test_fwd_bn/dnnmark_test_fwd_bn",
        "-config config_example/bn_config.dnnmark -mmap mmap.bin",
    ),
    "bypass": (
        "build/benchmarks/test_fwd_bypass/dnnmark_test_fwd_bypass",
        "-config config_example/bypass_config.dnnmark -mmap mmap.bin",
    ),
    "activation": (
        "build/benchmarks/test_fwd_activation/dnnmark_test_fwd_activation",
        "-config config_example/activation_config.dnnmark -mmap mmap.bin",
    ),
    "composed_model": (
        "build/benchmarks/test_fwd_composed_model/dnnmark_test_fwd_composed_model",
        "-config config_example/composed_model_config.dnnmark -mmap mmap.bin",
    ),
    "conv": (
        "build/benchmarks/test_fwd_conv/dnnmark_test_fwd_conv",
        "-config config_example/conv_config.dnnmark -mmap mmap.bin",
    ),
    "dropout": (
        "build/benchmarks/test_fwd_dropout/dnnmark_test_fwd_dropout",
        "-config config_example/dropout_config.dnnmark -mmap mmap.bin",
    ),
    "fc": (
        "build/benchmarks/test_fwd_fc/dnnmark_test_fwd_fc",
        "-config config_example/fc_config.dnnmark -mmap mmap.bin",
    ),
    "lrn": (
        "build/benchmarks/test_fwd_lrn/dnnmark_test_fwd_lrn",
        "-config config_example/lrn_config.dnnmark -mmap mmap.bin",
    ),
    "pool": (
        "build/benchmarks/test_fwd_pool/dnnmark_test_fwd_pool",
        "-config config_example/pool_config.dnnmark -mmap mmap.bin",
    ),
    "VGG": (
        "build/benchmarks/test_VGG/dnnmark_test_VGG",
        "-config config_example/VGG.dnnmark -mmap mmap.bin",
    ),
}
