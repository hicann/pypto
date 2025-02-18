/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_transpose_debug.cpp
 * \brief
 */

#ifdef ENABLE_IMPORT_PYTHON

#include "test_suite_stest_ops.h"
#include <regex>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "machine/mem.h"
#include "test_common.h"
#include "operator/models/llama/llama_def.h"
#include "interface/inner/tilefwk/tilefwk_api.h"
#include <Python.h>
#include "test_dev_func_runner.h"
namespace {
int capacity;
PyObject *pFunc;
PyObject *pModule;
}

using namespace npu::tile_fwk;

class TransposeDebugTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

int python(std::string goldenPath, std::string goldenName, std::string caseName) {
    // 1️⃣ 初始化 Python 解释器
    Py_Initialize();

    // 2️⃣ 执行简单的 Python 代码
    PyRun_SimpleString("print('Hello from Python!')");

    // 3️⃣ 导入 Python 脚本
    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString(goldenPath.c_str()));
    pModule = PyImport_ImportModule(goldenName.c_str());  // 加载 script.py
    if (!pModule) {
        std::cerr << "Failed to load script.py\n";
        Py_Finalize();
        return 1;
    }

    // 4️⃣ 获取 Python 函数
    pFunc = PyObject_GetAttrString(pModule, caseName.c_str());
    if (!pFunc || !PyCallable_Check(pFunc)) {
        std::cerr << "Cannot find function 'add'\n";
        Py_DECREF(pModule);
        Py_Finalize();
        return 1;
    }
    // return pFunc;
    // 5️⃣ 调用 Python 函数
    return 0;
}

int finishPython(PyObject* args){

    PyObject* pValue = PyObject_CallObject(pFunc, args);  // 执行 add(10, 20)

    if (pValue) {
        long result = PyLong_AsLong(pValue);
        std::cout << "Result from Python: " << result << std::endl;  // 输出 30
        Py_DECREF(pValue);
    } else {
        std::cerr << "Function call failed!\n";
    }

    // 6️⃣ 释放资源
    Py_DECREF(args);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    // 7️⃣ 关闭 Python 解释器
    Py_Finalize();

    return 0;
}

void TransposePre(uint8_t** out_ptr, uint64_t* outsize) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    *outsize = capacity * sizeof(float);
    *out_ptr = allocDevAddr(*outsize);
}

void TransposePost(uint8_t* outputGmAddr, uint64_t outputSize) {
    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    std::vector<float> input(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outputGmAddr, outputSize);
    readInput("res.bin", golden);
    readInput("input.bin", input);
    int ret = resultCmp(golden, res, 0.001f, 64);
    EXPECT_EQ(ret, true);
}


TEST_F(TransposeDebugTest, TestTranspose_BNSD_BSND) {
    int b = 2;
    int n = 32;
    int s = 16;
    int d = 16;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 16, 16, 16);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNSD_BSND", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_ABC_BAC) {
    int bs = 128;
    int n = 2;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_ABC_BAC");
    PyObject *args = PyTuple_Pack(3, PyLong_FromLong(bs), PyLong_FromLong(n), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(32, 1, 128);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ABC_BAC", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_BNSD2_BNS2D_small) {
    int b = 2;
    int n = 4;
    int s = 32;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d/2, 2};
    std::vector<int64_t> resShape{b, n, s, 2, d/2};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD2_BNS2D_small");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 2, 32, 32, 2);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNSD2_BNS2D_small", {input, output}) {
            output = Transpose(input, {3, 4});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_ROPE_5D) {
    int b = 4;
    int n = 64;
    int s = 1;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d/2, 2};
    std::vector<int64_t> resShape{b, n, s, 2, d/2};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD2_BNS2D_small");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 2, 1, 32, 2);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ROPE_5D", {input, output}) {
            output = Transpose(input, {3, 4});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

// ok
TEST_F(TransposeDebugTest, TestTranspose_MLA_3D_0) {
    int bs = 32;
    int n = 32;
    int d = 64;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_ABC_BAC");
    PyObject *args = PyTuple_Pack(3, PyLong_FromLong(bs), PyLong_FromLong(n), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(4, 4, 64);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_3D_1) {
    int bs = 32;
    int n = 32;
    int d = 512;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_ABC_BAC");
    PyObject *args = PyTuple_Pack(3, PyLong_FromLong(bs), PyLong_FromLong(n), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(2, 2, 512);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_1", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_0) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 512);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_1) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 64);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_1", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_2) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 64);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_2", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_3) {
    int b = 32;
    int n = 1;
    int s = 1;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_ADD_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        // TileShape::Current().SetVecTile(1, 1, 32, 64);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_3", {input, output}) {
            TileShape::Current().SetVecTile(8, 1, 1, 8);
            output = Add(input, input);
            output = Transpose(output, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_4) {
    int b = 32;
    int n = 1;
    int s = 1;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_ADD_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 512);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_4", {input, output}) {
            TileShape::Current().SetVecTile(8, 1, 1, 128);
            output = Add(input, input);
            output = Transpose(output, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

// error
TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_5) {
    int b = 32;
    int n = 1;
    int s = 256;
    int d = 512 + 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNDS_BNSD");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 16, d);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_5", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_50) {
    int b = 1;
    int n = 1;
    int s = 128;
    int d = 128;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNDS_BNSD");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, s, d);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_50", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_51) {
    int b = 1;
    int n = 1;
    int s = 256;
    int d = 512 + 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNDS_BNSD");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 16, d);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_51", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

// ok
TEST_F(TransposeDebugTest, TestTranspose_MLA_4D_6) {
    int b = 32;
    int n = 32;
    int s = 1;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNSD_BSND");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        //  1,1,1,512 ok
        TileShape::Current().SetVecTile(4, 4, 1, 512);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_6", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

//ok
TEST_F(TransposeDebugTest, TestTranspose_MLA_3D_2) {
    int bs = 32;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_ABC_BAC");
    PyObject *args = PyTuple_Pack(3, PyLong_FromLong(bs), PyLong_FromLong(n), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(2, 2, 128);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_2", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeDebugTest, TestTranspose_BNDS_BNSD) {
    int b = 2;
    int n = 32;
    int s = 32;
    int d = 32;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, n, d, s};
    capacity = b * n * s * d;
    python("../tests/script/golden/op", "transpose_operator", "TestTranspose_DEBUG_BNDS_BNSD");
    PyObject *args = PyTuple_Pack(4, PyLong_FromLong(b), PyLong_FromLong(n), PyLong_FromLong(s), PyLong_FromLong(d));
    finishPython(args);
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 16, 16, 16);
        void *input_ptr = readToDev("input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNDS_BNSD", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

#endif
