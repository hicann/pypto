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
 * \file function.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/flow_verifier.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
constexpr int MAX_IDENT_LEVEL = 20;
const std::unordered_set<std::string> copyOpCode = {"COPY_IN", "COPY_OUT", "L1_TO_L0A",
    "L1_TO_L0B", "L1_TO_L0_AT", "L1_TO_L0_BT", "TRANSPOSE_MOVEIN", "TRANSPOSE_MOVEOUT", "INDEX_OUTCAST"};

static std::string HtmlEscape(const std::string &src, bool escapeLineBreak = true) {
    std::string ret;
    for (auto &c : src) {
        switch (c) {
            case '<': ret += "&lt;"; break;
            case '>': ret += "&gt;"; break;
            case '&': ret += "&amp;"; break;
            case '\n':
                if (escapeLineBreak) {
                    ret += "<br/>";
                }
                ret.push_back(c);
                break;
            default: ret.push_back(c);
        }
    }
    return ret;
}

void FunctionInterpreter::DumpFunctionHead(Function *func) {
    if (execDumpLevel >= EXEC_DUMP_LEVEL_OPERATION) {
        int indent = GetFrameSize();
        auto head = func->DumpSSATitle();
        auto raw = func->DumpSSARawTensor(indent);
        auto incast = func->DumpSSAIncast(indent);
        auto outcast = func->DumpSSAOutcast(indent);
        auto attr = func->DumpSSAAttribute(indent);
        auto symbol = DumpSymbolDict();
        ALOG_INFO_F("%s Function %s\n", execDumpFuncKey.c_str(), head.c_str());
        ALOG_INFO_F("%s\n", raw.c_str());
        ALOG_INFO_F("%s\n", incast.c_str());
        ALOG_INFO_F("%s\n", outcast.c_str());
        ALOG_INFO_F("%s\n", attr.c_str());
        ALOG_INFO_F("%s\n", symbol.c_str());
        if (execDumpFile) {
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, execDumpFuncKey.c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">frameIndex=%s</div>", indent, GetFrameCurrIndex().c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(head).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(raw).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(incast).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(outcast).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(attr).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(symbol).c_str());
        }
    }
}

void FunctionInterpreter::DumpOperation(Operation *op) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;
    int indent = GetFrameSize();
    auto dump = op->Dump();
    ALOG_INFO(execDumpFuncKey, " Operation: ", dump);
    if (execDumpFile) {
        std::string tensorId = GetDumpTensorId(GetFrameCurr(), op);
        std::string operationId = GetDumpOperationId(GetFrameCurr(), op);
        fprintf(execDumpFile,
            "<div class=\"indent_%d\" id=\"%s\">"
            "  <div class=\"operation\" id=\"%s\">%s</div>"
            "</div>\n",
            indent, tensorId.c_str(), operationId.c_str(), HtmlEscape(dump).c_str());
    }
}

static void DumpLine(FILE *f, const std::vector<std::string> &textList, const std::string &cssClass = "") {
    fprintf(f, "<tr>\n");
    for (size_t k = 0; k < textList.size(); k++) {
        std::string attr = " class=\"" + cssClass + "\"";
        fprintf(f, "  <td%s>%s</td>\n", attr.c_str(), textList[k].c_str());
    }
    fprintf(f, "</tr>\n");
}

static void DumpDataViewParallel(const std::shared_ptr<LogicalTensorData> &dataView,
    std::vector<ElementDump> &elementDumpList, util::ThreadPool *pool) {
    struct DumpTask {
        DumpTask(std::vector<ElementDump> *ret_, const std::shared_ptr<LogicalTensorData> view_, int indexBegin_,
            int indexEnd_)
            : ret(ret_), view(view_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

        std::vector<ElementDump> *ret;
        const std::shared_ptr<LogicalTensorData> view;
        int indexBegin;
        int indexEnd;

        static void Entry(void *c) {
            auto [ret, view, indexBegin, indexEnd] = *(DumpTask *)c;
            for (int i = indexBegin; i < indexEnd; i++) {
                view->DumpElement(i, &ret->at(i));
            }
        }
    };

    if (static_cast<int>(elementDumpList.size()) < dataView->GetSize()) {
        elementDumpList.resize(dataView->GetSize());
    }

    std::vector<DumpTask> dumpTaskList;
    int count = (dataView->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
    for (int i = 0; i < pool->GetThreadCount(); i++) {
        dumpTaskList.emplace_back(
            &elementDumpList, dataView, count * i, std::min(count * (i + 1), dataView->GetSize()));
    }
    for (size_t i = 0; i < dumpTaskList.size(); i++) {
        pool->SubmitTask(&dumpTaskList[i], DumpTask::Entry);
    }
    pool->NotifyAll();
    pool->WaitForAll();
}

std::string FunctionInterpreter::DumpDataView(const std::shared_ptr<LogicalTensorData> &dataView) {
   DumpDataViewParallel(dataView, execDumpElementList, &operationInterpreter->GetPool());
   return dataView->Dump(&execDumpElementList);
}

std::string FunctionInterpreter::GetDumpFilePath(const std::string &lv0, const std::string &lv1, const std::string &filename) {
    std::string baseDirName = lv0 + "/" + lv1;
    if (!IsPathExist(baseDirName)) {
        CreateDir(baseDirName);
    }
    return baseDirName + "/" + filename;
}

void FunctionInterpreter::DumpBinary(std::vector<int64_t> &shape, std::vector<int64_t> &stride, std::vector<int64_t> &offset, 
        FILE *fdata, uint8_t *data, size_t dtypeSize) {
    if(shape.size() > 1) {
        for (int64_t k = 0; k < shape[0]; k++) {
            auto newShape = std::vector<int64_t>(shape.begin() + 1, shape.end());
            auto newStride = std::vector<int64_t>(stride.begin() + 1, stride.end());
            auto newOffset = std::vector<int64_t>(offset.begin() + 1, offset.end());
            auto newData = data + offset[0] * dtypeSize * stride[0] +  k * stride[0] * dtypeSize;
            DumpBinary(newShape, newStride, newOffset, fdata, newData, dtypeSize);
        }
    } else {
        size_t res = fwrite(data + offset[0] * dtypeSize, dtypeSize, shape[0], fdata);
        if (res != static_cast<size_t>(shape[0])) {
            ALOG_WARN("Write size is not equal actual size.");
        }
    }
}

void FunctionInterpreter::DumpTensorBinary(
        const std::shared_ptr<LogicalTensor> &tensor,
        const std::shared_ptr<LogicalTensorData> &dataView) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;
    std::string dumpTensorDirName = GetDumpFrameDirName();
    std::string dumpTensorFileName = GetDumpTensorFileName(tensor);
    std::string dumpTensorFilePath = GetDumpFilePath(execDumpDir, dumpTensorDirName, dumpTensorFileName);
    dataView->Save(dumpTensorFilePath);
}

void FunctionInterpreter::DumpTensorBinary(
        const std::shared_ptr<LogicalTensorData> &dataView,
        std::string dumpTensorFileName) {
    std::string dumpTensorFilePath = execDumpDir + "/" + dumpTensorFileName;
    int totalSize = dataView->GetSize();
    if (totalSize <= 0) {
        ALOG_WARN("The tensor size is not greater than 0.");
        return;
    }
    auto validShape = dataView->GetValidShape();
    if (std::any_of(validShape.begin(), validShape.end(), [](const int64_t& val) {return val <= 0;})) {
        return;
    }
    auto offset = dataView->GetOffset();
    auto stride = dataView->GetData()->GetStride();
    if (offset.size() != validShape.size() || stride.size() != validShape.size()) {
        return; 
    }
    FILE *fdata = fopen(dumpTensorFilePath.c_str(), "wb");
    DumpBinary(validShape, stride, offset, fdata, dataView->GetData()->data(), BytesOf(dataView->GetDataType()));
    fclose(fdata);
}

std::shared_ptr<LogicalTensorData> FunctionInterpreter::LoadTensorBinary(
        const std::shared_ptr<LogicalTensor> &tensor,
        const std::string filepath) {
    if (!FileExist(filepath)) {
        return nullptr;
    }
    std::vector<int64_t> shape = tensor->GetShape();
    if (std::any_of(shape.begin(), shape.end(), [](int64_t num) {return num <= 0;})) {
        return nullptr;
    }
    FILE *fdata = fopen(filepath.c_str(), "rb");
    auto data = std::make_shared<RawTensorData>(static_cast<DataType>(tensor->Datatype()), shape);
    fread(data->data(), 1, data->size(), fdata);
    auto dataView = std::make_shared<LogicalTensorData>(data, shape, shape, std::vector<int64_t>(shape.size(), 0));
    fclose(fdata);
    return dataView;
}

void FunctionInterpreter::DumpTensorList(const std::string &name, const std::vector<std::shared_ptr<LogicalTensor>> *tensorList,
    const std::vector<std::shared_ptr<LogicalTensorData>> *dataViewList) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;

    std::string dumpTensorDirName = GetDumpFrameDirName();
    std::string dumpTensorFileName = GetDumpTensorListFileName(name);
    int indent = GetFrameSize();
    fprintf(execDumpFile, "<div class=\"detail indent_%d\"><a href=\"%s\">%s</a></div>", indent,
        (dumpTensorDirName + "/" + dumpTensorFileName).c_str(), dumpTensorFileName.c_str());

    std::string dumpTensorFilePath = GetDumpFilePath(execDumpDir, dumpTensorDirName, dumpTensorFileName);
    FILE *dumpTensorFile = fopen(dumpTensorFilePath.c_str(), "w");
    fprintf(dumpTensorFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../../verifier.css">
    </head>
    <body>
)HTML");

    std::vector<std::string> textList(tensorList->size());
    constexpr int ONE_THOUSAND = 1000;
    fprintf(dumpTensorFile, "<table width=\"%dpx\">", static_cast<int>(tensorList->size() * ONE_THOUSAND));
    for (size_t k = 0; k < tensorList->size(); k++) {
        textList[k] = std::to_string(static_cast<int>(k));
        if (tensorList->at(k)) {
            textList[k] += " tileOpFormat:" + std::to_string(tensorList->at(k)->Format());
        }
    }
    DumpLine(dumpTensorFile, textList, "table_head");

    for (size_t k = 0; k < tensorList->size(); k++) {
        if (tensorList->at(k) != nullptr) {
            textList[k] = HtmlEscape(tensorList->at(k)->Dump());
        } else {
            textList[k] = "";
        }
    }
    DumpLine(dumpTensorFile, textList);

    for (size_t k = 0; k < tensorList->size(); k++) {
        textList[k] = HtmlEscape(k < dataViewList->size() ? DumpDataView(dataViewList->at(k)) : "--");
    }
    DumpLine(dumpTensorFile, textList, "tensor_data");
    fprintf(dumpTensorFile, "</table>\n");

    fprintf(execDumpFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(dumpTensorFile);
}

void FunctionInterpreter::FillOperationBasicInfo(Operation *op, FunctionFrame *frame, std::vector<std::string> &opInfo) {
    opInfo[toIndex(OpInfoCsvHeader::rootFuncID)] = std::to_string(frame->rootFuncIndex); 
    opInfo[toIndex(OpInfoCsvHeader::funcID)] = std::to_string(frame->funcIndex);
    opInfo[toIndex(OpInfoCsvHeader::verifyType)] = execDumpFuncKey;
    opInfo[toIndex(OpInfoCsvHeader::loopInfo)] = GetLoopSymbolString();
    opInfo[toIndex(OpInfoCsvHeader::opCode)] = op->GetOpcodeStr();
    opInfo[toIndex(OpInfoCsvHeader::opMagic)] = std::to_string(op->GetOpMagic());
}

void FunctionInterpreter::FillOperationOffsetInfo(Operation *op, FunctionFrame *frame, 
                                                  const std::vector<SymbolicScalar> &linearArgList,
                                                  std::vector<std::string> &opInfo) {
    auto opAttr = std::static_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    if (opAttr) {
        if (copyOpCode.count(op->GetOpcodeStr())) {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
            auto offset = copyAttr->IsCopyOut() ? copyAttr->GetToOffset() : copyAttr->GetFromOffset();
            auto offsetView = operationInterpreter->EvaluateOpImmediate(frame, offset);
            opInfo[toIndex(OpInfoCsvHeader::offset)] = ShapeToString(offsetView);
        } else {
            Offset offsetView = EvaluateOffset(opAttr->GetFromOffset(), opAttr->GetFromDynOffset(), linearArgList);
            opInfo[toIndex(OpInfoCsvHeader::offset)] = ShapeToString(offsetView);
        }
    }
}

void FunctionInterpreter::FillOperationInputInfo(Operation *op, FunctionFrame *frame,
                                                 const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList,
                                                 std::vector<std::string> &opInfo) {
    auto iopSize = op->GetIOperands().size();
    for (size_t k = 0; k < iopSize; k++) {
        if (k < ioperandDataViewList->size()) {
            auto dataView = ioperandDataViewList->at(k);
            if (k > 0) {
                opInfo[toIndex(OpInfoCsvHeader::inputShape)] += ", ";
                opInfo[toIndex(OpInfoCsvHeader::inputValidShape)] += ", ";
                opInfo[toIndex(OpInfoCsvHeader::inputDtype)] += ", ";
                opInfo[toIndex(OpInfoCsvHeader::inputTensors)] += ", ";
            }
            opInfo[toIndex(OpInfoCsvHeader::inputShape)] += ShapeToString(dataView->GetShape());
            opInfo[toIndex(OpInfoCsvHeader::inputValidShape)] += ShapeToString(dataView->GetValidShape());
            opInfo[toIndex(OpInfoCsvHeader::inputDtype)] += DataType2String(dataView->GetDataType());
            auto it = frame->tensorDataBinDict.find(op->GetIOperands()[k]);
            if (it != frame->tensorDataBinDict.end()) {
                opInfo[toIndex(OpInfoCsvHeader::inputTensors)] += it->second;
            }
        }
    }
}

void FunctionInterpreter::FillOperationOutputInfo(Operation *op, FunctionFrame *frame,
                                                  const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
                                                  const std::vector<SymbolicScalar> &linearArgList,
                                                  int indent, std::vector<std::string> &opInfo) {
    auto oopSize = op->GetOOperands().size();
    for (size_t k = 0; k < oopSize; k++) {
        if (k < ooperandDataViewList->size()) {
            auto dataView = ooperandDataViewList->at(k);
            std::string dumpTensorFileName = GetDumpTensorFileName(op->GetOOperands()[k], op, frame);
            DumpTensorBinary(dataView, dumpTensorFileName);
            fprintf(execDumpFile, "<div class=\"detail indent_%d\">%s</a></div>\n", indent, dumpTensorFileName.c_str());
 
            frame->tensorDataBinDict[op->GetOOperands()[k]] = dumpTensorFileName;
            opInfo[toIndex(OpInfoCsvHeader::tensorMagic)] = std::to_string(op->GetOOperands()[k]->GetMagic());
            opInfo[toIndex(OpInfoCsvHeader::rawTensorMagic)] = std::to_string(op->GetOOperands()[k]->GetRawTensor()->GetRawMagic());
            opInfo[toIndex(OpInfoCsvHeader::outputShape)] = ShapeToString(dataView->GetShape());
            opInfo[toIndex(OpInfoCsvHeader::outputValidShape)] = ShapeToString(dataView->GetValidShape());
            opInfo[toIndex(OpInfoCsvHeader::outputDynValidShape)] = ShapeToString(EvaluateValidShape((op->GetOOperands()[k]->GetDynValidShape()), linearArgList));
            opInfo[toIndex(OpInfoCsvHeader::outputDtype)] = DataType2String(dataView->GetDataType());
            opInfo[toIndex(OpInfoCsvHeader::outputTensor)] = dumpTensorFileName;
            opInfo[toIndex(OpInfoCsvHeader::verifyResult)] = "-";
        }
        WriteCsvRow(opInfo);
    }
}

void FunctionInterpreter::DumpOperationTensor(Operation *op, FunctionFrame *frame,
    const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;
 
    int indent = GetFrameSize();
    std::vector<SymbolicScalar> linearArgList;
    if (frame->callopAttr != nullptr) {
        linearArgList = frame->callopAttr->GetLinearArgList();
    }

    std::vector<std::string> opInfo(toIndex(OpInfoCsvHeader::COL_COUNT));
    FillOperationBasicInfo(op, frame, opInfo);
    FillOperationOffsetInfo(op, frame, linearArgList, opInfo);
    FillOperationInputInfo(op, frame, ioperandDataViewList, opInfo);
    FillOperationOutputInfo(op, frame, ooperandDataViewList, linearArgList, indent, opInfo);
}

void FunctionInterpreter::DumpPassTensorDiff(
        const std::shared_ptr<FunctionCaptureExecution> &captureExecution,
        const std::shared_ptr<FunctionCaptureExecution> &captureGolden) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR)
        return;
    if (captureExecution->GetFrameList().size() != captureGolden->GetFrameList().size())
        return;

    std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
    float rtol = static_cast<float>(tolerance[0]);
    float atol = static_cast<float>(tolerance[1]);

    std::string dumpStyleFilePath = execDumpDir + "/entry_" + std::to_string(captureIndex) + ".css";
    execDumpStyleFile = fopen(dumpStyleFilePath.c_str(), "w");
    for (size_t idx = 0; idx < captureGolden->GetFrameList().size(); idx++) {
        auto frameExecution = captureExecution->GetFrameList()[idx];
        auto frameGolden = captureGolden->GetFrameList()[idx];
        const auto &tensorDictExecution = frameExecution->tensorDataBinDict;
        const auto &tensorDictGolden = frameGolden->tensorDataBinDict;
        std::vector<std::shared_ptr<LogicalTensor>> tensorList;
        for (auto &[tensor, dataViewExecution] : tensorDictExecution) {
            (void)dataViewExecution;
            if (tensorDictGolden.count(tensor)) {
                tensorList.push_back(tensor);
            }
        }
        for (size_t k = 0; k < tensorList.size(); k++) {
            auto tensor = tensorList[k];
            ALOG_INFO("Dump tensor diff: ", k, "/", tensorList.size(), " ", tensor->Dump(), " ",
                tensor->GetRawTensor()->Dump());
            auto executionFileName = tensorDictExecution.find(tensor)->second;
            auto dataViewExecution = LoadTensorBinary(tensor, execDumpDir + "/" + executionFileName);
            auto filename = tensorDictGolden.find(tensor)->second;
            auto dataViewGolden = LoadTensorBinary(tensor, dumpPath + "/tensor_graph/" + filename);
            if (!dataViewExecution || !dataViewGolden) {
                continue;
            }
            auto compare = FlowVerifier::VerifyResult(dataViewGolden, dataViewExecution, rtol, atol);
            std::string tensorId = GetDumpTensorId(frameExecution, tensor);
            if (compare.Check()) {
                fprintf(execDumpStyleFile, "#%s { background-color: LightGreen; }\n", tensorId.c_str());
            } else {
                fprintf(execDumpStyleFile, "#%s { background-color: LightCoral; }\n", tensorId.c_str());
            }
        }
    }
    fclose(execDumpStyleFile);
}

void FunctionInterpreter::DumpBegin() {
    frameCount = 0;
    execDumpDir = dumpPath + execDumpFuncKey;
    CreateMultiLevelDir(execDumpDir);

    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;

    std::string styleFilePath = dumpPath + "/verifier.css";
    if (GetFileSize(styleFilePath) == 0) {
        FILE *fcss = fopen(styleFilePath.c_str(), "w");
        for (int i = 0; i < MAX_IDENT_LEVEL; i++) {
            fprintf(fcss, ".indent_%d { margin-left: %dpx; }\n", i, i * 50); // 50 is left margin
        }
        fprintf(fcss, ".table_head { width: 10%%; }\n");
        fprintf(fcss, ".tensor_data { vertical-align: top; }\n");
        fclose(fcss);
    }

    std::string dumpFilePath = execDumpDir + "/entry_" + std::to_string(captureIndex) + ".html";
    execDumpFile = fopen(dumpFilePath.c_str(), "w");
    fprintf(execDumpFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../verifier.css">
    <link rel="stylesheet" type="text/css" href="entry_%s.css">
    </head>
    <body>
)HTML", std::to_string(captureIndex).c_str());
}

void FunctionInterpreter::DumpEnd() {
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;
    fprintf(execDumpFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(execDumpFile);
}

} // namespace npu::tile_fwk
