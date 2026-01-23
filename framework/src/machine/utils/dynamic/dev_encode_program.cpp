/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_encode_program.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_program.h"

namespace npu::tile_fwk::dynamic {
namespace {
const size_t WIDTH = 16;
const int ADDRESS_MIN_WIDTH = 6;
}
void DevAscendProgram::DumpCce(std::ostringstream& oss, int indent) const {
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << INDENTINNER << "#cce:" << cceCodeList.size() << "\n";
    for (size_t i = 1; i < cceCodeList.size(); i++) {
        const DevCceBinary &cceCode = At(cceCodeList, i);
        oss << INDENTINNER << "#cce-" << i << " #CoreType:" << cceCode.coreType
            << " #FuncHash:" << cceCode.funcHash;
        oss << "\n";
    }
}

void DevAscendProgram::DumpControlFlow(const int indent, const bool dumpAddr, std::ostringstream& oss) const {
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << "====\n"; // Dump control flow code (begin)

    oss << INDENTINNER << "#HostControlCodeSize:" << hostControlFlowBinary.size();
    if (dumpAddr) {
        oss << " #HostControlCodeAddr:" <<
            AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(hostControlFlowBinary, 0)));
    }
    oss << "\n";

    for (size_t i = 0; i < hostControlFlowBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, hostControlFlowBinary.size()); off++) {
            oss << " " << DumpByte(At(hostControlFlowBinary, off));
        }
        oss << "\n";
    }

    oss << "====\n"; // Dump control flow code: ^^^ Host / Dev vvv

    oss << INDENTINNER << "#DevControlCodeSize:" << devControlFlowBinary.size();
    if (dumpAddr) {
        oss << " #DevControlCodeAddr:" <<
            AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(devControlFlowBinary, 0)));
    }
    oss << "\n";

    for (size_t i = 0; i < devControlFlowBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, devControlFlowBinary.size()); off++) {
            oss << " " << DumpByte(At(devControlFlowBinary, off));
        }
        oss << "\n";
    }

    oss << "====\n"; // Dump control flow code (ends)
}

void DevAscendProgram::DumpExpressionTable(const int indent, const bool dumpAddr, std::ostringstream& oss) const {
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << INDENTINNER << "#ExprCount:" << expressionTableSize << "\n";

    oss << INDENTINNER << "#ExprCodeSize:" << expressionTableBinary.size();
    if (dumpAddr) {
        if (expressionTableBinary.size() != 0) {
            oss << " #ExprCodeAddr:" << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(expressionTableBinary, 0)));
        }
    }
    oss << "\n";

    for (size_t i = 0; i < expressionTableBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, expressionTableBinary.size()); off++) {
            oss << " " << DumpByte(At(expressionTableBinary, off));
        }
        oss << "\n";
    }

    oss << INDENTINNER << "#func:" << devEncodeList.size() << "\n";
    for (size_t i = 0; i < devEncodeList.size(); i++) {
        const DevAscendFunction *func = reinterpret_cast<const DevAscendFunction *>(&At(At(devEncodeList, i), 0));
        oss << func->Dump(IDENT_SIZE) << "\n";
    }
}

std::string DevAscendProgram::Dump(const int indent, const bool dumpAddr) const {
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::ostringstream oss;
    oss << "DevProgram {\n";
    oss << INDENTINNER << "#tensorMemBudget:" << memBudget.tensor.Total() << "\n";
    oss << INDENTINNER << "#metadataMemBudget:" << memBudget.metadata.Total() << "\n";
    oss << INDENTINNER << "#deviceSchMode:" << devArgs.machineConfig << "\n";
    oss << INDENTINNER << "#stitchFunctionNumInitial:" << stitchFunctionNumInitial << "\n";
    oss << INDENTINNER << "#stitchFunctionNumStep:" << stitchFunctionNumStep << "\n";
    oss << INDENTINNER << "#stitchFunctionsize:" << stitchFunctionsize << "\n";
    oss << INDENTINNER << "#slot{" << slotSize << "}\n";
    oss << INDENTINNER << "#assembleSlot{" << assembleSlotSize << "}\n";
    oss << INDENTINNER << "#symbolCount:" << symbolTable.size() << "\n";
    for (size_t i = 0; i < symbolTable.size(); i++) {
        const DevAscendProgramSymbol &symbol = At(symbolTable, i);
        oss << INDENTINNER << "#symbol:" << symbol.index << " = " << &At(symbol.name, 0) << "\n";
    }
    oss << INDENTINNER << "#inputCount:" << startArgsInputTensorSlotIndexList.size() << "\n";
    for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#input:" << i << " -> #slot:" << At(startArgsInputTensorSlotIndexList, i) << "\n";
    }
    oss << INDENTINNER << "#outputCount:" << startArgsOutputTensorSlotIndexList.size() << "\n";
    for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#output:" << i << " <- #slot:" << At(startArgsOutputTensorSlotIndexList, i) << "\n";
    }
    oss << INDENTINNER << "#assembleSlotCount:" << assembleSlotIndexList.size() << "\n";
    for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#assembleSlot:" << i << " -> #slot:" << At(assembleSlotIndexList, i) << "\n";
    }
    oss << INDENTINNER << "#outputInplaceSlotCount:" << outputInplaceSlotList.size() << "\n";
    for (size_t i = 0; i < outputInplaceSlotList.size(); i++) {
        oss << INDENTINNER << "#outputInplaceSlot:" << i << " -> #slot:" << At(outputInplaceSlotList, i) << "\n";
    }
    for (size_t i = 0; i < partialUpdateList.size(); i++) {
        auto &partialUpdate = At(partialUpdateList, i);
        oss << INDENTINNER << "#slot-partial-update-" << i << ":" << !partialUpdate.Empty();
        if (!partialUpdate.Empty()) {
            oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(partialUpdate.cellMatchTableDesc)
                << " | #cellMatchStaticTable:" << partialUpdate.cellMatchRuntimePartialUpdateTable.size();
        }
        oss << "\n";
    }
    for (size_t i = 0; i < startArgsInputSymbolIndexList.size(); i++) {
        oss << INDENTINNER << "#symbol:" << i << " -> #symbolTable:" << At(startArgsInputSymbolIndexList, i) << "\n";
    }

    DumpExpressionTable(indent, dumpAddr, oss);

    DumpControlFlow(indent, dumpAddr, oss);

    DumpCce(oss, indent);
    oss << "}";
    return oss.str();
}

void DevAscendProgram::DumpFile(const std::string &filePath) const {
    std::ofstream ofs(filePath);
    ofs << Dump();
    ofs.close();
}
}
