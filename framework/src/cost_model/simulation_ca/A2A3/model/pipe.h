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
 * \file pipe.h
 * \brief
 */

#ifndef __COST_MODEL_PIPE__
#define __COST_MODEL_PIPE__

#include <unordered_map>
#include "def.h"
#include "cost_model/simulation/common/CommonType.h"
namespace CostModel {
class CostModelPipe {
public:
    CostModelPipe(const std::string &moduleName, const uint32_t coreId, const uint32_t pipeId);
    virtual ~CostModelPipe();
    bool Build() const;
    void Attach(const std::function<void(const uint32_t, const uint32_t, const uint32_t)>& popSetFlag
                                    /* pipeId, triggerPipeId, eventId */,
                const std::function<void(const uint32_t, const uint32_t, const uint32_t)>& rlsSetFlag
                                    /* pipeId, triggerPipeId, eventId */,
                const std::function<bool(const uint32_t, const uint32_t, const uint32_t)>& getFlag
                                    /* pipeId, triggerPipeId, eventId */,
                const std::function<void(const PInstrParam&)>& dump,
                const std::function<void(const uint32_t)>& retire);
    bool IsqFull() const;
    bool IsqEmpty() const;
    void Push(const PInstrParam& instr);
    inline std::string Name() const
    {
        return pipeName;
    }
    void DisPatch();
    void Release();

    void AttachClk(const std::function<uint64_t()>& time);
    uint64_t GetOpCycle(std::deque<PInstrParam>& program);
    inline void SetReadGmFactor(const uint32_t factor)
    {
        ddrRdLatency_ = factor;
    }
    inline uint32_t GetReadGmFactor() const
    {
        return ddrRdLatency_;
    }
    inline void SetWriteGmFactor(const uint32_t factor)
    {
        ddrWrLatency_ = factor;
    }
    inline uint32_t GetWriteGmFactor() const
    {
        return ddrWrLatency_;
    }
private:
    void CalcInstrLatency(PInstrParam& instr);
    void CalcNdNzOutL1(PInstrParam &instr);
    void CalcMovOutUb(PInstrParam &instr);
    void CalcMovUbOut(PInstrParam &instr);
    void CalcMovOutUbAlign(PInstrParam &instr);
    void CalcMovUbOutAlign(PInstrParam &instr);
    void CalcLoad2d(PInstrParam &instr);
    void CalcLoad3d(PInstrParam &instr);
    void CalcFixL0cOut(PInstrParam &instr);
    void CalcVcadd(PInstrParam &instr);
    void CalcMoveMask(PInstrParam &instr);
    void CalcVecOp(PInstrParam &instr);
    void CalcVconv(PInstrParam &instr);
    void CalcMovUbUb(PInstrParam &instr);
    void CalcMmad(PInstrParam &instr);
    void CalcVnchwconv(PInstrParam &instr);
    void CalcVreducev2(PInstrParam &instr);
    void CalcMoveVa(PInstrParam &instr);
    void CalcLd(PInstrParam &instr);
    void CalcSt(PInstrParam &instr);
    void CalcAlu(PInstrParam &instr);
    uint64_t GetTime() const;
private:
    const std::string pipeName;
    const uint32_t coreId_ = 0;
    const uint32_t pipeId_;
    std::deque<PInstrParam> isq_;
    std::deque<PInstrParam> module_;
    uint32_t isqOtsd_ = 0;
    const uint32_t isqMaxOtsd_ = 32;
    uint32_t ddrRdLatency_ = 347;
    const uint32_t ddrRdLatencyDiver_ = 4;
    uint32_t ddrWrLatency_ = 203;
    const uint32_t ddrWrLatencyDiver_ = 4;
    const uint32_t ddrRdBandWidth_ = 87;
    const uint32_t ddrWrBandWidth_ = 87;
    const uint32_t l0aWrBandWidth_ = 256;
    const uint32_t l0bWrBandWidth_ = 128;
    const uint32_t l1WrBandWidth_ = 256;
    const uint32_t l0cRdBandWidth_ = 128;
    const uint32_t ubRdBandWidth_ = 128;
    const uint32_t ubWrBandWidth_ = 128;
    const uint32_t dmaUeLatency_ = 9;
    const uint32_t load2dUeLatency_ = 9;
    const uint32_t load3dUeLatency_ = 27;
    const uint32_t cubeStateLatency_ = 20;
    const uint32_t vecDelay_ = 15;
    const uint32_t vecRmwDelay_ = 7;
    const uint32_t fracSize = 512;
    const uint32_t vecRptSize = 256;
    const uint32_t bL4 = 512;
    const uint32_t bL2 = 256;
    const uint32_t bL1 = 128;
    std::function<void(const uint32_t, const uint32_t, const uint32_t)> popSetFlag_ = nullptr;
    std::function<void(const uint32_t, const uint32_t, const uint32_t)> rlsSetFlag_ = nullptr;
    std::function<bool(const uint32_t, const uint32_t, const uint32_t)> getFlag_ = nullptr;
    std::function<void(const PInstrParam&)> dump_ = nullptr;
    std::function<uint64_t()> time_ = nullptr;
    std::function<void(const uint32_t)> retire_ = nullptr;
    static std::unordered_map<InstrName, std::function<void(CostModelPipe*, PInstrParam&)>> getLatency_;
    uint64_t tick_ = 0;
    uint64_t sprNdParam_ = 0;
};

extern "C" {
    int GetTileOpCycle(const char* input);
}
}  // namespace CostModel

#endif