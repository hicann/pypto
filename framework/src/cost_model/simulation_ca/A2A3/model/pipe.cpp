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
 * \file pipe.cpp
 * \brief
 */

#include "pipe.h"
namespace CostModel {

using namespace std;

CostModelPipe::CostModelPipe(const std::string &moduleName, const uint32_t coreId, const uint32_t pipeId)
    : pipeName(moduleName), coreId_(coreId), pipeId_(pipeId)
{}

CostModelPipe::~CostModelPipe() {}

bool CostModelPipe::Build() const
{
    return true;
}

void CostModelPipe::Attach(const function<void(const uint32_t, const uint32_t, const uint32_t)>& popSetFlag,
                           const function<void(const uint32_t, const uint32_t, const uint32_t)>& rlsSetFlag,
                           const function<bool(const uint32_t, const uint32_t, const uint32_t)>& getFlag,
                           const std::function<void(const PInstrParam&)>& dump,
                           const std::function<void(const uint32_t)>& retire)
{
    popSetFlag_ = popSetFlag;
    rlsSetFlag_ = rlsSetFlag;
    getFlag_ = getFlag;
    dump_ = dump;
    retire_ = retire;
}

void CostModelPipe::AttachClk(const std::function<uint64_t()>& time)
{
    time_ = time;
}

uint64_t CostModelPipe::GetTime() const
{
    return time_ ? time_() : tick_;
}

bool CostModelPipe::IsqFull() const
{
    return isqOtsd_ >= isqMaxOtsd_;
}

bool CostModelPipe::IsqEmpty() const
{
    return isq_.empty() && module_.empty();
}

void CostModelPipe::Push(const PInstrParam &instr)
{
    if (instr->name == InstrName::SET_FLAG) {
        popSetFlag_(instr->param.at(static_cast<uint32_t>(SetFlagParam::PIPE)),
            instr->param.at(static_cast<uint32_t>(SetFlagParam::TRIGGER_PIPE)),
            instr->param.at(static_cast<uint32_t>(SetFlagParam::EVENT_ID)));
    } else {
        isqOtsd_++;
    }
    isq_.push_back(instr);
}

uint64_t CostModelPipe::GetOpCycle(std::deque<PInstrParam>& program)
{
    if (program.empty()) {
        return 1;
    }
    tick_ = 0;
    while (!program.empty()) {
        if (!module_.empty()) {
            auto instr = module_.front();
            if (instr->endTime <= tick_) {
                module_.pop_front();
            }
        }
        auto instr = program.front();
        if (instr->name == InstrName::WAIT_FLAG ||
            instr->name == InstrName::SET_FLAG ||
            instr->name == InstrName::BAR) {
            if (module_.empty()) {
                program.pop_front();
            }
        } else {
            CalcInstrLatency(instr);
            module_.push_back(instr);
            program.pop_front();
        }
        ++tick_;
    }
    uint64_t endCycle = tick_;
    if (!module_.empty()) {
        endCycle = max(module_.back()->endTime, tick_);
    }
    module_.clear();
    return endCycle;
}

void CostModelPipe::DisPatch()
{
    if (isq_.empty()) {
        return;
    }
    auto instr = isq_.front();
    if (instr->name == InstrName::WAIT_FLAG) {
        auto pipe = instr->param.at(static_cast<uint32_t>(SetFlagParam::PIPE));
        auto triggerPipe = instr->param.at(static_cast<uint32_t>(SetFlagParam::TRIGGER_PIPE));
        auto eventId = instr->param.at(static_cast<uint32_t>(SetFlagParam::EVENT_ID));
        if (getFlag_(pipe, triggerPipe, eventId)) {
            --isqOtsd_;
            isq_.pop_front();
        }
    } else if (instr->name == InstrName::BAR) {
        if (instr->param.at(static_cast<uint32_t>(BarParam::PIPE)) != static_cast<uint32_t>(PipeId::ALL)) {
            if (module_.empty()) {
                --isqOtsd_;
                isq_.pop_front();
            }
        }
    } else {
        CalcInstrLatency(instr);
        module_.push_back(instr);
        if (instr->name != InstrName::SET_FLAG) {
            dump_(instr);
        }
        isq_.pop_front();
    }
}

void CostModelPipe::Release()
{
    if (module_.empty()) {
        return;
    }
    auto instr = module_.front();
    if (instr->endTime > GetTime()) {
        return;
    }
    if (instr->name == InstrName::SET_FLAG) {
        rlsSetFlag_(instr->param.at(static_cast<uint32_t>(SetFlagParam::PIPE)),
            instr->param.at(static_cast<uint32_t>(SetFlagParam::TRIGGER_PIPE)),
            instr->param.at(static_cast<uint32_t>(SetFlagParam::EVENT_ID)));
        module_.pop_front();
    } else {
        --isqOtsd_;
        module_.pop_front();
        retire_(instr->seqNo);
    }
}

void CostModelPipe::CalcInstrLatency(PInstrParam &instr)
{
    auto iter = getLatency_.find(instr->name);
    if (iter != getLatency_.end()) {
        iter->second(this, instr);
    } else {
        instr->popTime = module_.empty() ? GetTime() + 1 : module_.back()->popTime + 1;
        instr->exeTime = module_.empty() ? GetTime() + 1 : module_.back()->exeTime + 1;
        instr->endTime = module_.empty() ? GetTime() + 1 : module_.back()->endTime + 1;
    }
}

void CostModelPipe::CalcNdNzOutL1(PInstrParam &instr)
{
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    uint32_t type = instr->param.at(static_cast<uint32_t>(NdNzParam::TYPE));
    uint32_t srcByte = type == static_cast<uint32_t>(DataType::DT_INT8) ? 1
                     : type == static_cast<uint32_t>(DataType::DT_INT16) ? 2 : 4;
    uint32_t dValue = instr->param.at(static_cast<uint32_t>(NdNzParam::D)) * srcByte;
    bool padding = (dValue % 32) != 0;
    auto burstNum = instr->param.at(static_cast<uint32_t>(NdNzParam::ND_NUM)) *
        instr->param.at(static_cast<uint32_t>(NdNzParam::N));
    auto burstSize = Ceiling(dValue, 32) * 32;
    uint32_t threshold1 = 64;
    uint32_t threshold2 = 256;
    if (dValue >= threshold1 &&
        dValue <= threshold2 &&
        instr->param.at(static_cast<uint32_t>(NdNzParam::D)) == instr->param.at(static_cast<uint32_t>(NdNzParam::SRC_D)) &&
        !padding) {
        auto totalSize = burstNum * burstSize;
        uopNum = Ceiling(totalSize, bL4);
        latency = max(Ceiling(totalSize, ddrRdBandWidth_), burstNum * Ceiling(burstSize, l1WrBandWidth_));
    } else {
        uopNum = burstNum * Ceiling(burstSize, bL4);
        latency = burstNum * max(Ceiling(burstSize, ddrRdBandWidth_), Ceiling(burstSize, l1WrBandWidth_));
    }
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrRdLatency_ + ddrRdLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcMovOutUb(PInstrParam &instr)
{
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    auto burstNum = instr->param.at(static_cast<uint32_t>(MovParam::BURST_NUM));
    auto burstLen = instr->param.at(static_cast<uint32_t>(MovParam::BURST_LEN));
    auto srcGap = instr->param.at(static_cast<uint32_t>(MovParam::SRC_STD));
    auto dstGap = instr->param.at(static_cast<uint32_t>(MovParam::DST_STD));
    auto burstSize = burstLen * BLK_SIZE;
    if (srcGap == 0) {
        auto totalSize = burstNum * burstSize;
        uopNum = Ceiling(totalSize, bL4);
        if (dstGap == 0) {
            latency = max(Ceiling(totalSize, ddrRdBandWidth_), Ceiling(totalSize, ubWrBandWidth_));
        } else {
            latency = max(Ceiling(totalSize, ddrRdBandWidth_), burstNum * Ceiling(burstSize, ubWrBandWidth_));
        }
    } else {
        uopNum = burstNum * Ceiling(burstSize, bL4);
        latency = burstNum * max(Ceiling(burstSize, ddrRdBandWidth_), Ceiling(burstSize, ubWrBandWidth_));
    }
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrRdLatency_ + ddrRdLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcMovUbOut(PInstrParam &instr)
{
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    auto burstNum = instr->param.at(static_cast<uint32_t>(MovParam::BURST_NUM));
    auto burstLen = instr->param.at(static_cast<uint32_t>(MovParam::BURST_LEN));
    auto srcGap = instr->param.at(static_cast<uint32_t>(MovParam::SRC_STD));
    auto dstGap = instr->param.at(static_cast<uint32_t>(MovParam::DST_STD));
    auto burstSize = burstLen * BLK_SIZE;
    if (dstGap == 0) {
        auto totalSize = burstNum * burstSize;
        uopNum = Ceiling(totalSize, bL4);
        if (srcGap == 0) {
            latency = max(Ceiling(totalSize, ddrWrBandWidth_), Ceiling(totalSize, ubRdBandWidth_));
        } else {
            latency = burstNum * max(Ceiling(burstSize, ddrWrBandWidth_), Ceiling(totalSize, ubRdBandWidth_));
        }
    } else {
        uopNum = burstNum * Ceiling(burstSize, bL4);
        latency = burstNum * max(Ceiling(burstSize, ddrWrBandWidth_), Ceiling(burstSize, ubRdBandWidth_));
    }
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrWrLatency_ + ddrWrLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcMovOutUbAlign(PInstrParam &instr)
{
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    auto burstNum = instr->param.at(static_cast<uint32_t>(MovPadParam::BURST_NUM));
    auto burstLen = instr->param.at(static_cast<uint32_t>(MovPadParam::BURST_LEN));
    auto padl = instr->param.at(static_cast<uint32_t>(MovPadParam::LPAD));
    auto padr = instr->param.at(static_cast<uint32_t>(MovPadParam::RPAD));
    auto srcGap = instr->param.at(static_cast<uint32_t>(MovPadParam::SRC_STD));
    auto dstGap = instr->param.at(static_cast<uint32_t>(MovPadParam::DST_STD));
    auto burstSize = burstLen + (padl + padr) * BytesOf(DataType(instr->param.at(static_cast<uint32_t>(MovPadParam::TYPE))));
    if (srcGap == 0) {
        auto totalSize = burstNum * burstSize;
        uopNum = Ceiling(totalSize, bL4);
        if (dstGap == 0) {
            latency = max(Ceiling(totalSize, ddrRdBandWidth_), Ceiling(totalSize, ubWrBandWidth_));
        } else {
            latency = max(Ceiling(totalSize, ddrRdBandWidth_), burstNum * Ceiling(burstSize, ubWrBandWidth_));
        }
    } else {
        uopNum = burstNum * Ceiling(burstSize, bL4);
        latency = burstNum * max(Ceiling(burstSize, ddrRdBandWidth_), Ceiling(burstSize, ubWrBandWidth_));
    }
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrRdLatency_ + ddrRdLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcMovUbOutAlign(PInstrParam &instr)
{
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    auto burstNum = instr->param.at(static_cast<uint32_t>(MovPadParam::BURST_NUM));
    auto burstLen = instr->param.at(static_cast<uint32_t>(MovPadParam::BURST_LEN));
    auto padl = instr->param.at(static_cast<uint32_t>(MovPadParam::LPAD));
    auto padr = instr->param.at(static_cast<uint32_t>(MovPadParam::RPAD));
    auto srcGap = instr->param.at(static_cast<uint32_t>(MovPadParam::SRC_STD));
    auto dstGap = instr->param.at(static_cast<uint32_t>(MovPadParam::DST_STD));
    auto burstSize = burstLen + (padl + padr) * BytesOf(DataType(instr->param.at(static_cast<uint32_t>(MovPadParam::TYPE))));
    if (dstGap == 0) {
        auto totalSize = burstNum * burstSize;
        uopNum = Ceiling(totalSize, bL4);
        if (srcGap == 0) {
            latency = max(Ceiling(totalSize, ddrWrBandWidth_), Ceiling(totalSize, ubRdBandWidth_));
        } else {
            latency = burstNum * max(Ceiling(burstSize, ddrWrBandWidth_), Ceiling(totalSize, ubRdBandWidth_));
        }
    } else {
        uopNum = burstNum * Ceiling(burstSize, bL4);
        latency = burstNum * max(Ceiling(burstSize, ddrWrBandWidth_), Ceiling(burstSize, ubRdBandWidth_));
    }
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrWrLatency_ + ddrWrLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcLoad2d(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(VecOp0Param::REPEAT));
    auto bandwidth = InstrName::LOAD_L1_L0A_2D == instr->name ? l0aWrBandWidth_ : l0bWrBandWidth_;
    uint32_t uopNum = Ceiling(repeat * fracSize, bandwidth);
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime + 1, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint64_t l1DataPath = 16;
    instr->endTime = instr->exeTime + load2dUeLatency_ + l1DataPath;
}

void CostModelPipe::CalcLoad3d(PInstrParam &instr)
{
    DataType type = DataType(instr->param.at(static_cast<uint32_t>(Load3dParam::TYPE)));
    uint32_t kStep = 16;
    uint32_t c0Size = 32;
    uint32_t int4C0Size = 64;
    uint64_t l1DataPath = 16;
    uint64_t colSize = 16;
    if (type == DataType::DT_INT4 || type == DataType::DT_HF4)  {
        kStep = int4C0Size;
    } else {
        kStep = c0Size / BytesOf(type);
    }
    auto repeat = instr->param.at(static_cast<uint32_t>(Load3dParam::M)) *
        instr->param.at(static_cast<uint32_t>(Load3dParam::K)) / kStep / colSize;
    auto bandwidth = InstrName::LOAD_L1_L0A_3D == instr->name ? l0aWrBandWidth_ : l0bWrBandWidth_;
    uint32_t uopNum = Ceiling(repeat * fracSize, bandwidth);
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime + 1, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + load3dUeLatency_ + l1DataPath;
}

void CostModelPipe::CalcFixL0cOut(PInstrParam &instr)
{ 
    uint32_t latency = 1;
    uint32_t uopNum = 1;
    auto mlen = instr->param.at(static_cast<uint32_t>(FixpParam::M));
    auto nlen = instr->param.at(static_cast<uint32_t>(FixpParam::N));
    auto ndNum = (sprNdParam_ & 0xffff) == 0 ? 1 : (sprNdParam_ & 0xffff);
    auto nC0 = Ceiling(nlen, 16);
    auto burstSize = nC0 * 16 * 4;
    uopNum = ndNum * mlen * Ceiling(burstSize, bL4);
    latency = ndNum * mlen * max(Ceiling(burstSize, ddrWrBandWidth_), Ceiling(burstSize, l0cRdBandWidth_));
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + uopNum;
    uint32_t movdelay = latency + ddrWrLatency_ + ddrWrLatencyDiver_;
    instr->endTime = module_.empty() ? instr->exeTime + movdelay : module_.back()->endTime + movdelay;
}

void CostModelPipe::CalcVcadd(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(VecOp1Param::REPEAT));
    auto type = instr->param.at(static_cast<uint32_t>(VecOp1Param::TYPE));
    auto mode = instr->param.at(static_cast<uint32_t>(VecOp1Param::MODE));
    constexpr uint32_t parallel = 256;
    const uint32_t thrput = (mode == 0) ? vecRmwDelay_ : 1;
    const uint32_t delay = (type == static_cast<uint32_t>(DataType::DT_FP16)) ? 23 : 20;
    uint32_t uopNum = Ceiling(repeat * vecRptSize, parallel) * thrput;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    auto endTime = instr->popTime + uopNum + delay + vecDelay_;
    if (!module_.empty() && module_.back()->endTime >= endTime) {
        instr->popTime = module_.back()->endTime - (uopNum + delay + vecDelay_);
    }
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + delay + vecDelay_;
}

void CostModelPipe::CalcVreducev2(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(VecOp1Param::REPEAT));
    auto mode = instr->param.at(static_cast<uint32_t>(VecOp1Param::MODE));
    constexpr uint32_t parallel = 256;
    const uint32_t thrput = (mode == 0) ? vecRmwDelay_ : 1;
    const uint32_t delay = 4;
    uint32_t uopNum = Ceiling(repeat * vecRptSize, parallel) * thrput;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    auto endTime = instr->popTime + uopNum + delay + vecDelay_;
    if (!module_.empty() && module_.back()->endTime >= endTime) {
        instr->popTime = module_.back()->endTime - (uopNum + delay + vecDelay_);
    }
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + delay + vecDelay_;
}
void CostModelPipe::CalcMoveMask(PInstrParam &instr)
{
    uint64_t moveMaskLatency = 8;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + moveMaskLatency;
    instr->endTime = instr->exeTime;
}

void CostModelPipe::CalcVnchwconv(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(VecOp0Param::REPEAT));
    auto type = instr->param.at(static_cast<uint32_t>(VecOp0Param::TYPE));
    const uint32_t parallel = 128;
    const uint32_t thrput = type == static_cast<uint32_t>(DataType::DT_INT8) ? vecRmwDelay_ : 1;
    const uint32_t delay = 7;
    auto uopNum = Ceiling(repeat * vecRptSize, parallel) * thrput;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    auto endTime = instr->popTime + uopNum + delay + vecDelay_;
    if (!module_.empty() && module_.back()->endTime >= endTime) {
        instr->popTime = module_.back()->endTime - (uopNum + delay + vecDelay_);
    }
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + delay + vecDelay_;
}

void CostModelPipe::CalcMoveVa(PInstrParam &instr)
{
    uint64_t moveALatency = 2;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = instr->popTime + 1;
    instr->endTime = instr->exeTime + moveALatency;
}

void CostModelPipe::CalcVecOp(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(VecOp0Param::REPEAT));
    auto type = instr->param.at(static_cast<uint32_t>(VecOp0Param::TYPE));
    auto iter = VEC_VALU_TAB.find(type << 24 | static_cast<uint32_t>(instr->name));
    if (iter != VEC_VALU_TAB.end()) {
        const uint32_t parallel = iter->second.at(0);
        const uint32_t thrput = (instr->name == InstrName::VCMAX || instr->name == InstrName::VCMIN) ?
                                   vecRmwDelay_ : iter->second.at(1);
        const uint32_t delay = iter->second.at(2);
        auto uopNum = Ceiling(repeat * vecRptSize, parallel) * thrput;
        instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
        auto endTime = instr->popTime + uopNum + delay + vecDelay_;
        if (!module_.empty() && module_.back()->endTime >= endTime) {
            instr->popTime = module_.back()->endTime - (uopNum + delay + vecDelay_);
        }
        instr->exeTime = instr->popTime + uopNum;
        instr->endTime = instr->exeTime + delay + vecDelay_;
    }
}

void CostModelPipe::CalcVconv(PInstrParam &instr)
{
    auto repeat = instr->param.at(static_cast<uint32_t>(ConvParam::REPEAT));
    constexpr uint32_t parallel = 256;
    constexpr uint32_t thrput = 1;
    constexpr uint32_t delay = 4;
    uint32_t uopNum = Ceiling(repeat * vecRptSize, parallel) * thrput;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    auto endTime = instr->popTime + uopNum + delay + vecDelay_;
    if (!module_.empty() && module_.back()->endTime >= endTime) {
        instr->popTime = module_.back()->endTime - (uopNum + delay + vecDelay_);
    }
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + delay + vecDelay_;
}

void CostModelPipe::CalcMovUbUb(PInstrParam &instr)
{
    uint32_t uopNum = 1;
    uint64_t ubDatapassLatency = 12;
    auto burstNum = instr->param.at(static_cast<uint32_t>(MovParam::BURST_NUM));
    auto burstLen = instr->param.at(static_cast<uint32_t>(MovParam::BURST_LEN));
    auto burstSize = burstLen * BLK_SIZE;
    uopNum = burstNum * Ceiling(burstSize, vecRptSize);
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    auto endTime = instr->popTime + uopNum + ubDatapassLatency;
    if (!module_.empty() && module_.back()->endTime >= endTime) {
        instr->popTime = module_.back()->endTime - (uopNum + ubDatapassLatency);
    }
    instr->exeTime = instr->popTime + uopNum;
    instr->endTime = instr->exeTime + ubDatapassLatency;
}

void CostModelPipe::CalcMmad(PInstrParam &instr)
{
    auto mlen = instr->param.at(static_cast<uint32_t>(MmadParam::M));
    auto klen = instr->param.at(static_cast<uint32_t>(MmadParam::K));
    auto nlen = instr->param.at(static_cast<uint32_t>(MmadParam::N));
    auto ctype = instr->param.at(static_cast<uint32_t>(MmadParam::CTYPE));
    auto atype = instr->param.at(static_cast<uint32_t>(MmadParam::ATYPE));
    auto btype = instr->param.at(static_cast<uint32_t>(MmadParam::BTYPE));
    auto type = (ctype << 16) | (atype << 8) | (btype);
    if (CUBE_PE_TAB.find(type) == CUBE_PE_TAB.end()) {
        return;
    }
    auto kstep = CUBE_PE_TAB.at(type);
    uint64_t colSize = 16;
    instr->popTime = module_.empty() ? GetTime() : max(module_.back()->exeTime, GetTime());
    instr->exeTime = GetTime() + Ceiling(mlen, colSize) * Ceiling(nlen, colSize) * Ceiling(klen, kstep);
    instr->endTime = instr->exeTime + cubeStateLatency_;
}

void CostModelPipe::CalcLd(PInstrParam &instr)
{
    uint64_t ldLatency = 6;
    instr->popTime = module_.empty() ? GetTime() : module_.back()->endTime;
    instr->exeTime = instr->popTime + ldLatency;
    instr->endTime = instr->exeTime;
}

void CostModelPipe::CalcSt(PInstrParam &instr)
{
    uint64_t sdLatency = 4;
    instr->popTime = module_.empty() ? GetTime() : module_.back()->endTime;
    instr->exeTime = instr->popTime + sdLatency;
    instr->endTime = instr->exeTime;
}

void CostModelPipe::CalcAlu(PInstrParam &instr)
{
    int index = 2;
    uint64_t shift = 32;
    uint64_t aluLatency = 2;
    if (!instr->param.empty() && instr->param.at(0) == static_cast<uint32_t>(SprId::NDPARA)) {
        sprNdParam_ = static_cast<uint64_t>(instr->param.at(index)) << shift | instr->param.at(1);
    }
    instr->popTime = module_.empty() ? GetTime() : module_.back()->endTime;
    instr->exeTime = instr->popTime + aluLatency;
    instr->endTime = instr->exeTime;
}

int GetTileOpCycle(const char* input) {
    std::string inputStr(input);
    auto program = SplitString(inputStr, '\n');
    auto prog = GetProgram(program);
    auto pipe = std::make_shared<CostModelPipe>("cost", 0, 0);
    uint64_t ret = pipe->GetOpCycle(prog);
    return static_cast<int>(ret);
}

std::unordered_map<InstrName, std::function<void(CostModelPipe*, PInstrParam&)>> CostModelPipe::getLatency_ = {
    { InstrName::NDNZ_OUT_L1, [](CostModelPipe* self, PInstrParam &instr) { self->CalcNdNzOutL1(instr); } }
,   { InstrName::MOV_OUT_UB, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMovOutUb(instr); } }
,   { InstrName::MOV_UB_OUT, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMovUbOut(instr); } }
,   { InstrName::MOV_OUT_UB_PAD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMovOutUbAlign(instr); } }
,   { InstrName::MOV_UB_OUT_PAD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMovUbOutAlign(instr); } }
,   { InstrName::LOAD_L1_L0A_2D, [](CostModelPipe* self, PInstrParam &instr) { self->CalcLoad2d(instr); } }
,   { InstrName::LOAD_L1_L0B_2D, [](CostModelPipe* self, PInstrParam &instr) { self->CalcLoad2d(instr); } }
,   { InstrName::LOAD_L1_L0A_3D, [](CostModelPipe* self, PInstrParam &instr) { self->CalcLoad3d(instr); } }
,   { InstrName::LOAD_L1_L0B_3D, [](CostModelPipe* self, PInstrParam &instr) { self->CalcLoad3d(instr); } }
,   { InstrName::FIX_L0C_OUT, [](CostModelPipe* self, PInstrParam &instr) { self->CalcFixL0cOut(instr); } }
,   { InstrName::VADD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VSEL, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VBITSORT, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMRGSORT4, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VSUB, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMUL, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VDIV, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMIN, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCOPY, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VEXP, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VLN, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VSQRT, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VRSQRT, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMAX, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCADD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVcadd(instr); } }
,   { InstrName::MOVEV, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::MOVEMASK, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMoveMask(instr); } }
,   { InstrName::VCMAX, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCMIN, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VADDS, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMULS, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMINS, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VMAXS, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCGADD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCGMAX, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCGMIN, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VREC, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::VCONV, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVconv(instr); } }
,   { InstrName::MOV_UB_UB, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMovUbUb(instr); } }
,   { InstrName::MOVEVA, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMoveVa(instr); } }
,   { InstrName::VNCHWCONV, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVnchwconv(instr); } }
,   { InstrName::VREDUCEV2, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVreducev2(instr); } }
,   { InstrName::VBRCB, [](CostModelPipe* self, PInstrParam &instr) { self->CalcVecOp(instr); } }
,   { InstrName::MMAD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcMmad(instr); } }
,   { InstrName::LD, [](CostModelPipe* self, PInstrParam &instr) { self->CalcLd(instr); } }
,   { InstrName::ST, [](CostModelPipe* self, PInstrParam &instr) { self->CalcSt(instr); } }
,   { InstrName::ALU, [](CostModelPipe* self, PInstrParam &instr) { self->CalcAlu(instr); } }
};

}  // namespace CostModel