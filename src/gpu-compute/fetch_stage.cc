/*
 * Copyright (c) 2014-2015 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpu-compute/fetch_stage.hh"

#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{

FetchStage::FetchStage(const ComputeUnitParams &p, ComputeUnit &cu)
    : numVectorALUs(p.num_SIMDs), computeUnit(cu),
      _name(cu.name() + ".FetchStage"), stats(&cu)
{
    for (int j = 0; j < numVectorALUs; ++j) {
        FetchUnit newFetchUnit(p, cu);
        _fetchUnit.push_back(newFetchUnit);
    }
}

FetchStage::~FetchStage()
{
    _fetchUnit.clear();
}

void
FetchStage::init()
{
    for (int j = 0; j < numVectorALUs; ++j) {
        _fetchUnit[j].bindWaveList(&computeUnit.wfList[j]);
        _fetchUnit[j].init();
    }
}

void
FetchStage::exec()
{
    // Reset fetch related stats every cycle
    bool fetched_any = false;

    for (int i = 0; i < computeUnit.numVectorALUs; ++i) {
        _fetchUnit[i].exec();
        // If any fetch unit successfully fetched an instruction
        if (!_fetchUnit[i].) {
            fetched_any = true;
        }
    }

    // If no units fetched anything and there are active wavefronts,
    // this indicates a fetch stall
    if (!fetched_any && computeUnit.activeWaves > 0) {
        computeUnit.trackPipelineStall(ComputeUnit::PipelineStage::Fetch);
    }
}

void
FetchStage::processFetchReturn(PacketPtr pkt)
{
    ComputeUnit::SQCPort::SenderState *sender_state =
        safe_cast<ComputeUnit::SQCPort::SenderState*>(pkt->senderState);

    Wavefront *wavefront = sender_state->wavefront;

    const unsigned num_instructions = pkt->req->getSize() /
        sizeof(TheGpuISA::RawMachInst);

    stats.instFetchInstReturned.sample(num_instructions);
    uint32_t simdId = wavefront->simdId;
    _fetchUnit[simdId].processFetchReturn(pkt);
}

void
FetchStage::fetch(PacketPtr pkt, Wavefront *wavefront)
{
    _fetchUnit[wavefront->simdId].fetch(pkt, wavefront);
}

FetchStage::FetchStageStats::FetchStageStats(statistics::Group *parent)
    : statistics::Group(parent, "FetchStage"),
      ADD_STAT(instFetchInstReturned, "For each instruction fetch request "
               "received record how many instructions you got from it")
{
        instFetchInstReturned.init(1, 32, 1);
}

} // namespace gem5
