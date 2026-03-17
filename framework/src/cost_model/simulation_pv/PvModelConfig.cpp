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
 * \file PvModelConfig.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include "PvModelConfig.h"
#include "tilefwk/pypto_fwk_log.h"
#include "cost_model/simulation/utils/simulation_error.h"

namespace CostModel {
void PvModelSystemA2A3Config::Dump(std::string path) {
    std::ofstream outFile(path);

    // 检查文件是否成功打开
    if (!outFile.is_open()) {
        return;
    }

    std::string config = R"!!!(
title = "PV Config"
[Project]
    name = "A2A3"

[LOG]
    disable_list            = [  ]
    enable_list             = [  ]
    # trace: 0, debug: 1, info: 2, warn: 3, error: 4, critical: 5, off: 6
    file_print_level        = 1
    screen_print_level      = 3
    flush_level             = 1
    rotating_file_size      = 134217728     # 0x8000000     # ~130MB
    rotating_file_number    = 2
    core_enable_mask        = ["0xffffffff"]

[ARCH]
    cube_core_num = 1
    vec_core_num = 2
    inorder_acc = 1
    wait_flag_dev_en = 0
    max_sim_time = 30000000

[BIU]
    atomic_switch = 1
    sc_mte_pcie_win_size = 18
    sc_mte_pcie_win_base = 0

[UB_Buffer]
    total_size = 196608
    wrap_en = 1

[L0A_Buf]
    total_size = 65536
    wrap_en = 0

[L0B_Buf]
    total_size = 65536
    wrap_en = 0

[L0A_Wino_Buf]
    total_size = 65536
    wrap_en = 0

[L0B_Wino_Buf]
    total_size = 65536
    wrap_en = 0

[L0C_Buf]
    total_size = 131072
    wrap_en = 1

[L1_Buf]
    total_size = 524288
    wrap_en = 0

[FIX_Buf]
    total_size = 6144
    wrap_en = 0

[BT_Buf]
    total_size = 1024
    wrap_en = 0

[SCALAR_BUF]
    total_size              = 16384
    wrap_en                 = 0
    start_address           = 262144
    sys_va_base_config      = 1           # 0: config by model spr    1: config by spec
    sys_va_base             = 0           # sys va base adress
    stack_phy_base_config   = 1           # 0: config by model spr    1: config by spec
    stack_phy_base          = 34603008    # (stack_va_base_addr[48:25] == sys_va_addr[48:25]) == > stack in ub  0x2100000

[SMASK]
    total_size              = 256
    wrap_en                 = 0
)!!!";

    outFile << config;

    outFile.close();
    return;
}

void PvModelSystemA5Config::Dump(std::string path) {
    std::ofstream outFile(path);

    // 检查文件是否成功打开
    if (!outFile.is_open()) {
        return;
    }

    std::string config = R"!!!(
title = "PV Config"

[Project]
    name                    = "A5"
    version_name            = 3
    release_tag             = ""
    release_info            = ""

[LOG]
    disable_list            = [  ]
    enable_list             = [  ]
    core_enable_mask        = ["0xffffffff"]
    # trace = 0, debug = 1, info = 2, warn = 3, error = 4, critical = 5, off = 6
    file_print_level        = 2
    screen_print_level      = 3
    flush_level             = 2
    rotating_file_size      = 134217728     # 0x8000000     # ~130MB
    rotating_file_number    = 20
    separate_vf_log         = false
    perf_event_log          = false
    perf_rveccores          = ["veccore0", "veccore1"]
    sampling_period         = 32
    # Hook signal handlers
    # (i) print_bt - print bracktrace
    # (ii) flush - flush log
    # (iii) exit_mode - disabled = 0, exit simulation = 1, deadloop = 2
    sighook_fd              = 0             # Print message to 0:STDOUT; 1:STDERR
    sigint_print_bt         = false
    sigint_flush            = true
    sigint_exit_mode        = 1
    sigfpe_print_bt         = true
    sigfpe_flush            = true
    sigfpe_exit_mode        = 1
    sigabrt_print_bt        = false
    sigabrt_flush           = false
    sigabrt_exit_mode       = 0

[STAT]
    disable_list            = []
    enable_list             = []
    core_enable_mask        = ["0xffffffff"]
    path                    = "./"
    window_size             = 10
    summary_log_enable      = true
    aicore_trace_enable     = true
    aicore_trace_core0_only = false     # enable core0 trace only to save disk space
    ld_est_enable           = 0
    # ld_est_param: [w_cube, w_vec0, w_vec1, f1, f2, f3, f4, f5, th_up, th_low]
    ld_est_param            = [0.6, 0.2, 0.0, 0.1, 0.2, 0.3, 0.1, 0.3, 0.4, 0.2]
    print_level             = 2
    flush_window            = 100

[ARCH]
    cube_core_num       = 1
    vec_core_num        = 2                       # cube:vector = 1:1
    core_ostd_num       = 1                       # 2 early end  1 normal mode
    vector_core_mode    = 1                       # 0 1981       1 1911
    inorder_acc         = 0
    mem_reset_val       = 0                       # Model bus only
    wait_flag_dev_en    = 0
    func_switch         = 1
    max_sim_time        = 30000000                # max sim time, ST env only
    sim_end_notify_time = 100                     # notify delay for sim end event, ST env only
    exception_as_error  = 0                       # treat exception as error, ST env only
    ub_overlap_chk      = 1
    separate_arch       = 1
    simt_flag_en        = 1
    sub_core            = ["cubecore0", "veccore0", "veccore1"]
    coupling_core       = []
    exception_break_en  = 0                       # 0 just report 1 report and break
    exception_log_ctrl  = 0                       # 0 follows normal log config; 1 file and screen log; 2 file log only
    su_rdPort_option    = 1                       # 0 rd separately 1 rd merge
    wait_intra_block_en = 0                       # 1 enable; 0 disable
    icache_flush_en     = false                   # flush icache at block start

[WRAPPER]
    # model wrapper config
    aic_wrap_log_file_level = 2                   # aic wrapper log (file)
    adapter_log_file_level  = 6                   # adapter log (file)
    cosim_log_scr_level     = 3                   # k3 cosim wrapper log (screen)
    cosim_log_file_level    = 2                   # k3 cosim wrapper log (file)
    cosim_log_flush_level   = 3                   # k3 cosim wrapper log (flush)

[PI_CFG]
    pi_cfg_enable                   = 0
    pi_cfg_force_mix                = 0     # 0:no-force; 1:force mix

[PI_CFG_DISABLED]
    # arch
    cube_vec_pi_ivr                 = false
    # cube
    cube_ctrl_base_config           = 0          # 0: config by model spr; 1: config by spec toml
    CUBE_CTRL_0                     = 1743011328 # defult: 1743011328
    # CUBE_CTRL_1 defult: 11162892, can use CUBE_CTRL_1[24:31] with CUBE_CTRL_0[5:8] when dummy cycle > 64
    CUBE_CTRL_1                     = 11162892
    cube_global_sync_en             = false
    cube_global_sync_phase_offset   = 20
    cube_global_sync_period         = 15
    cube_global_sync_interval       = 10
    cube_resonance_en               = false
    cube_resonance_cnt              = 100
    cube_resonance_delt             = 50
    cube_pi_ivr_mmad_dt             = 20
    # vector
    rvec_global_sync_en             = false
    rvec_global_sync_phase_offset_0 = 40         # offset for all vector subcore at aicore, times core_id
    rvec_global_sync_phase_offset_1 = 20         # offset among vector subcore, times sub_core_id
    rvec_global_sync_period         = 10
    rvec_global_sync_window         = 1
    rvec_global_sync_id_offset_0    = 1          # core_id
    rvec_global_sync_id_offset_1    = 1          # sub_core_id
    rvec_resonance_en               = false
    rvec_resonance_period           = 100
    rvec_resonance_delta            = 50
    rvec_pi_ivr_vf_dt               = 20

[PI_CFG_NORMAL]
    # arch
    cube_vec_pi_ivr                 = true
    # cube
    cube_ctrl_base_config           = 1          # 0: config by model spr; 1: config by spec toml
    CUBE_CTRL_0                     = 1743011440 # cycle = 67 interval = 21
    # CUBE_CTRL_1 defult: 11162892, can use CUBE_CTRL_1[24:31] with CUBE_CTRL_0[5:8] when dummy cycle > 64
    CUBE_CTRL_1                     = 78270548   # cycle = 67 interval = 21
    cube_global_sync_en             = true
    cube_global_sync_phase_offset   = 1
    cube_global_sync_period         = 16
    cube_global_sync_interval       = 10
    cube_resonance_en               = false
    cube_resonance_cnt              = 100
    cube_resonance_delt             = 50
    cube_pi_ivr_mmad_dt             = 0
    # vector
    rvec_global_sync_en             = false
    rvec_global_sync_phase_offset_0 = 1          # offset for all vector subcore at aicore, times core_id
    rvec_global_sync_phase_offset_1 = 2          # offset among vector subcore, times sub_core_id
    rvec_global_sync_period         = 16
    rvec_global_sync_window         = 1
    rvec_global_sync_id_offset_0    = 1          # core_id
    rvec_global_sync_id_offset_1    = 1          # sub_core_id
    rvec_resonance_en               = false
    rvec_resonance_period           = 100
    rvec_resonance_delta            = 50
    rvec_pi_ivr_vf_dt               = 16

[PI_CFG_MIX]
    # arch
    cube_vec_pi_ivr                 = true
    # cube
    cube_ctrl_base_config           = 1          # 0: config by model spr; 1: config by spec toml
    CUBE_CTRL_0                     = 1743011664 # cycle = 42 interval = 21
    # CUBE_CTRL_1 defult: 11162892, can use CUBE_CTRL_1[24:31] with CUBE_CTRL_0[5:8] when dummy cycle > 64
    CUBE_CTRL_1                     = 44716116   # cycle = 42 interval = 21
    cube_global_sync_en             = false
    cube_global_sync_phase_offset   = 20
    cube_global_sync_period         = 15
    cube_global_sync_interval       = 10
    cube_resonance_en               = false
    cube_resonance_cnt              = 100
    cube_resonance_delt             = 50
    cube_pi_ivr_mmad_dt             = 42
    # vector
    rvec_global_sync_en             = false
    rvec_global_sync_phase_offset_0 = 40         # offset for all vector subcore at aicore, times core_id
    rvec_global_sync_phase_offset_1 = 20         # offset among vector subcore, times sub_core_id
    rvec_global_sync_period         = 10
    rvec_global_sync_window         = 1
    rvec_global_sync_id_offset_0    = 1          # core_id
    rvec_global_sync_id_offset_1    = 1          # sub_core_id
    rvec_resonance_en               = false
    rvec_resonance_period           = 100
    rvec_resonance_delta            = 50
    rvec_pi_ivr_vf_dt               = 16

[DDR]
    addr_begin              = 67108864      # 0x4000000
    min_read_latency        = 330
    read_latency_diver      = 180
    min_write_latency       = 330
    write_latency_diver     = 180
    min_dbid_latency        = 20
    dbid_latency_diver      = 20
    bandwidth_limit         = 30
    max_credit_num          = 4096

[L2]
    addr_begin              = 0             # 0x0
    size                    = 67108864      # 0x4000000
    read_bandwidth_limit    = 100
    write_bandwidth_limit   = 100
    max_credit_num          = 4096
    min_read_latency        = 216
    read_latency_diver      = 4
    min_write_latency       = 238
    write_latency_diver     = 4
    min_dbid_latency        = 20
    dbid_latency_diver      = 20
    l2c_op                  = 0             # 0(no l2c), 1(random l2c), 2(true l2c, not support now)
    rand_l2c_hitrate        = 60            # 0~100

[BIU]
    icache_port_num          = 3
    dcache_port_num          = 3
    simt_dcache_port_num     = 2
    cube_core_num            = 1
    mte_port_num             = 3                    # equal to cube_core_num + vec_core_num
    brif_wrr_weight          = [2, 1, 1]            # read req arbiter for qiling version:size equal to cube_core_num + vec_core_num
    vdcache_mte_wrr_weight   = [1, 1, 2, 1, 1]      # read req arbiter for ict version: vdcache, aic_mte2, aiv_mte2
    bwif_wrr_weight          = [2, 1, 1]            # size equal to cube_core_num + vec_core_num
    rob_depth                = 128
    tag_bandwith             = 256                  # read ostd = rob_depth * 512 / tag_bandwith;
    cmd_buf_depth            = 128
    store_buf_depth          = 64                   # write ostd = store_buf_depth * 512 / tag_bandwith;
    intlv_granularity        = 0                    # 0:disable 1:128B 2:256B 3:512B 4:1024B
    scramble_granularity     = 0                    # 0:disable 1:128B 2:256B 3:512B 4:1024B
    biu_port_width           = 128
    bus_read_port_num        = 2
    bus_write_port_num       = 1
    atomic_switch            = 1
    standard_axi             = 0                    # 0: ict, 1: qiling
    su_read_latency          = 1
    bwif_wr_port_option      = 1                    # 0 write separately 1 wr merge
    brif_suRdPort_option     = 1                    # same with su_rdPort_option
    # ostd limit
    rd_ostd_water_line       = 256
    rd_ostd_water_line_mode  = 0                    # 0 : all can send, 1: top 3 send, 2: top 2 send
    mte_rd_otsd_waterline    = [256,  256,  256,  256]
    wr_ostd_water_line       = 128
    wr_ostd_water_line_mode  = 0                    # 0 : all can send, 1: top 2 send, 2: top 1 send
    mte_wr_otsd_waterline    = [128,  128,  128,  128]
    # bandwith limte
    cbusy_rd_ostd_water_line = [256, 192, 144, 128] # for SOC read reverse otsd limit
    cbusy_wr_ostd_water_line = [128, 96, 72, 64]    # for SOC read reverse otsd limit
    cbusy_win_num            = 16
    rd_bdwt_limit_en         = 0
    rd_bdwt_limit_del_val    = 32                   # Decreased read bandwidth per cycle
    rd_bdwt_limit_wtline     = 512                  # read bandwith water line, exceed over it will be stall
    wr_bdwt_limit_en         = 0
    wr_bdwt_limit_del_val    = 32                   # Decreased write bandwidth per cycle
    wr_bdwt_limit_wtline     = 512                  # write bandwith water line, exceed over it will be stall
    # DSE PARAM
    256B_tag_backpressure    = 1
    rob_sp_arb               = 0                    # 0: su, 1: vec icache, 2: su dcache
    simt_use_fack_ack        = 0                    # 0: fack ack, 2: true ack
    wdapter_width            = 128
    su_dcache_width          = 32
    rdata_aic_mte_width      = 256
    rdata_aiv_mte_width      = 128
    cmd_buf_option           = 0                    # 0:all req rr 1:direct send
    wr_data_option           = 1                    # 0:all rr  1:simt sp + other rr
    brif_BL4_use_one_tag     = 0                    # 0: 512B use two tag, 256/128 use one; 1: all burst use one
    bwif_BL4_use_one_tag     = 0                    # 0: 512B use two tag, 256/128 use one; 1: all burst use one
    inorder_use_one_link     = 0                    # 0: default; 1: all inorder req put in one link;
    brif_rob_retry_          = 1
    brif_link_table_         = 2

[PARAM_BUFFER]
    num_slots         = 32  # 4KB 128B/slot
    num_sreg_per_slot = 64

[RVEC]
    pv_mode                         = "exec_instr"
    enable_peekvf_prefetch          = false
    lsu_dst_cflt_en                 = false
    csw_en                          = 0
    vc_vcb_mode                     = 1
    vag_ad_en                       = 1       # 0 1911l 1 1982
    area_reduce                     = false
    single_issue                    = false
    issue_option                    = 3       # 0: exq0; 1: exq0/1 one by one; 2: random, 3: group in fu, exq0/1 one by one in fu group; 4: shq_ent idx
    future_isa                      = false
    vgatherb_has_pg                 = true
    vag_32bit_enable                = true
    valu_try_en                     = true
    warm_up_cycle                   = 0
    vdintlv_single_dst              = false

[IFU]
    ibuf_size                        = 32
    fetch_size                       = 16
    dispatch_size                    = 2
    # icache
    icache_prefetch_uint             = 2048                     # 2K
    ic_addr_width                    = 48                       # [47:0]
    aic_asso_num                     = 4
    aiv_asso_num                     = 4
    aic_ic_size                      = 32768                    # 0x8000    #32k
    aiv_ic_size                      = 16384                    # 0x4000    #16k
    ic_line_size                     = 128                      # 1024/8
    aic_ic_entry_num                 = 64                       # (ic_size/ic_line_size)/ic_asso_num
    aic_ic_line_num                  = 256                      # (ic_size/ic_line_size)
    aiv_ic_entry_num                 = 32                       # (ic_size/ic_line_size)/ic_asso_num
    aiv_ic_line_num                  = 128                      # (ic_size/ic_line_size)
    ic_prefetch_en                   = 1
    ic_prefetch_num                  = 3
    ic_merge_en                      = 1
    ic_max_otsd_num                  = 16
    ic_max_preload_num               = 15
    aic_ic_idx_addr_lsb              = 7                        # VA[12:7]for tag idx
    aic_ic_idx_addr_mask             = 63
    aic_ic_tag_addr_lsb              = 13                       # VA[47:13]for va tag
    aic_ic_tag_addr_mask             = 34359738367
    aiv_ic_idx_addr_lsb              = 7                        # VA[11:7]for tag idx
    aiv_ic_idx_addr_mask             = 31
    aiv_ic_tag_addr_lsb              = 12                       # VA[47:12]for va tag
    aiv_ic_tag_addr_mask             = 68719476735
    read_stall_en                    = 0                        # 0 no stall 1 stall
    # vec ifu
    ifu_new_arch                     = true                     # false: 1951 lite/nsv   true:v310 1982
    vf_queue_depth                   = 32
    peek_queue_depth                 = 10
    instr_buffer_size                = 32
    prefetch_trigger_times           = 64
    peekvf_prefetch_trigger_times    = 64
    num_sreg_to_copy                 = 32
    para_write_size                  = 32
    icache_fetch_size                = 8
    max_instr_to_idu                 = 6
    icache_line_size                 = 128
    vf_early_fetch                   = true
    # vec icache
    vec_ic_size                      = 8192
    vec_ic_biu_bw                    = 128                      # 1911 128 / Cpro 32
    vec_ic_line_size                 = 128
    vec_ic_way_num                   = 2
    vec_ic_entry_num                 = 32                       # same with set_num : line_num / way_num
    vec_ic_ecc                       = 0
    vec_ic_data_ram_width            = 32
    vec_ic_offset_addr_mask          = 127                      # 0x7f aligned addr with cacheline
    vec_ic_otsd_num                  = 16                       # misq size
    vec_ic_pipe_length               = 1
    hit_info_buffer_size             = 1
    # calculate
    vec_ic_line_num                  = 64                       # size / (line_size)
    vec_ic_data_ram_num              = 4                        # line_size / data_ram_width
    # 2 way
    vec_ic_idx_addr_lsb              = 7                        # [6:0]cacheline addr for tag ram
    vec_ic_idx_addr_mask             = 31                       # [11:7] set_index
    vec_ic_tag_addr_lsb              = 12                       # [47:12] Tag info stored in ram
    vec_ic_tag_addr_mask             = 68719476735
    vec_ic_per_way_data_ram_bank_num = 2
    vec_ic_data_ram_bank_line_size   = 8

[CCU]
    aic_issque_size              = 16                              # cube issque size
    aic_issque_ostd_num          = 15
    aiv_issque_size              = 32                              # vector issque size
    aiv_issque_otsd_num          = 31
    mte1_issque_size             = 32
    mte1_issque_otsd_num         = 31
    fixp_issque_size             = 32
    fixp_issque_otsd_num         = 31
    aic_mte2_issque_size         = 16                              # aic mte2 issque size
    aic_mte3_issque_size         = 32                              # aic mte2/mte3 issque size
    aic_mte23_issque_otsd_num    = 31
    aiv_mte2_issque_size         = 16
    aiv_mte3_issque_size         = 16                              # aiv mte2/mte3 issque size
    aiv_mte23_issque_otsd_num    = 16
    event_queue_size             = 32
    I1_slot_num                  = 2                               # 1 single issue 2 exu dual issue
    # issque single port
    ar_option                    = 0                               # 0 unable 1:RR 2:WRR
    cube_weight                  = 2
    mte1_weight                  = 3
    fixp_weight                  = 1
    mte2_weight                  = 3
    vec_weight                   = 2
    mte3_weight                  = 1
    # issque pop multicycle
    mte_rs                       = [ 0, 0, 0, 0 ]
    vec_rs                       = 0
    cube_rs                      = 0
    # buffer_id
    buffer_id_ostd               = 32                              # max counter
    rld_entry_en                 = 1                               # 0 enable 1 enable  #rld_entry_size = 32 for every id
    bufId_Entry_print_en         = 0                               # 0 no print 1 print for every get_buf
    mte3_pipe_support            = 1

[SCALAR_Buf]
    base_address               = 262144            # 0x40000   # 256K
    total_size                 = 16384             # 0x4000
    wrap_en                    = 0
    sys_va_base_config         = 1                 # 0: config by model spr    1: config by spec
    sys_va_base                = 0
    stack_phy_base_config      = 1                 # 0: config by model spr    1: config by spec
    stack_phy_base             = 34603008
    mem_init_befor_start_en    = 0
    stack_buffer_mode          = 0
    memory_map_option          = 0                 # 0:1911L/1951L = 15M 1:nsv = 1M
    # asu
    idu_to_asu_ostd            = 1                 # a.k.a. issue queue depth

[SSBuf]
    base_address             = 262144           # 0x40000
    total_size               = 3072             # 3KB
    wrap_en                  = 0
    ssb_bandwidth            = 8                # B/cycle
    ssb_instr_size           = 4                # instr max num access ssb
    ssb_mode                 = 1                # 1 mix mode 0 single mode
    ssb_base_addr            = 1024             # aic and aic share
    set_intra_block_latency  = 2
    update_block_cnt_latency = 9
    rs_latency               = 5
    rs_size                  = 1

[DCACHE]
    aic_dc_set_size         = 128
    aic_dc_line_num         = 512
    aiv_dc_set_size         = 128
    aiv_dc_line_num         = 512                       # dc_line_num = dc_set_size * dc_way_size
    aic_dc_size             = 32                        # dc_size = dc_line_size * dc_way_size * dc_set_size;
    aiv_dc_size             = 32                        # dc_size = dc_line_size * dc_way_size * dc_set_size;
    aic_dc_line_size        = 64
    aiv_dc_line_size        = 64
    aic_dc_way_size         = 4
    aiv_dc_way_size         = 4
    dc_max_read_otsd_num    = 8
    dc_max_write_otsd_num   = 4
    dc_mshr_main_entry_num  = 8
    dc_mshr_sub_entry_num   = 8
    dc_stb_main_entry_num   = 8
    dc_stb_sub_entry_num    = 8
    dc_stb_timeout_cycles   = 32
    dc_dstb_buf_size        = 8
    dc_dstb_entry_ary_size  = 1
    dc_req_que_size         = 4
    dc_ub_write_allocate    = 0                         # 0: write without-allocate    1: write with-allocate    ### UB:  fixed write-through
    dc_ddr_write_allocate   = 1                         # 0: write without-allocate    1: write with-allocate    ### DDR: fixed write-back
    dc_ub_cacheable         = 0                         # cacheable=0 => without-allocate
    dc_lock_cacheline       = 0
    dc_lock_cacheline_num   = 2
    dc_lock_en              = 0
    dc_lock_base_addr       = 16384
    dc_stack_core_offset    = 32768
    aic_dc_idx_addr_lsb     = 6                         # [5:0] cacheline addr for tag ram
    aiv_dc_idx_addr_lsb     = 6
    aic_dc_idx_addr_mask    = 127                       # set(asso) num size
    aiv_dc_idx_addr_mask    = 127
    aic_dc_tag_addr_lsb     = 13                        # [11:6] cacheline addr for va tag
    aiv_dc_tag_addr_lsb     = 13
    dc_tag_addr_mask        = 68719476735               # [47:12] Tag info stored in ram
    dc_set_flag_en          = 1
    dc_stb_time_cnt         = 32
    dc_stb_drain_option     = 0                         # 0 old 1 caseA 2 caseB 3 caseC
    dc_stb_fakeReq_latency  = 2
    dc_hw_prefetch_en       = 0
    dc_hw_prefetch_num      = 1
    dc_hw_space_dis_cnt     = 12
    dc_hw_prefetch_max_ostd = 4

[MTE]
    # cube and vec
    cubecore_ue_mte1                  = ["UE3DV2", "UE2D", "UESET", "UEDMA", "UEWINOA", "UEWINOB", "UEDPW", "NZ2ND"]
    cubecore_ue_mte2                  = ["UE2D", "UESET", "UEDMA", "ND2NZ", "AIPP", "ALIGNv2_DMA2", "UNZIP"]
    cubecore_ue_mte3                  = ["UEDMA", "ALIGNv2_DMA3", "WAIPP", "L1OUT","NZ2ND_MTE3"]
    cubecore_ue_fixp                  = ["NZ2ND"]
    veccore_ue_mte1                   = []
    veccore_ue_mte2                   = ["UEDMA", "ALIGNv2_DMA2", "GATHER", "NDDMA"]
    veccore_ue_mte3                   = ["UEDMA", "ALIGNv2_DMA3", "SCATTER"]
    veccore_ue_fixp                   = []
    cubecore_intf                     = ["L1RIF", "L0CRIF", "L1WIF", "L0AWIF", "L0BWIF", "L1WIF_UB",]
    veccore_intf                      = ["UBRIF", "UBWIF",]
    # mixcore
    mixcore_ue_mte1                   = []                   # not used
    mixcore_ue_mte2                   = []                   # not used
    mixcore_ue_mte3                   = []                   # not used
    mixcore_ue_fixp                   = []                   # not used
    mixcore_intf                      = []                   # not used
    biu_burst                         = [512, 256, 128]
    split_burst_option                = 1                    # 0: old version, 1: Unpack as big as possible
    write_data_buffer_depth           = 0
    write_data_buffer_latency         = 5
    per_channel_en                    = 1
    wr_otsd_waterline                 = 0                    # write outstanding waterline, 0 means no waterline
    # move align
    merge_burst_len                   = 256
    ooo_burst_len                     = 128
    small_burst_merge                 = 1
    dst_merge_mode                    = 2
    # ub
    ub_to_l1_merge                    = 1                    # 0: 128B req send to l1; 1: merge 2 128B req to l1;
    ub2l1_intf_bandwidth              = 256
    l1_to_ub_merge                    = 0                    # 0: 128B req send to ub; 1: merge 2 128B req to ub;
    out_to_ub_merge                   = 0                    # 0: 128B req send to ub; 1: merge 2 128B req to ub;
    l1_to_ub_bandwidth                = 128
    ubrif_path_bandwidth              = 256
    ub_to_l1_bandwidth                = 128
    biu_to_ub_bandwidth               = 128
    ub_to_biu_bandwidth               = 128
    ubrif_retry_enable_               = 1
    ubrif_retry_path_empty_           = 1
    # mte1
    l1_to_l0a_bandwidth               = 256
    l1_to_l0b_bandwidth               = 256
    l1_to_l0a_mx_bandwidth            = 32
    l1_to_l0b_mx_bandwidth            = 32
    l1_to_l0c_bandwidth               = 64
    l1_to_smask_bandwidth             = 32
    l1_to_fb_post_bandwidth           = 128
    l1_to_fb_pre_bandwidth            = 32
    l1_to_fbv2_bandwidth              = 64
    mte2_antiq_parallelism            = 64
    l1_to_bt_bandwidth                = 64
    l1_to_pt_bandwidth                = 128
    l1_to_sp_bandwidth                = 128
    set2d_to_l0a_bandwidth            = 256
    set2d_to_l0b_bandwidth            = 256
    l0c_read_bandwidth                = 512
    l1_to_biu_bandwidth               = 128
    l1_to_l0a_wino_bandwidth          = 256
    l1_to_l0b_wino_bandwidth          = 256
    winoa_fetch_num                   = 8

    # mte2
    biu_to_l1_bandwidth               = 256
    biu_to_l0a_bandwidth              = 256
    biu_to_l0b_bandwidth              = 256
    set2d_to_l1_bandwidth             = 256
    lut_enable                        = 1

    # move align
    move_align_v2                     = 1
    mov_align_buf_size                = 1024                 # 512+128
    out2xx_l1wif_bdwt_split           = 1

    # move bt
    move_bt_fb_v2                     = 0

    # nd2nz
    nd2nz_buf_depth                   = 256
    nd2nz_buf_num                     = 32                   # only B8 is 32, nd2nz is 16, dn2nz is 8
    nd2nz_to_l1_bandwdith             = 256
    nd2nz_rd_burst_size               = 256                  # 1982 rd size max is 256B, old version is 512B
    bubble_cycle                      = 2                    # align buf addr recompute cycles
    advanced_align_len                = 256
    nd2nz_small_data_buf_depth        = 512
    shift_rs_mode                     = 1

    # fixp
    nz2nd_trans_buf_size_             = 16
    nz2nd_trans_buf_tmp_size_         = 8
    nz2nd_trans_buf_depth_            = 8
    fixp_pre_out_bandwidth            = 256
    fixp_wr_l1_bandwidth              = 128
    fixp_wr_ub_bandwidth              = 128
    fixp_wr_out_bandwidth             = 128
    fixp_rd_l0c_bandwidth             = 512
    compute_resource_parsim           = 64
    fixp_instr_fifo_depth             = 4
    bwif_acc_fixp_latency             = 5
    n_direction_pad                   = 0
    fixp_l0c_req_ost_                 = 256
    fixp_status_report_en             = 1
    fixp_bitmask_en                   = 0
    fixp_read_col_num                 = 16
    fixp_ndummy_en                    = 0
    fixp_nz2nd_row_merge_opt          = true
    nz2nd_row_merge_n_num             = 32
    fixp_to_ub_merge                  = false                # false: 128B req send to ub; true: merge 2 128B req to ub;
    fixp_nddn_512B_merge              = false
    fixp_head_latency                 = 3
    fixp_ub_increment_latency         = 7
    fixp_out_increment_latency        = 4
    que_pre_version                   = 3                    # 0: mini(1951 lite) | 1: lite(nashville) | 2: tiny | 3: cloud(1982)
    clear_rd_uf_crtl                  = true
    cube_l0c_f16_illegal              = true
    fb_legal_blk01                    = true
    david_quant_relu_cfg              = true
    david_atomic_cfg                  = true

    # aipp
    min_h_res                         = 8
    max_h_res                         = 4096
    byte_per_pixel_in_l1              = 32
    y_dat_buf_size                    = 64
    uv_dat_buf_size                   = 64
    rgb_dat_buf_size                  = 128
    uv_upsample_buf_size              = 4096
    sync_buf_size                     = 96
    csc_buf_size                      = 24
    dtc_buf_size                      = 48
    cpadding_buf_size                 = 256
    pixels_per_trans                  = 8
    img_dat_channels                  = 3
    aipp_dat_buf_bubble               = 3
    aipp_max_dtc_lat                  = 5
    aipp_dtc_u8_fp16_lat              = 5
    aipp_dtc_u8_s8_lat                = 1
    aipp_dtc_chl_offset               = 0
    chickenBit_en                     = 0
    dma_buffer_size_dc                = 8192
    dma_buffer_size_uc                = 16384
    dma_y_ping_buf_addr_uc            = 0                    # 0x0
    dma_y_pong_buf_addr_uc            = 16384                # 0x4000
    dma_uv_ping_buf_addr_dc           = 0                    # 0x0
    dma_uv_pong_buf_addr_dc           = 8192                 # 0x2000
    dma_y_ping_buf_addr_dc            = 16384                # 0x4000
    dma_y_pong_buf_addr_dc            = 24576                # 0x6000
    aipp_print_img_en                 = 1

    # waipp
    dtc_normalized                    = 1
    rd_proc_bytes                     = 32
    waipp_print_img_en                = 1

    # unzip
    unzip_fm_size                     = 512
    fm_size                           = 512
    unzip_pkt_size                    = 8
    unzip_head_size                   = 8
    unzip_dict_size                   = 34
    unzip_low_sparse_dict_size        = 36
    unzip_str_dict_size               = 1
    unzip_out_len                     = 64
    unzip_seg_size                    = 64
    unzip_buffer_size                 = 32768
    unzip_entry_size                  = 2
    max_fetch_idx_num                 = 32
    max_uzp_uop_crdt                  = 1
    unzip_delay_time                  = 4
    unzip_bypass_delay_time           = 2
    unzip_buffer_depth                = 1
    unzip_write_band_width            = 256
    unzip_engine_num                  = 4
    uzp_to_l1_bus_width               = 256
    uzp_to_l0b_bus_width              = 128

    # nd dma
    nddma_toml_en                     = 0
    shared_one_nacache                = 3
    nddma_aixs_num                    = [1, 16, 7, 5, 3, 23] # high->low
    nddma_dst_stride                  = [0, 2415, 345, 69, 23, 1]
    nddma_src_stride                  = [0, 105, 3, 21, 1, 1049055]
    nddma_lp_size                     = [1, 10, 31, 31, 20]
    nddma_rp_size                     = [0, 0, 0, 0, 0]
    nddma_add_stride                  = []
    src_addr                          = 352324896
    dst_addr                          = 97280
    element_size                      = 4
    subtensor_cycle                   = 15
    # ndcache
    ndcache_size                      = 8
    split_axis_ndcache_size           = 8192                 # for split axis algorithem
    ndcache_line_size                 = 128
    ndcache_fifo_depth                = 6144
    batch_element_num                 = 32
    cache_line_hash_idx               = 2
    cache_bank_hash_en                = 1
    8KB_hash_mode                     = 0                    # 0: default; 1: RL search
    biu_rd_que_depth                  = 64
    tag_ram_bank_num                  = 16
    ndcache_rd_ostd                   = 32
    ndcache_tag_ram_clr               = 1
    ndcache_retire_tag_ram_clr        = 0
    src_addr_cls_num_per_cycle        = 1
    ndcache_req_buf_depth             = 1
    dcache_dual_port                  = 0
    miss_fifo_buf                     = 5
    miss_fifo_rs_buf                  = 1
    wr_cache_backpressure             = 1
    addr_coalescer                    = 0                    # 0 : all_coalescer; 1: neighbor_coalescer
    dst_asize_delete_padding          = 1                    # the value is hardcoded in file
    32_element_gen_need_cycle         = 2
    ub_128_unalign_split              = 1
    ub_merge_en                       = 1
    ndcache_read_cycle                = 0
    delete_rd_cache_bank_cflt         = 0
    miss_replace_single_cycle         = 0
    div_scramble_mode                 = 2
    hash_scramble_mode                = 0
    # gather dma
    merge_all_burst                   = 0
    merge_burst_option                = 0                    # 0: default, 1: burst_num = 4 if burst_len <= 16, 2: burst_len <= 8, 3: burst_len <=4

    # ld2d lut mode
    ld2d_lut_3toX_384B_inorder        = 1
    lutxto4_fullbw                    = 1
    ld2d_early_start_otsd             = 2

[VEC]
    su_vec_depth     = 4
    aiv_stat_en_list = ["pem.veccore0.vec"]

[CUBE]
    cube_dummy_cycle_number         = 8
    cube_spec_npe                   = 256
    FP_partial_columns              = 4          # only 4(full), 2, 1 columns is supported
    FP32_FP16_mac_ratio             = 16         # only 4, 16 is supported
    C0_K_size                       = 32
    m_frac_size                     = 16
    n_frac_size                     = 16
    global_sync_pulse_phase_type    = 1
    vdrop_tick                      = 48
    fsm_ver                         = 1          # ver0: v100~v200; ver1: v210 and later projects
    mmad_fsm_n2_mode                = 0
    fsm_m_pri                       = 1
    fsm_loopm_loop8                 = true
    inner_loop_n_dir_ctrl           = false
    fsm_accum_mode3                 = false
    cube_stage_num                  = 26
    hset_l0c_check_stage            = 14
    hset_l0ab_check_stage           = 3
    hset_bt_check_stage             = 3
    wino_en                         = 0
    depthwise_en                    = 0
    group_conv_en                   = 0
    u8_en                           = false
    f16f16_en                       = false
    f16u2_en                        = false
    u8s8_en                         = false
    s4_en                           = false
    b8u2_en                         = false
    s16s8_en                        = false
    s8s4_en                         = false
    aic_stat_en_list                = ["pem.cubecore0.cube"]
    bt_fp16_compactly               = 1
    cube_sum_man_bw_mask            = 2
    is_shrink_exp_bits              = true
    cube_l0c_depend_check           = true
    cube_l0c_depend_depth           = 10

[L0A]
    total_size  = 65536  # 64
    wrap_en     = 0
    layoutzN_en = 1

[L0B]
    total_size = 65536  # 64 KB
    wrap_en    = 0

[L0C]
    total_size                  = 262144  # 256KB
    wrap_en                     = 0
    bank_count                  = 32
    cube_unit_flag_rd_lat       = 12      # cube wr req sent to rd unit flag set latency.
    l0c_vec_read_dat_latency    = 6       # arbitration to data back latency.

[SMASK]
    total_size = 256
    wrap_en    = 0

[L1]
    total_size                  = 524288       # 512KB
    wrap_en                     = 0
    buffer_bg_offset            = 18           # bank bit, if bank1 is 256KB, is the 18-th bit
    buffer_bg_num               = 2            # is bank num actually
    buffer_bank_count           = 8            # is bank group actually
    buffer_bank_width           = 32
    core_access_width           = 256
    dmac_access_width           = 256
    read_port_num               = 1
    write_port_num              = 2

[UB]
    total_size                   = 262144   # 256KB
    wrap_en                      = 0
    buffer_line_size             = 32
    subbank_line_size            = 8
    buffer_bank_count            = 16
    bank_id_offset               = 8        # bank ID bit offset in address
    bank_group_number            = 8
    bank_num_in_group            = 2
    # port: LSU_R, LSU_W, GSU, VEC_SU_R, VEC_SU_W, AIC_SU_R, AIC_SU_W, AIV_SU_R, AIV_SU_W, MTE_R, MTE_W, EXT_R, EXT_W
    port_num                     = [8, 8, 16, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4]
    # buffer_depth: VEC_SU_R, VEC_SU_W, AIC_SU_R, AIC_SU_W, AIV_SU_R, AIV_SU_W, MTE_R, MTE_W
    buffer_depth                 = [2, 7, 2, 7, 2, 7, 2, 2]
    # UB read access latency (not include UB_ARB): LSU_R, GSU_R, VEC_SU_R, AIC_SU_R, AIV_SU_R, MTE_R, SFU_R, SIMT_R, BHU_R
    read_latency                 = [3, 6, 5, 5, 5, 5, 5, 7, 3]
    # UB full write access latency (not include UB_ARB): LSU_W, GSU_W, VEC_SU_W, AIC_SU_W, AIV_SU_W, MTE_W, SFU_W, SIMT_W, BHU_W
    write_latency                = [2, 2, 2, 2, 2, 2, 2, 3, 6]
    exu_wb_latency               = 4
    ecc                          = 1
    ecc_byte                     = 4
    stu_ecc_byte                 = 4
    bank_conflict_chk            = 1
    xbar_conflict_chk            = true
    rdb_conflict_both            = false    # true: gsu_wb bpress both ldu port0/1; false: gsu_wb only bpress ldu port0
    ub_arb_option                = 0        # 0: default; 5: mte_w > ld > st
    ub_area_reduce               = true     # true: 32 bank group, 8-B bank_line; false: 16 bank group, 32-B bank_line
    bhu_rd_block_rmw             = true     # false
    alloc_rwdb_at_i3             = true
    rwdb_ent_num                 = 12
    first_wr_bypass_rmw          = false    # true(no rmw for dc write cacheline for the 1st time)
    free_uwdb_delayed            = false    # true(postpond free_uwdb by one cycle)
    gsu_lsu_share_shft           = false    # true(ga and ld share shft logic, ga will stall ld by two cycle)
    mte_high_pri_period          = 0        # 0: no high pri period, 1: high pri period is 1 cycle, 2: high pri period is 2 cycles
    ext_reuse_ldu_port0          = false    # true(ext_rd can access ub only when there is no ldu_port0 request)
    ext_rmw_rd_bubble            = false    # false(no bubble between ext_rmw_rd and prev ext_wr/ext_rmw_wr)
    ext_rmw_rd_check_addr        = false    # when ext_rmw_rd_bubble = true; true(has bubble only when addr is the same)
    xbar_cflt_stall_gsu_ib       = true     # true(stall gsu_ib-2-gsu_i2 when i3_xbar_cflt)

[FB]
    total_size                  = 6144 # 6 K
    wrap_en                     = 0
    fb_depth                    = 512

[FBv2]
    total_size                  = 4096 # 36 K
    wrap_en                     = 0

[BT]
    total_size                  = 4096 # 1 K
    wrap_en                     = 0

[PT]
    total_size                  = 131072 # 128K
    wrap_en                     = 0

[SPIDX]
    total_size                  = 16384 # 16 K
    wrap_en                     = 0

[L0AMX]
    total_size                  = 4096 # 4 K
    wrap_en                     = 0

[L0BMX]
    total_size                  = 4096 # 4 K
    wrap_en                     = 0

[REG]
    vreg_length                 = 256
    phy_vreg_num                = 68        # 52
    phy_preg_num                = 32        # 16; 64 availabe, simd only use 32
    vir_vreg_num                = 0         # for ooo vrat size  0:no pre-mapping  !0:just num pre-mapping
    vir_preg_num                = 0         # for ooo prat size  0:no pre-mapping  !0:just num pre-mapping
    is_reset_en                 = 1         # 0:reserved state    1:reset state

[IDU]
    idu_ib_buf_depth            = 12        # default 12 (including disp_window 6 entry)
    # dispatch number settings
    max_total_disp_num          = 6         # max of instr can be dispatched in a tick (IDU FIFO depth = this * 2)
    max_asu_disp_num            = 1
    max_vag_disp_num            = 1
    max_ldq_disp_num            = 5
    max_stq_disp_num            = 1         # used when stq available and shq not available
    max_exq_disp_num            = 2         # used when single issue
    # max_exq_disp_num            = 3 #single_issue
    max_shq_disp_num            = 5
    delay_isu2idu               = 1
    delay_ooo2idu               = 1
    check_sreg_hazard_en        = 1
    sreg_post_update_num        = 5
    max_uop_split_num           = 1

[ISU]
    shq_depth                   = 58
    exq_depth                   = 13
    ldq_depth                   = 24
    stq_depth                   = 0                            # if stq_depth == 0, sc/st/ga instr goes into shareq
    ga_enter_shq                = true
    ga_enter_stq                = false
    ldq_fifo_depth              = 0
    stq_fifo_depth              = 0
    shq_fifo_depth              = 0
    exq_fifo_depth              = 0
    max_ld_iss_width            = 2                            # from ldq to trailing stage
    max_ga_iss_width            = 1                            # from ldq/shq/stq to trailing stage
    max_st_iss_width            = 1                            # from shq/stq to trailing stage
    shq2exq0_iss_width          = 1
    shq2exq1_iss_width          = 1
    exq0_iss_width              = 1                            # ex_iss_width per que
    # exq0_iss_width               = 1 #area_reduce
    exq1_iss_width              = 1
    sqzn_fifo_depth             = 8                            # actual usable fifo depth is 8-1=7
    exq_output_pipe             = 0
    exq_decode_and_congruence   = 2
    exq_early_free_reg_enable   = 0                            # early-free-register switch
    exq_early_free_reg_time     = 4                            # tick to free reg before WB, =4
    exq_early_free_valu_thres   = 3                            # valu latency threshold, =3
    ld1_qsize                   = 4
    ld2_qsize                   = 4
    st_qsize                    = 4
    ex_qsize                    = 4
    gsu_intf_depth              = 1
    ooo_ldq_inf_size            = 5
    ooo_stq_inf_size            = 5
    ooo_shq_inf_size            = 5
    ooo_exq_inf_size            = 0                            # dual_issue
    # ooo_exq_inf_size            = 3 #single_issue
    ex_wreg_rd_pt_num           = 1
    ex_vreg_rd_pt_num           = 4
    ex_preg_rd_pt_num           = 4
    ex_wreg_wr_pt_num           = 1
    ex_vreg_wr_pt_num           = 1                            # dual_issue
    # ex_vreg_wr_pt_num           = 4 #single_issue
    ex_preg_wr_pt_num           = 1                            # dual_issue
    # ex_preg_wr_pt_num           = 2 #single_issue
    iss_alloc_inv_en            = 1                            # issue alloc_vld=0 (not mapped) instr
    max_stalled_tick            = 300                          # max stalled/stuck tick, for debugging
    ldq_max_inorder_iss_num     = 2                            # max inorder instr can be issued in one tick
    stq_max_inorder_iss_num     = 1
    exq_max_inorder_iss_num     = 1
    b2b_fwd_min_intvl           = 1                            # back-to-back forwarding min interval
    b2b_fwd_delta_time          = 2
    wb2byps_fwd_min_intvl       = 2                            # wb-to-bypass forwarding min interval
    wb2byps_fwd_delta_time      = 1
    regrwbpys_fwd_min_intvl     = 3                            # register-rw-bypass min interval
    regrwbyps_fwd_delta_time    = 0
    shq_pstg_window_plus        = 0                            # valid pipe stg modification
    shq_pstg_window_minus       = 1                            # valid pipe stg modification
    shq_pstg_rdy_thres          = 5                            # wake up by shq pstg since which stg
    shq_spclt_valu_ltcy_thres   = 4                            # put issued instr dst reg into shq reg rdy tbl if its valu_latency <= shq_spclt_valu_ltcy_thres
    exq_spclt_valu_ltcy_thres   = 7                            # put issued instr dst reg into shq reg rdy tbl if its shq_spclt_valu_ltcy_thres < valu_latency <= exq_spclt_valu_ltcy_thres
    shq_spclt_rd_ahead          = 1                            # for speculate src ready, src_ready more earlier in cycles
    exq_age_inorder_issue       = 0                            # 1(exq issue wakeup instr in age order, oldest first), 0 (exq issue wakeup instr in entering exq order)
    exq_fu_instr_limit          = [13, 13, 13, 13, 13, 13, 13] # [ADD_CONV, MUL, MOV, LUT, LNEXP, SLIDE, SFU]
    exq_fu_spcul_mgn            = [2, 2, 2, 2, 2, 2, 2]        # [ADD_CONV, MUL, MOV, LUT, LNEXP, SLIDE, SFU]
    exq_fu_alloc_init_status    = [1, 1, 1, 1, 1, 1, 1]        # [ADD_CONV, MUL, MOV, LUT, LNEXP, SLIDE, SFU] >0: exq1 first; 0: exq0 first
    shq_per_exq_b2b_threas      = [0, 0]                       # [EXQ0, EXQ1]
    shq_per_exq_idle_thres      = [3, 3]                       # [EXQ0, EXQ1]
    shq_per_exq_in_order_thres  = [3, 3]                       # [EXQ0, EXQ1]
    exq_fu_alloc_option         = [2, 2, 2, 2, 2, 0, 0]        # [ADD_CONV, MUL, MOV, LUT, LNEXP, SLIDE, SFU], 0 for exq0, 1 for exq1, 2 for exq0+exq1
    ldq_pipe_reduce             = true
    is_vsldb_ga_instr           = true
    no_cross_exq_bypass         = false                        # no wb_to_bypass data forwarding cross two exqs
    barrier_stu_gsu_both        = true                         # smem_bar barrier STU/GSU(GA,SC) at the same time
    src_rdy_no_spec_first       = false                        # shq to exq, instr whose src_rdy without speculate goes first
    bar_block_till_retire       = true                         # ture: smem_bar block shq instr till bar retire; false: till bar issue
    spec_que_size_dly           = 3
    que_fast_release            = false                        # fast release queue with fixed latency=1 when instr is issued
    spec_switch_for_both        = true                         # switch off shq spec for both exq instr at the same time
    alu_bps_with_dtype          = true                         # enable data_type chk for b2b, float type must match
    isu_no_shq_ready_tbl        = false                        # true: no shq ready table, but shq PSTG
    vsqz_no_spcl_src_rdy        = false                        # true: treat as vst instr

[RAT]
    vrat_rport_num              = 15
    vrat_wport_num              = 5
    prat_rport_num              = 15
    prat_wport_num              = 5

[OOO]
    idu_to_ooo_ostd             = 5
    ooo_to_ldq_ostd             = 5
    ooo_to_stq_ostd             = 5
    ooo_to_shq_ostd             = 5
    ooo_to_exq_ostd             = 2       # dual_issue
    # ooo_to_exq_ostd             = 2  #single_issue
    # vf_bc mode
    is_vf_bc_en                 = 0       # is vf_bc mode need remap rat
    pre_vrat_num                = 16      # vf_bc mode rat remap num
    pre_prat_num                = 8       # vf_bc mode rat remap num
    reg_group_free_en           = false   # true: free buffer; false: bitmap directly
    reg_free_buf_size           = 15
    reg_cycle_free_num          = 5
    mask_by_2cycle_pre          = false   # false: mask_by_1_cycle_previous

[GSU]
    instr_buf_depth             = [2, 1, 1]                   # [GATHER, SCATTER, TSU]
    timing_buffer_depth         = 4
    scatter_timing_buffer_depth = 6
    # pipe_stage_latency          = [1, 4, 1, 0, 3, 2, 5, 2, 1] #[INTF, VX, I0, I1, I2RD, I2WR, MX, RDB, WB]
    pipe_stage_latency          = [1, 5, 1, 1, 0, 2, 5, 2, 1] # [INTF, VX, I0, I1, I2RD, I2WR, MX, RDB, WB]
    ub_access_para              = [0, 2, 3, 6, 7, 17]         # [bank_addr_lsb_, bank_addr_msb_, bank_id_lsb_, bank_id_msb_, align_addr_lsb_, align_addr_msb_]  #area_reduce
    # ub_access_para              = [0, 4, 5, 7, 8, 17] #[bank_addr_lsb_, bank_addr_msb_, bank_id_lsb_, bank_id_msb_, align_addr_lsb_, align_addr_msb_]
    input_parallelism           = 128
    paralleism                  = 64
    # paralleism                  = 16  #area_reduce
    bank_num                    = 32
    # bank_num                    = 16  #area_reduce
    coaleas_en                  = 0
    gsu_element_size            = 2

[SFU]
    nchw_pipe_latency           = [ 2, 13, 2, 4, 2,]
    # ub_access_para              = [ 16, 17, 5, 8,]
    ub_access_para              = [ 8, 8, 3, 7 ] # [bank_id_lsb_, bank_id_msb_, bank_bg_lsb_, bank_bg_msb_]
    nchw_max_rd_blk             = 16
    nchw_max_wr_blk             = 8
    tran_max_rd_blk             = 16
    tran_max_wr_blk             = 8
    dma_max_rd_blk              = 8
    dma_max_wr_blk              = 8
    vms4_max_rd_rp_num          = 16
    vms4_max_wr_rp_num          = 8
    vms4_ibuf_crdt              = 15
    vms4_ibuf_latency           = 9
    max_rd_ub_crdt              = 8
    vms4_retire_latency         = 10
    no_access_latency           = 3
    vbs32_latency               = 18
    vbs32_thrput                = 16
    vnchwconv_rd_bank_num       = 4

[TSU]
    dir0_b8_chn_num             = 16
    dir0_b16_chn_num            = 16
    dir0_b32_chn_num            = 8
    dir1_b8_chn_num             = 8
    dir1_b16_chn_num            = 8
    dir1_b32_chn_num            = 8
    dir0_b8_hw_num              = 4
    dir0_b16_hw_num             = 4
    dir0_b32_hw_num             = 8
    dir1_b8_hw_num              = 16
    dir1_b16_hw_num             = 16
    dir1_b32_hw_num             = 8
    v4dtrans_uop_intlv_num      = 16
    v4dtrans_engine_latcy       = 2
    v4dtrans_retire_latcy       = 8
    vnchwconv_retire_latcy      = 9
    vnchwconv_wr_latency        = 21
    vnchwconv_wr_port           = 1

[LSU]
    ldu_rd_reg_latency          = 1
    stu_rd_reg_latency          = 2
    ldu_time_buf_depth          = 2
    stu_time_buf_depth          = 6
    max_blk_num                 = 8
    single_issue_vldu           = true          # true: vldu/vldus/vldui goes to ldu_port_0
    single_issue_vld            = true          # true: vld/vlds/vldi(dist_mode=e2b_b16/e2b_b32/unpk4_b8/unpk_b32/unpk_b16/unpk_b8/us_b8/us_b16/brc_b8/brc_b16/brc_b32) goes to ldu_port_0
    single_issue_by_stu         = false         # true: ldu single issue when stu timing buf is full

[MVF]
    mvf_dcache_en                 = 2
    dc_cache_size                 = 192         # 192KB
    dc_cacheline_size             = 128
    dc_bank_num                   = 4           # 32*4=128
    dc_cacheline_num              = 1536        # 1536=192*1024/128
    dc_tag_bank_size              = 4
    dc_way_size                   = 4
    dc_set_size                   = 128         # 96 = 192*1024/128/4/4   or 128 = 256*1024/128/4/4
    dc_tag_bank_lsb               = 7           # [8:7]
    dc_tag_bank_mask              = 3           # 0x3
    dc_addr_coalescer_lab         = 7           # [MAB:7]
    dc_addr_coalescer_mask        = 67108863    # 0x3ffffff
    dc_tag_index_lsb              = 9           # [15:9]
    dc_tag_index_mask             = 127         # 0x7f
    dc_tag_addr_lsb               = 16          # [31:16]
    dc_tag_addr_mask              = 65535       # 0xffff
    dc_max_miss_fifo_size         = 200         # tmp
    dc_max_rd_biu_ostd            = 32          # tmp
    dc_free_cacheline_reclaim_num = 32          # tmp
    dc_max_process_ostd           = 32          # tmp
    ub_index_window               = 32
    # DSE
    fix_window                    = 1           # 1: fix windows merge wr req; 0: slide windows

[SIMT]

[SIMT.SYS]
    num_cores               = 2

[SIMT.ARCH]
    total_reg_num           = 131072 # 131072
    total_phy_p_reg         = 32     # we have 16.  But each element is 256 bit, / (32threads x 4issue) = 2 entries
    total_phy_v_reg         = 26     # we have 52, but this time it's the reverse.  Each registers are 256B, while there are 32thread x 4B x 4 issue = 512B is required per entry
    total_sb                = 32
    max_warp_num            = 64
    thread_num              = 32     # per warp
    issue_num               = 4
    stack_base              = 327680
    warp_stack_size         = 4096
    debug                   = false
    area_reduce             = false
    merge_ic_dc_biu_port    = true
    # valu_option (SIMD/SIMT)
    # SIMD
    # 9:            v0.8.0==v0.8.1 # DavidV100
    # 109:          v0.8.0==v0.8.1 # DavidFPGA
    # 210:          v0.8.1 # DavidV130
    # SIMT
    # 1:            v0.7.2
    # 2:            v0.7.3
    # otherwise:    v0.7.5
    valu_option             = 9
    isa_version             = 1176
    su_mte_wr_ub_backpr     = 0      # 0: no backpressue, 1: mte backpressue, 2: su/mte backpressue, 3: bif backpressue mte, 4: bif backpressue su/mte
    retire_delay            = 5
    clock_ini_val           = 0
    clock_from_time         = true
    thread_idx_option       = 0

[SIMT.SCH]
    l1_sch_slot_num         = 8
    l2_sch_slot_num         = 8
    l2_copy_scb_latency     = 6
    swap_wait_cycle         = 1
    s2r_bkpr_vthId_init     = 2
    preg_buffer_num         = 2
    exec_token_bitwidth     = 12
    exec_token_cmp_lsb      = 3
    exec_token_threshold    = 2048 # 2^11
    bypass_occupy_rd_port   = true
    wb_rd0_bypass_enable    = true
    src2_bypass_enable      = false
    gto_arb_switch          = 0    # 0: greedy-then-oldest;
                                   # 1: 一个warp slot不能连续发射两条指令
                                   # 2: a.第1优先级GTO
                                   # b.第2优先级选择exec_token最小的warp
                                   # c.exec token一样，RR策略

[SIMT.REG]
    max_r_reg_num           = 127   # to reserve the RZ encoding
    max_p_reg_num           = 7     # to reserve the PT encoding
    max_s_reg_num           = 64    # SREG can only be used as the source register within a vector function
    sram_bank_num           = 2     # TBC
    dc_rd_latency           = 6
    dc_wr_latency           = 2
    lsu_wr_port             = 2
    lsu_write_option        = 1     # 0: old option; 1: new option1: single port; 2: new option2: dual port, only wsu0/2 or wsu1/3

[SIMT.IFU]
    task_start_latency      = 7
    usable_scb_id_num       = 4
    scb_cnt_bitwidth        = 3     # [0-31]
    update_scb_latency      = 3
    ibuf_fifo_depth         = 16
    ibuf_fifo_width         = 8
    ibuf_fifo_write_bw      = 64
    lsu_req_pre_buf_size    = 2
    vthreadid_dop           = 8
    total_vthread_calc      = 4
    current_prefetch_num    = 4
    hold_range              = 8
    memory_range            = 4096
    ibuf_lookup_back        = false
    miss_bypass             = false
    cur_prefetch_en         = true
    ins_fetch_reset         = false
    prefetch_switch         = 1

[SIMT.DVG]
    stack_entry_num         = 16      # 16 for option1, option2 will write overflow stack to ddr, model will modify this option later
    dvg_update_latency      = 1
    stack_base              = 655360
    warp_stack_size         = 4096
    spill_entry_num         = 6
    spill_size              = 128
    stack_spill_otsd        = 1

[SIMT.LSU]
    agu_que_depth           = 10
    wpayload_buf_depth      = 12    # uwdb entry num
    wpayload_size_per_entry = 128   # uwdb size per entry
    exclusive_switch        = 0     # 0: atomic_load, 1: LDEX/STEX
    atom_unit_fifo          = 64    # only valid for ARlock/AWlock approach, ostd to wait for both Rd/Wr EXOK
    atomic_option           = 0     # 0 reqMerge 1 as_review
    atomic_agu_option       = 1     # 0 orign case 1 new case
    atomic_s_option         = 1     # 0 old case 1 new case:no ostd between warp
    stack_exception_en      = 1     # 1 no access with exception
    uwdb_write_port         = 2
    pblsq_num               = 4
    atomic_func_en          = 1     # 1 function enable 0 no function
    ub_expert_mode          = 1

[SIMT.EXU]
    rd_reg_latency          = 10    # TBC
    wr_reg_latency          = 10    # TBC
    mufu_option             = 1
    pv_shfl_option          = 0     # 0: ISA; 1: RTL

[SIMT.DCACHE]
    line_size               = 128     # bytes
    sector_size             = 128
    way_num                 = 4
    bank_num                = 4
    set_num                 = 64      # per bank
    coalesce_depth          = 32
    bpsq_depth              = 1
    bpsq_uop_buf_depth      = 8
    miss_fifo_len           = 256
    mrob_ram_type           = 0       # 0(SPRAM), 1(TPRAM)
    frc_thr_low             = 40
    frc_thr_hign            = 80
    drcu_stall_clc_thr      = 25
    reclaim_option          = 2       # 0(random) , 1(LRU), 2(inorder)
    freelist_option         = 1       # 0(fifo) , 1(bitmap)
    cl_group_num            = 32
    random_min              = 0
    random_max              = 512
    max_ref_cnt             = 31
    max_wref_cnt            = 7
    ub_sz                   = 262144
    shmem_sz                = 131072
    max_shmem_sz            = 229376  # (SIMT, DCU vld case check,don't change)
    max_dcu_sz              = 131072  # (SIMT, DCU vld case check,don't change)
    stk_wrmode              = 0       # 0(writeback) , 1(writethrough)
    stg_wrmode              = 1       # 0(writeback) , 1(writethrough)
    missfifo_num            = 1
    evictfifo_depth         = 24
    evictfifo_rdy_delay     = 4
    evictfifo_cl_free_delay = 4
    rwdb_entry_size         = 128
    bpsq_rwdb_ent_num       = 16
    mrob_rwdb_ent_num       = 16
    mrob_thread_inner_split = 1       # 0(disable mrob req thread inner split), 1(enable mrob req thread inner split)
    dc_st_bpsq_req_no_merge = 1       # 0(bpsq st req from dc will merge), 1(bpsq st req from dc will merge not)
    arb_bpsq_wr_over_mrob   = true    # false(mrob_rd has higher priority), true(bpsq_wr has higher priority)
    # BHU prebuf
    bhu_prebuf_depth        = 9
    bhu_prebuf_ram_type     = 1       # 0(SPRAM), 1(TPRAM)
    # BHU postbuf
    bhu_postbuf_ram_type    = 0       # 0(SPRAM), 1(TPRAM)
    bhu_comp_hash_op        = 2       # 0(no hash 41bit) #1(no hash 32bit) #2(hash,16bit) #3(hash,10bit) #4(no hash,16bit)
    bhu_postbuf_depth       = 128
    bhu_postbuf_rd_issfifo  = 4       # minimum is 4
    bhu_postbuf_wr_issfifo  = 4       # minimum is 4
    bhu_rd_pipe_req_fifo    = 32
    bhu_perf                = 0       # 0(close) #1(open)
    bhu_biu_rw_otsd         = 128
    bhu_dvg_otsd            = 1
    tagram_lru_op           = 0       # 0(LRU) #1(PLRU)
    atomg_cas_one_uop       = false
    atom_all_sig_thread     = 1       # 0(!cas multi thread) #1(!cas sigle thread)
    acc_bmp_op              = 0       # 0(drcu chk acc_bmp) #1(drcu ignore acc_bmp)
    drcu_cnt_op             = 1       # 0(drcu chk refcnt/wrefcnt) #1(drcu ignore refcnt/wrefcnt)
    tagram_squez_bubble     = 0       # 0(C5/C6 don't squez bubble) #1(C5/C6 squez bubble)

[SIMD.IFU]
    tde_sreg_latency             = 11
    max_issue_cnt                = 6
    max_loop_layer               = 4
    ib_size                      = 32
    ib_write_cnt                 = 8
    buf_mode_ib_size             = 24
    buf_mode_inner_second_delay  = 1
    buf_mode_second_inner_delay  = 2

[L2CACHE]
    cache_enable                = 0
    cache_set_size              = 24
    cache_way_size              = 16384
    cache_line_size             = 512
    cache_read_latency          = 241
    cache_write_latency         = 96
)!!!";

    outFile << config;

    outFile.close();
    return;
}

void PvModelCaseConfigBase::SetTitle(std::string title)
{
    title_ = title;
}

void PvModelCaseConfigBase::SetCoreType(uint64_t coreType)
{
    subcoreId_ = coreType;
}

std::uint64_t PvModelCaseConfigBase::GetCoreType() 
{
    return subcoreId_;
}

void PvModelCaseConfigBase::SetBin(uint64_t addr, std::string path) {
    binAddr_ = addr;
    binPath_ = path;
}

void PvModelCaseConfigBase::AddInputArg(uint64_t addr, uint64_t size, std::string path) {
    inputArgs_.emplace_back(std::make_tuple(addr, size, path));
}

void PvModelCaseConfigBase::AddOutputArg(uint64_t addr, uint64_t size, std::string path) {
    outputArgs_.emplace_back(std::make_tuple(addr, size, path));
}

void PvModelCaseConfig::Dump(std::string path) {
    std::fstream file(path, std::ios::out);

    if (!file.is_open()) {
        SIMULATION_LOGE("ErrCode: F%u, [PVMODEL]open config file error: %s", 
                    static_cast<unsigned>(CostModel::ExternalErrorScene::FILE_OPEN_FAILED), path.c_str());
        return;
    }

    file << "title = \"" << title_ << "\"" << std::endl;
    file << "log_open_value = 0xffffffff" << std::endl;
    file << "path = \"./\"" << std::endl;
    file << "hbm_para_addr = 0xffff8000" << std::endl;
    file << "chip_version = 6" << std::endl;
    file << "subcore_id = " << subcoreId_ << std::endl;
    file << "block_idx = 0" << std::endl;
    file << std::endl;

    file << "[BIN]" << std::endl;
    file << "name = \"" << binPath_ << "\"" << std::endl;
    file << "addr = " << "0x" << std::hex << binAddr_ << std::endl;
    file << std::endl;

    constexpr int pathIdx = 2;
    constexpr int addrIdx = 0;
    constexpr int sizeIdx = 1;
    int paraOffset = 0;
    for (const auto &arg : inputArgs_) {
        file << "[[input_para_array]]" << std::endl;
        file << "name = \"" << std::get<pathIdx>(arg) << "\"" << std::endl;
        file << "addr = " << "0x" << std::hex << std::get<addrIdx>(arg) << std::endl;
        file << "size = " << "0x" << std::hex << std::get<sizeIdx>(arg) << std::endl;
        file << "valid = 1" << std::endl;
        file << "para_offset = " << std::dec << paraOffset << std::endl;
        paraOffset++;
        file << std::endl;
    }

    for (const auto &arg : outputArgs_) {
        file << "[[output_para_array]]" << std::endl;
        file << "name = \"" << std::get<pathIdx>(arg) << "\"" << std::endl;
        file << "addr = " << "0x" << std::hex << std::get<addrIdx>(arg) << std::endl;
        file << "size = " << "0x" << std::hex << std::get<sizeIdx>(arg) << std::endl;
        file << "valid = 1" << std::endl;
        file << "para_offset = " << std::dec << paraOffset << std::endl;
        paraOffset++;
        file << std::endl;
    }

    file.close();
}
} // namespace CostModel
