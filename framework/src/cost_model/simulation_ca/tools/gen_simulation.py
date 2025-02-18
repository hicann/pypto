#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import subprocess
import argparse
from collections import defaultdict, deque

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))


def parse_args():
    parser = argparse.ArgumentParser(description="处理文件示例")
    parser.add_argument("input", type=str, help="输入文件名称")

    return parser.parse_args()

tileop_dict = {"UB_ALLOC":"PIPE_S",
               "TileOp::Tadd":"PIPE_V",
               "TileOp::Tsub":"PIPE_V",
               "TileOp::Tmul":"PIPE_V",
               "TileOp::Tdiv":"PIPE_V",
               "TileOp::Tmax":"PIPE_V",
               "TileOp::exp":"PIPE_V",
               "TileOp::Texp":"PIPE_V",
               "TileOp::Trsqrt":"PIPE_V",
               "TileOp::Tsqrt":"PIPE_V",
               "TileOp::Tadds":"PIPE_V",
               "TileOp::Tsubs":"PIPE_V",
               "TileOp::Tmuls":"PIPE_V",
               "TileOp::Tdivs":"PIPE_V",
               "TileOp::Tgather":"PIPE_V",
               "TileOp::Tgatherelement":"PIPE_V",
               "TileOp::Tscatterelement":"PIPE_S",
               "TileOp::Tindexput":"PIPE_V",
               "TileOp::Treshape":"PIPE_V",
               "TileOp::Tconcat":"PIPE_V",
               "TileOp::TtransposeMoveOut":"PIPE_MTE3",
               "TileOp::TRowsumline":"PIPE_V",
               "TileOp::Ttranspose_vnchwconv":"PIPE_V",
               "TileOp::Trowsumsingle":"PIPE_V",
               "TileOp::Trowmaxsingle":"PIPE_V",
               "TileOp::Treducemaxsingle":"PIPE_V",
               "TileOp::Treducesumsingle":"PIPE_V",
               "TileOp::Tcompact":"PIPE_V",
               "TileOp::Texpand":"PIPE_V",
               "TileOp::Trec":"PIPE_V",
               "TileOp::Tcast":"PIPE_V",
               "TileOp::Treducesum":"PIPE_V",
               "TileOp::Trowmaxexpand":"PIPE_V",
               "TileOp::UBCopyIn":"PIPE_MTE2",
               "TileOp::UBCopyOut":"PIPE_MTE3",
               "TileOp::TIndexoutcast":"PIPE_MTE3",
               "L1_ALLOC":"PIPE_S",
               "L0A_ALLOC":"PIPE_S",
               "L0B_ALLOC":"PIPE_S",
               "L0C_ALLOC":"PIPE_S",
               "FIX_ALLOC":"PIPE_S",
               "TileOp::L1CopyIn":"PIPE_MTE2",
               "TileOp::L1CopyOut":"PIPE_FIX",
               "TileOp::L0CCopyOut":"PIPE_FIX",
               "TileOp::L1ToL0A":"PIPE_MTE1",
               "TileOp::L1ToL0B":"PIPE_MTE1",
               "TileOp::L1ToL0Bt":"PIPE_MTE1",
               "FIX_COPY_IN":"PIPE_MTE1",
               "L1_COPY_UB":"PIPE_FIX",
               "L0C_COPY_UB":"PIPE_FIX",
               "UB_COPY_L1":"PIPE_MTE3",
               "TileOp::Tmad":"PIPE_M",
               "SYNC_SRC":"PIPE_S",
               "SYNC_DST":"PIPE_S",
               "BAR.V":"PIPE_S",
               "BAR.M":"PIPE_S",
               "CV_SYNC_SRC":"PIPE_S",
               "CV_SYNC_DST":"PIPE_S",
               "PHASE1":"PIPE_S",
               "PHASE2":"PIPE_S",
               "TileOp::AllGather":"PIPE_V",
               "TileOp::ReduceScatter":"PIPE_V",
               "TileOp::AllReduce":"PIPE_V",
               "TileOp::AlltoAll":"PIPE_V",
               "TileOp::Send":"PIPE_V",
               "TileOp::Recv":"PIPE_V",
               "TileOp::BitSort":"PIPE_V",
               "TileOp::MrgSort":"PIPE_V",
               "TileOp::Extract":"PIPE_V",
               "TileOp::Argsort":"PIPE_V",
               "REGISTER_COPY":"PIPE_V",
               "VIEW":"PIPE_S",
               "ASSEMBLE":"PIPE_S", }


def calc_graph(graph, line_dict):
    ans = 0
    nodes = []
    print(graph)
    for edge in graph:
        nodes.extend(edge)
    nodes = list(set(nodes))
    dis = {}

    for node in nodes:
        dis[node] = line_dict[node][0]

    in_degree = defaultdict(int)
    for edge in graph:
        in_degree[edge[1]] += 1

    queue = deque([node for node in nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        ans = max(ans, dis[node])

        for edge in graph:
            if edge[0] == node:
                neighbor = edge[1]
                in_degree[neighbor] -= 1
                dis[neighbor] = max(dis[neighbor], dis[node] + line_dict[neighbor][0])
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    sorted_dict = {key: dis[key] for key in sorted(dis)}
    print(sorted_dict)
    return ans


def main():

    args = parse_args()
    file_path = args.input

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.rstrip('\n') for line in lines]

        print(f"成功读取 {len(lines)} 行内容。")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        exit(-1)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        exit(-1)

    func_id = 0
    line_id = 0

    header_list = []
    prefix = ["#include <iostream>", "#include \"mock_inst.h\"", "#include \"vector.h\"",
              "#include \"mte.h\"", "#include \"cube.h\"", "int main(int argc, char **argv) {"]
    suffix = ["    return 0;", "}"]

    line_dict = {}

    graphs = []
    graph = []

    flag = {}

    pipe_dict = {"PIPE_V": 0, "PIPE_MTE2": 1, "PIPE_MTE3": 2, "PIPE_S": 3, "PIPE_FIX": 4, "PIPE_MTE1": 5, "PIPE_M": 6}
    last_pipe = [0 for i in range(len(pipe_dict))]
    next_flag = [0 for i in range(len(pipe_dict))]



    ans = 0
    for line in lines:
        line_id += 1
        if line.strip().startswith("[aicore]"):
            ans = 0
            header_list = []
            func_id += 1
            if len(graph) > 0:
                graphs.append(graph)
            graph = []
            last_pipe = [0 for i in range(len(pipe_dict))]
            header_list.append(f"char charArray0" + "[256] = {0};")
            header_list.append(f'__ubuf__ float *UB_ID0neg_Addr =(__ubuf__ float *)charArray0;')


        if line.strip().startswith("}"):
            if len(graph) > 0:
                graphs.append(graph)

        if line.strip().startswith("set_flag"):
            pl = line[9:-2]
            p_list = pl.split(', ')
            p_tuple = tuple(p_list)
            flag[p_tuple] = last_pipe[pipe_dict[p_list[0]]]

        if line.strip().startswith("wait_flag"):
            pl = line[10:-2]
            p_list = pl.split(', ')
            p_tuple = tuple(p_list)
            next_flag[pipe_dict[p_list[1]]] = flag[p_tuple]

        if line.strip().startswith("TileOp::"):
            line_params = line.split(',')

            for p in line_params:
                p = p.strip(");")

                if r"half*)" in p:
                    c = p.split(r"half*)")[1]
                    line = line.replace(c, "UB_ID0neg_Addr")
                if r"float*)" in p:
                    c = p.split(r"float*)")[1]
                    line = line.replace(c, "UB_ID0neg_Addr")
                if r"*)" in p:
                    c = p.split(r"*)")[1]
                    line = line.replace(c, "UB_ID0neg_Addr")
                if "float *)" in p:
                    c = p.split("float *)")[1]
                    line = line.replace(c, "UB_ID0neg_Addr")
                if "half *)" in p:
                    c = p.split("half *)")[1]
                    line = line.replace(c, "UB_ID0neg_Addr")
            file_path = f'TileOp{func_id}-{line_id}.cpp'
            with open(file_path, 'w', encoding='utf-8') as file:
                cur = prefix + header_list + [line] + suffix
                file.write("\n".join(cur))
            cmd = (f"g++ -w -std=c++17 {file_path} -o {file_path[0:-4]} "
                   f"-I {project_dir}/framework/src/codegen/tileop -I {project_dir}/framework/src/cost_model/simulation/arch/mock")
            return_code = os.system(cmd)

            process = subprocess.Popen(f"./{file_path[0:-4]}",
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
            stdout, stderr = process.communicate()

            input_content = stdout.strip()
            # 启动 .exe 程序，并将标准输入、标准输出和标准错误输出重定向
            simulation_exe = project_dir + "/build/output/bin/get_cce_simulation"
            process = subprocess.Popen(simulation_exe,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)

            # 向程序发送输入内容，并获取程序的输出结果
            stdout, stderr = process.communicate(input=input_content)
            ans += int(stdout.strip())

            func_type = tileop_dict[line.split("<")[0]]

            if next_flag[pipe_dict[func_type]] != 0:
                graph.append([next_flag[pipe_dict[func_type]], line_id])
                next_flag[pipe_dict[func_type]] = 0

            if last_pipe[pipe_dict[func_type]] != 0:
                graph.append([last_pipe[pipe_dict[func_type]], line_id])
            last_pipe[pipe_dict[func_type]] = line_id
            line_dict[line_id] = [int(stdout.strip()), func_type]

            # 获取程序的返回状态码

    for graph in graphs:
        print("graph cycle result: ", calc_graph(graph, line_dict))

if __name__ == "__main__":
    main()
