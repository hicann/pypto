#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor compare utilility."""
import os
import logging
from typing import NamedTuple, Optional, Tuple, Union
import torch
import pandas as pd
from tabulate import tabulate


class IsCloseConfig(NamedTuple):
    """check_isclose 函数的配置参数"""
    rtol: float = 1.0e-2
    atol: float = 1.0e-2
    calc_dtype: torch.dtype = torch.float64
    shape: Optional[Union[Tuple, list]] = None
    is_ignore_bothzero: bool = True
    is_detail: bool = False
    fail_factor: int = 128
    is_extra: bool = False


class TensorComparator:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", encoding="utf-8")
            ]
        )
    
    @staticmethod
    def save_info_to_csv(d_detail, csv_path, topk):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        num = len(d_detail[0])
        if num == 0:
            pd.DataFrame(columns=["No", "off", "a", "b", "ad", "rd", "ad_tol", "off_raw"]).to_csv(csv_path, index=False)
            return
        if num > 0:
            num_picked = min(topk, num) if topk and topk >= 0 else num
        table_data = []
        for i in range(num_picked):  # 遍历所有数据，不再受 topk 限制
            info_off = d_detail[0][i].item()
            off_raw = d_detail[1][i]
            off_raw = off_raw.tolist() if off_raw.dim() > 0 else [off_raw.item()]
            info_off_raw = '_'.join([str(s) for s in off_raw])

            row = [
                i + 1,
                info_off,
                f"{d_detail[2][i].item():.6g}",
                f"{d_detail[3][i].item():.6g}",
                f"{d_detail[4][i].item():.6g}",
                f"{d_detail[5][i].item():.6g}",
                f"{d_detail[6][i].item():.6g}",
                info_off_raw
            ]
            table_data.append(row)

        headers = ["No", "off", "a", "b", "ad", "rd", "ad_tol", "off_raw"]
        df = pd.DataFrame(table_data, columns=headers)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"数据已保存到: {csv_path}")
    
    @staticmethod
    def check_isclose(a, b, config: IsCloseConfig = IsCloseConfig()):
        rtol, atol, calc_dtype, shape, is_ignore_bothzero, is_detail, fail_factor, is_extra = config
        if calc_dtype not in [torch.float64, torch.float32]:
            raise ValueError(f'not support calculating dtype: {calc_dtype}')
        aa = a.flatten()
        bb = b.flatten()
        a = aa.to(calc_dtype)
        b = bb.to(calc_dtype)

        a_abs = a.abs()
        b_abs = b.abs()
        ab_sub = (a - b)
        ab_sub_abs = ab_sub.abs()
        ab_abs_add = (a_abs + b_abs)
        tol_warn = ab_abs_add * rtol / 2 + atol
        tol_fail = tol_warn * fail_factor
        mask_bothzero = (ab_abs_add == 0)
        mask_warn = torch.gt(ab_sub_abs, tol_warn)
        mask_infnan = ab_sub_abs.isfinite().logical_not()
        mask_fail = torch.gt(ab_sub_abs, tol_fail)
        cnt_all = mask_warn.numel()
        cnt_out_warn = mask_warn.sum().item()
        cnt_out_bothzero = mask_bothzero.sum().item()
        cnt_out_pass = cnt_all - cnt_out_warn - cnt_out_bothzero
        if cnt_out_pass < 0:
            raise ValueError(f'cnt_out_pass > 0: {cnt_out_pass}')
        if is_ignore_bothzero:
            cnt_picked = cnt_all - cnt_out_bothzero    
            if (cnt_all - cnt_out_bothzero) != (cnt_out_warn + cnt_out_pass):
                raise ValueError(f'(cnt_all - cnt_out_bothzero) == (cnt_out_warn + cnt_out_pass)')
        else:
            cnt_picked = cnt_all
        cnt_fail = mask_fail.sum().item()
        cnt_infnan = mask_infnan.sum().item()

        tol_cnt = tol_cnt_raw = int(cnt_picked * min(rtol, atol))
        if tol_cnt_raw == 0:
            # tol_cnt is zero, adjust to small value
            tol_cnt = min(16, int(cnt_picked**0.5) // 2)                            

        if not is_detail:
            const_empty_arg = torch.tensor([], dtype=torch.int64)
            const_empty_value = torch.tensor([], dtype=calc_dtype)
            data_warn_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
            data_fail_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
            data_infnan_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 5])
            data_diff_info_list = tuple([*[const_empty_arg] * 2, *[const_empty_value] * 4])
        else:
            ab_ad = ab_sub
            ab_rd = ab_sub_abs * 2 / ab_abs_add
            arg_warn_raw = arg_warn = torch.argwhere(mask_warn).flatten()
            arg_fail_raw = arg_fail = torch.argwhere(mask_fail).flatten()
            arg_infnan_raw = arg_infnan = torch.argwhere(mask_infnan).flatten()
            if (a.dim() > 1) or (b.dim() > 1):
                if shape:
                    to_shape = shape
                else:
                    to_shape = a.shape if a.dim() >= b.dim() else b.shape
                arg_warn_raw = torch.argwhere(mask_warn.reshape(to_shape))
                arg_fail_raw = torch.argwhere(mask_fail.reshape(to_shape))
                arg_infnan_raw = torch.argwhere(mask_infnan.reshape(to_shape))
            data_warn_info_list = (arg_warn, arg_warn_raw, aa.take(arg_warn), bb.take(arg_warn), 
                                    ab_ad.take(arg_warn), ab_rd.take(arg_warn), tol_warn.take(arg_warn))
            data_fail_info_list = (arg_fail, arg_fail_raw, aa.take(arg_fail), bb.take(arg_fail), ab_ad.take(arg_fail),
                                     ab_rd.take(arg_fail), tol_fail.take(arg_warn))
            data_infnan_info_list = (arg_infnan, arg_infnan_raw, aa.take(arg_infnan), bb.take(arg_infnan), 
                                        ab_ad.take(arg_infnan), ab_rd.take(arg_infnan), arg_infnan)
            ab_rd_max = (ab_rd[~torch.isnan(ab_rd)].max().item() if ab_rd[~torch.isnan(ab_rd)].numel() != 0
                            else float('nan'))
            data_diff_info_list = (ab_sub_abs.max().item(), ab_rd_max, ab_sub_abs.mean().item(), ab_rd.nanmean().item(),
                                    cnt_out_warn, cnt_out_warn / ab_sub_abs.numel(), cnt_out_bothzero,
                                     cnt_out_bothzero / ab_sub_abs.numel())

        # weak/strong warning
        if not is_extra:
            cnt_warn_ww = cnt_warn_w = cnt_warn_s = cnt_warn_ss = 0
        else:
            tol_warn_ww = tol_warn / 4
            tol_warn_w = tol_warn / 2
            tol_warn_s = tol_warn * 2
            tol_warn_ss = tol_warn * 4
            mask_warn_ww = torch.gt(ab_sub_abs, tol_warn_ww)
            mask_warn_w = torch.gt(ab_sub_abs, tol_warn_w)
            mask_warn_s = torch.gt(ab_sub_abs, tol_warn_s)
            mask_warn_ss = torch.gt(ab_sub_abs, tol_warn_ss)
            cnt_warn_ww = mask_warn_ww.sum().item()
            cnt_warn_w = mask_warn_w.sum().item()
            cnt_warn_s = mask_warn_s.sum().item()
            cnt_warn_ss = mask_warn_ss.sum().item()

        diff_cnt = (cnt_all, cnt_picked, cnt_out_bothzero, cnt_out_pass, cnt_out_warn, cnt_fail, cnt_infnan)
        diff_conf = (rtol, atol, fail_factor, tol_cnt)
        _diff_extra = (cnt_warn_ww, cnt_warn_w, cnt_out_warn, cnt_warn_s, cnt_warn_ss)
        diff_detail_warn = data_warn_info_list
        diff_detail_fail = data_fail_info_list
        diff_detail_infnan = data_infnan_info_list
        result_is_close = (cnt_out_warn <= tol_cnt) and (cnt_fail <= 0) and (cnt_infnan <= 0)
        result_reason_str = []
        if (cnt_out_warn > tol_cnt):
            result_reason_str.append(f'cnt_warn(={cnt_out_warn}) > tol_cnt(={tol_cnt})')
        if (cnt_fail > 0):
            result_reason_str.append(f'cnt_fail(={cnt_fail}) > 0)')
        if (cnt_infnan > 0):
            result_reason_str.append(f'cnt_infnan(={cnt_infnan}) > 0)')
        result_reason_str = ','.join(result_reason_str)

        result_info = (diff_cnt, diff_conf, _diff_extra, diff_detail_warn, diff_detail_fail,
                        diff_detail_infnan, data_diff_info_list)

        return result_is_close, result_reason_str, result_info

    def print_isclose_info(self, result_is_close, result_reason_str, result_info, path, topk=10000):
        (d_cnt, d_conf, _d_extra, d_detail_warn, d_detail_fail, d_detail_infnan, d_diff_conf) = result_info
        (cnt_all, cnt_picked, cnt_out_bothzero, cnt_out_pass, cnt_out_warn, cnt_fail, cnt_infnan) = d_cnt
        (rtol, atol, fail_factor, tol_cnt) = d_conf
        (cnt_warn_ww, cnt_warn_w, cnt_out_warn, cnt_warn_s, cnt_warn_ss) = _d_extra
        sep = ', '
        logging.info(f'cnt : {cnt_all}{sep}{cnt_picked}{sep}{cnt_out_bothzero}{sep}{cnt_out_pass}{sep}\
                    {cnt_out_warn}{sep}{cnt_fail}{sep}{cnt_infnan}\t(all/picked/zero/pass/warn/fail/infnan)')
        logging.info(f'conf : {rtol:g}{sep}{atol:g}{sep}{fail_factor}{sep}{tol_cnt}\t(rtol/atol/fail_factor/tol_cnt)')

        if sum(_d_extra) != _d_extra[2]:
            logging.info(f'_extra : {cnt_warn_ww}{sep}{cnt_warn_w}{sep}{cnt_out_warn}{sep}\
                        {cnt_warn_s}{sep}{cnt_warn_ss}\t(ww/w/warn/s/ss)')
        logging.info(f'is_close : {result_is_close}\t({result_reason_str})')

        self.save_info_to_csv(d_detail_warn, path, topk)