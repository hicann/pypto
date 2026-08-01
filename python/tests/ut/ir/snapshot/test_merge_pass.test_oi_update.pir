@ir.function
def foo(a@0: ir.Tensor, b@1: ir.Tensor):
    for loop_idx_10, (b_1,) in ir.range(0, 10, 1, init_values=(b@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        for loop_idx_16, (b_3,) in ir.range(0, 10, 1, init_values=(b_1@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
            oi_update_1@2 = TENSOR_ALLOC()
            for loop_idx_34, (b_5, oi_update_3) in ir.range(0, (((n-0)/2)*2), 2, init_values=(b_3@1, oi_update_1@2), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 2}):
                if (loop_idx_34==0):
                    if (n<=(loop_idx_34+2)):
                        oi_update_4@2 = ADDS(a@0)
                        oi_update_9@2 = SUBS(oi_update_4@2)
                        $6@9 = DIVS(oi_update_9@2)
                        b_9@1 = FLOOR($6@9)
                        oi_update_31, b_11 = ir.yield_(oi_update_9@2, b_9@1)
                    else:
                        oi_update_33@2 = ADDS(a@0)
                        oi_update_35@2 = SUBS(oi_update_33@2)
                        oi_update_31, b_11 = ir.yield_(oi_update_35@2, b_5@1)
                    oi_update_27, b_26 = ir.yield_(oi_update_31@2, b_11@1)
                else:
                    if (n<=(loop_idx_34+2)):
                        oi_update_5@2 = SUBS(oi_update_3@2)
                        oi_update_45@2 = SUBS(oi_update_5@2)
                        $6_0@9 = DIVS(oi_update_45@2)
                        b_30@1 = FLOOR($6_0@9)
                        oi_update_39, b_28 = ir.yield_(oi_update_45@2, b_30@1)
                    else:
                        oi_update_41@2 = SUBS(oi_update_3@2)
                        oi_update_43@2 = SUBS(oi_update_41@2)
                        oi_update_39, b_28 = ir.yield_(oi_update_43@2, b_5@1)
                    oi_update_27, b_26 = ir.yield_(oi_update_39@2, b_28@1)
                b_13, oi_update_13 = continue b_26@1, oi_update_27@2
            for loop_idx_34_0, (b_15, oi_update_15) in ir.range((((n-0)/2)*2), n, 1, init_values=(b_13@1, oi_update_13@2), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
                if (loop_idx_34_0==0):
                    if (n<=(loop_idx_34_0+1)):
                        oi_update_16@2 = ADDS(a@0)
                        $10@13 = DIVS(oi_update_16@2)
                        b_16@1 = FLOOR($10@13)
                        oi_update_47, b_18 = ir.yield_(oi_update_16@2, b_16@1)
                    else:
                        oi_update_49@2 = ADDS(a@0)
                        oi_update_47, b_18 = ir.yield_(oi_update_49@2, b_15@1)
                    oi_update_19, b_32 = ir.yield_(oi_update_47@2, b_18@1)
                else:
                    if (n<=(loop_idx_34_0+1)):
                        oi_update_17@2 = SUBS(oi_update_15@2)
                        $10_0@13 = DIVS(oi_update_17@2)
                        b_34@1 = FLOOR($10_0@13)
                        oi_update_51, b_36 = ir.yield_(oi_update_17@2, b_34@1)
                    else:
                        oi_update_53@2 = SUBS(oi_update_15@2)
                        oi_update_51, b_36 = ir.yield_(oi_update_53@2, b_15@1)
                    oi_update_19, b_32 = ir.yield_(oi_update_51@2, b_36@1)
                b_20, oi_update_21 = continue b_32@1, oi_update_19@2
            b_22 = continue b_20@1
        b_24 = continue b_22@1
    return a@0, b_24@1
