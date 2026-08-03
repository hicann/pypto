@ir.function
def foo(a@0: ir.Tensor, b@1: ir.Tensor):
    for loop_idx_12, (b_1,) in ir.range(0, (((n-0)/2)*2), 2, init_values=(b@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 2}):
        if ((loop_idx_12+1)==(n-1)):
            $3@4 = VEC_DUP()
            b_5@1 = ADDS($3@4)
            b_7 = ir.yield_(b_5@1)
        else:
            b_7 = ir.yield_(b_1@1)
        b_9 = continue b_7@1
    for loop_idx_12_0, (b_11,) in ir.range((((n-0)/2)*2), n, 1, init_values=(b_9@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_12_0==(n-1)):
            $8@6 = VEC_DUP()
            b_12@1 = ADDS($8@6)
            b_14 = ir.yield_(b_12@1)
        else:
            b_14 = ir.yield_(b_11@1)
        b_16 = continue b_14@1
    return a@0, b_16@1
