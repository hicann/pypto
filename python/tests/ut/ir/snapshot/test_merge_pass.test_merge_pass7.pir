@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    View_x@3 = VIEW(x@0)
    for loop_idx_37 in ir.range(0, (((RUNTIME_GetInputShapeDim(ARG_x,0)-0)/4)*4), 4, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 4}):
        if (loop_idx_37==0):
            if ((loop_idx_37+3)==(RUNTIME_GetInputShapeDim(ARG_x,0)-1)):
                $0@5 = ADDS(View_x@3)
                $5@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5@9)
                $8@11 = SUBS(View_x@3)
                $12@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12@14)
                $15@16 = SUBS(View_x@3)
                $19@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19@19)
                $22@21 = SUBS(View_x@3)
                $25@23 = ADDS(View_x@3)
                z@2 = ASSEMBLE($25@23)
                ir.yield_()
            else:
                $0_1@5 = ADDS(View_x@3)
                $5_2@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_2@9)
                $8_2@11 = SUBS(View_x@3)
                $12_2@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_2@14)
                $15_2@16 = SUBS(View_x@3)
                $19_2@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_2@19)
                $22_2@21 = SUBS(View_x@3)
                $26@24 = SUBS(View_x@3)
                z@2 = ASSEMBLE($26@24)
                ir.yield_()
            ir.yield_()
        else:
            if ((loop_idx_37+3)==(RUNTIME_GetInputShapeDim(ARG_x,0)-1)):
                $1@6 = SUBS(View_x@3)
                $5_5@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_5@9)
                $8_5@11 = SUBS(View_x@3)
                $12_5@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_5@14)
                $15_5@16 = SUBS(View_x@3)
                $19_5@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_5@19)
                $22_5@21 = SUBS(View_x@3)
                $25_0@23 = ADDS(View_x@3)
                z@2 = ASSEMBLE($25_0@23)
                ir.yield_()
            else:
                $1_1@6 = SUBS(View_x@3)
                $5_4@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_4@9)
                $8_4@11 = SUBS(View_x@3)
                $12_4@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_4@14)
                $15_4@16 = SUBS(View_x@3)
                $19_4@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_4@19)
                $22_4@21 = SUBS(View_x@3)
                $26_0@24 = SUBS(View_x@3)
                z@2 = ASSEMBLE($26_0@24)
                ir.yield_()
            ir.yield_()
        continue
    for loop_idx_37_0 in ir.range((((RUNTIME_GetInputShapeDim(ARG_x,0)-0)/4)*4), RUNTIME_GetInputShapeDim(ARG_x,0), 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_37_0==0):
            if (loop_idx_37_0==(RUNTIME_GetInputShapeDim(ARG_x,0)-1)):
                $30@25 = ADDS(View_x@3)
                $34@28 = ADDS(View_x@3)
                z@2 = ASSEMBLE($34@28)
                ir.yield_()
            else:
                $30_1@25 = ADDS(View_x@3)
                $35@29 = SUBS(View_x@3)
                z@2 = ASSEMBLE($35@29)
                ir.yield_()
            ir.yield_()
        else:
            if (loop_idx_37_0==(RUNTIME_GetInputShapeDim(ARG_x,0)-1)):
                $31@26 = SUBS(View_x@3)
                $34_0@28 = ADDS(View_x@3)
                z@2 = ASSEMBLE($34_0@28)
                ir.yield_()
            else:
                $31_1@26 = SUBS(View_x@3)
                $35_0@29 = SUBS(View_x@3)
                z@2 = ASSEMBLE($35_0@29)
                ir.yield_()
            ir.yield_()
        continue
    return x@0, y@1, z@2
