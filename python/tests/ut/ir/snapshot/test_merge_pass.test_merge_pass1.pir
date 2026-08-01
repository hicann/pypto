@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    for loop_idx_10 in ir.range(0, 2, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_10==0):
            View_x@3 = VIEW(x@0)
            View_y@4 = VIEW(y@1)
            $0@5 = ADD(View_x@3, View_y@4)
            $4@8 = SUB($0@5, View_x@3)
            z@2 = ASSEMBLE($4@8)
            ir.yield_()
        else:
            if (loop_idx_10==1):
                View_x_8@3 = VIEW(x@0)
                View_y_8@4 = VIEW(y@1)
                $1@6 = SUB(View_x_8@3, View_y_8@4)
                $3@7 = ADD($1@6, View_x_8@3)
                z@2 = ASSEMBLE($3@7)
                ir.yield_()
            else:
                View_x_6@3 = VIEW(x@0)
                View_y_6@4 = VIEW(y@1)
                $1_1@6 = SUB(View_x_6@3, View_y_6@4)
                $4_0@8 = SUB($1_1@6, View_x_6@3)
                z@2 = ASSEMBLE($4_0@8)
                ir.yield_()
            ir.yield_()
        continue
    return x@0, y@1, z@2
