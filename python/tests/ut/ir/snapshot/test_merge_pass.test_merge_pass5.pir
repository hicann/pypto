@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    View_x@3 = VIEW(x@0)
    View_y@4 = VIEW(y@1)
    for loop_idx_32 in ir.range(0, 2, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_32==0):
            $0@5 = ADD(View_x@3, View_y@4)
            $3@7 = ADD($0@5, View_x@3)
            z@2 = ASSEMBLE($3@7)
            ir.yield_()
        else:
            $1@6 = SUB(View_x@3, View_y@4)
            $3_1@7 = ADD($1@6, View_x@3)
            z@2 = ASSEMBLE($3_1@7)
            ir.yield_()
        continue
    return x@0, y@1, z@2
