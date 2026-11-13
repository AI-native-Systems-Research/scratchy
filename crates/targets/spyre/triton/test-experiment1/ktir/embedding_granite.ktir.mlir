#loc = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":103:1)
#loc2 = loc(unknown)
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 + d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1 + d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 49158 >= 0, d1 >= 0, -d1 + 4095 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 4095 >= 0)>
#set3 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 4095 >= 0)>
#loc11 = loc("desc_ids"(#loc))
#loc12 = loc("desc_table"(#loc))
#loc13 = loc("desc_o"(#loc))
module {
  tt.func public @embedding_fwd(%desc_ids: !tt.ptr<i32> loc("desc_ids"(#loc)), %desc_table: !tt.ptr<f16> loc("desc_table"(#loc)), %desc_o: !tt.ptr<f16> loc("desc_o"(#loc))) attributes {noinline = false} {
    %start_m = arith.constant 32 : index loc(#loc14)
    %start_m_0 = arith.constant 4 : index loc(#loc14)
    %c0 = arith.constant 0 : index loc(#loc2)
    %cst = arith.constant 1.200000e+01 : f16 loc(#loc3)
    %splat = tensor.splat %cst : tensor<64x4096xf16> loc(#loc3)
    %c64_i32 = arith.constant 64 : i32 loc(#loc2)
    %start_m_1 = ktdp.get_compute_tile_id : index loc(#loc14)
    scf.for %start_m_2 = %start_m_1 to %start_m_0 step %start_m {
      ktdf.corelet_plan pattern = "single_corelet" {
        ktdf.corelet 0 {data_bounds = [0, 1]} loc(#loc14)
      } loc(#loc14)
      %start_m_3 = arith.index_cast %start_m_2 : index to i32 loc(#loc14)
      %ids_desc = builtin.unrealized_conversion_cast %desc_ids : !tt.ptr<i32> to index loc(#loc15)
      %ids_desc_4 = ktdp.construct_memory_view %ids_desc, sizes: [256], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<256xsi32> loc(#loc15)
      %table_desc = builtin.unrealized_conversion_cast %desc_table : !tt.ptr<f16> to index loc(#loc16)
      %table_desc_5 = ktdp.construct_memory_view %table_desc, sizes: [49159, 4096], strides: [4096, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<49159x4096xf16> loc(#loc16)
      %o_desc = builtin.unrealized_conversion_cast %desc_o : !tt.ptr<f16> to index loc(#loc17)
      %o_desc_6 = ktdp.construct_memory_view %o_desc, sizes: [256, 4096], strides: [4096, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<256x4096xf16> loc(#loc17)
      %offs_m = arith.muli %start_m_3, %c64_i32 : i32 loc(#loc18)
      %ids = arith.index_cast %offs_m : i32 to index loc(#loc19)
      %ids_7 = ktdp.construct_access_tile %ids_desc_4[%ids] {access_tile_order = #map, access_tile_set = #set3} : memref<256xsi32> -> !ktdp.access_tile<64xindex> loc(#loc19)
      %ids_8 = ktdp.load %ids_7 : <64xindex> -> tensor<64xi32> loc(#loc19)
      %rows = ktdp.construct_indirect_access_tile %table_desc_5 captures(%ids, %c0 : index, index) indirect(%ids_desc_4 : memref<256xsi32>) {
      ^bb0(%arg4: index loc(unknown), %arg5: index loc(unknown)):
      } {per_dim_subscript_kinds = [true, false], per_dim_subscript_maps = [#map1, #map2], variables_space_order = #map3, variables_space_set = #set4} : memref<49159x4096xf16> -> <64x4096xindex> loc(#loc20)
      %rows_9 = ktdp.load %rows : <64x4096xindex> -> tensor<64x4096xf16> loc(#loc20)
      %0 = arith.mulf %rows_9, %splat : tensor<64x4096xf16> loc(#loc3)
      %1 = arith.index_cast %offs_m : i32 to index loc(#loc10)
      %2 = ktdp.construct_access_tile %o_desc_6[%1, %c0] {access_tile_order = #map3, access_tile_set = #set4} : memref<256x4096xf16> -> !ktdp.access_tile<64x4096xindex> loc(#loc10)
      ktdp.store %0, %2 : tensor<64x4096xf16>, <64x4096xindex> loc(#loc10)
    } loc(#loc14)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":109:15)
#loc3 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":129:31)
#loc4 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":112:16)
#loc5 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":115:18)
#loc6 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":118:14)
#loc7 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":122:14)
#loc8 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":123:11)
#loc9 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":126:12)
#loc10 = loc("/Users/nickm/git/triton-spyre/.claude/worktrees/paged-attn-lowering/third_party/spyre/test/fixtures/embedding.py":129:5)
#loc14 = loc("start_m"(#loc1))
#loc15 = loc("ids_desc"(#loc4))
#loc16 = loc("table_desc"(#loc5))
#loc17 = loc("o_desc"(#loc6))
#loc18 = loc("offs_m"(#loc7))
#loc19 = loc("ids"(#loc8))
#loc20 = loc("rows"(#loc9))
