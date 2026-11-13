#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#set5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
module attributes {spyre.grid_dead_chain_ops_erased = 11 : i64, spyre.grid_i32_ops_left = 9 : i64, spyre.grid_index_chains_rebuilt = 2 : i64} {
  func.func @attn_fwd(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) attributes {grid = [8 : index], spyre.folded_grid_loop = {num_cores = 32 : index, work_items = 8 : index}} {
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.275630e-01 : f16
    %splat = tensor.splat %cst : tensor<64x64xf16>
    %cst_0 = arith.constant 1.275630e-01 : f16
    %splat_1 = tensor.splat %cst_0 : tensor<64xf16>
    %cst_2 = arith.constant 1.000000e+00 : f16
    %splat_3 = tensor.splat %cst_2 : tensor<64xf16>
    %cst_4 = arith.constant 0xFC00 : f16
    %splat_5 = tensor.splat %cst_4 : tensor<64xf16>
    %cst_6 = arith.constant 0.000000e+00 : f16
    %splat_7 = tensor.splat %cst_6 : tensor<64x64xf16>
    %cst_8 = arith.constant 0.000000e+00 : f16
    %splat_9 = tensor.splat %cst_8 : tensor<128x64xf16>
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.divui %0, %c2 : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.divsi %2, %c4_i32 : i32
    %4 = arith.remsi %2, %c4_i32 : i32
    %5 = ktdp.construct_memory_view %arg0, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %6 = ktdp.construct_memory_view %arg1, sizes: [256, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<256x128x64xf16>
    %7 = ktdp.construct_memory_view %arg2, sizes: [128, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<128x256x64xf16>
    %8 = ktdp.construct_memory_view %arg3, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %9 = arith.muli %3, %c256_i32 : i32
    %10 = arith.divsi %4, %c2_i32 : i32
    %11 = arith.muli %10, %c128_i32 : i32
    %12 = arith.addi %9, %11 : i32
    %c2_10 = arith.constant 2 : index
    %c0_11 = arith.constant 0 : index
    %c2_12 = arith.constant 2 : index
    %13 = arith.divui %0, %c2_12 : index
    %c4 = arith.constant 4 : index
    %14 = arith.divui %13, %c4 : index
    %c512 = arith.constant 512 : index
    %15 = arith.muli %14, %c512 : index
    %c2_13 = arith.constant 2 : index
    %16 = arith.divui %0, %c2_13 : index
    %c4_14 = arith.constant 4 : index
    %17 = arith.remui %16, %c4_14 : index
    %c128 = arith.constant 128 : index
    %18 = arith.muli %17, %c128 : index
    %19 = arith.addi %15, %18 : index
    %c2_15 = arith.constant 2 : index
    %20 = arith.remui %0, %c2_15 : index
    %c64 = arith.constant 64 : index
    %21 = arith.muli %20, %c64 : index
    %22 = arith.addi %19, %21 : index
    %c2_16 = arith.constant 2 : index
    %23 = arith.muli %22, %c2_16 : index
    %24 = ktdp.construct_access_tile %5[%23, %c0_11] {access_tile_order = #map, access_tile_set = #set3} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
    %25 = ktdp.load %24 : <128x64xindex> -> tensor<128x64xf16>
    %26:5 = scf.for %arg5 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg6 = %splat_9, %arg7 = %splat_3, %arg8 = %splat_5, %arg9 = %12, %arg10 = %12) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
      %42 = arith.index_cast %arg9 : i32 to index
      %c0_28 = arith.constant 0 : index
      %c0_29 = arith.constant 0 : index
      %43 = ktdp.construct_access_tile %6[%42, %c0_28, %c0_29] {access_tile_order = #map1, access_tile_set = #set4} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
      %44 = ktdp.load %43 : <64x128x1xindex> -> tensor<64x128x1xf16>
      %45 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%44, %25 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %in_40: f16, %out: f16):
        %68 = arith.mulf %in, %in_40 : f16
        %69 = arith.addf %out, %68 : f16
        linalg.yield %69 : f16
      } -> tensor<64x64xf16>
      %cst_30 = arith.constant 0xFC00 : f16
      %splat_31 = tensor.splat %cst_30 : tensor<64xf16>
      %46 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%45 : tensor<64x64xf16>) outs(%splat_31 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %68 = arith.maxnumf %in, %out : f16
        linalg.yield %68 : f16
      } -> tensor<64xf16>
      %47 = arith.mulf %46, %splat_1 : tensor<64xf16>
      %48 = arith.maxnumf %arg8, %47 : tensor<64xf16>
      %49 = arith.mulf %45, %splat : tensor<64x64xf16>
      %expanded_32 = tensor.expand_shape %48 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_33 = tensor.collapse_shape %expanded_32 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %50 = tensor.empty() : tensor<64x64xf16>
      %51 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_33 : tensor<64xf16>) outs(%50 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<64x64xf16>
      %52 = arith.subf %49, %51 : tensor<64x64xf16>
      %53 = math.exp2 %52 : tensor<64x64xf16>
      %54 = arith.subf %arg8, %48 : tensor<64xf16>
      %55 = math.exp2 %54 : tensor<64xf16>
      %cst_34 = arith.constant 0.000000e+00 : f16
      %splat_35 = tensor.splat %cst_34 : tensor<64xf16>
      %56 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%53 : tensor<64x64xf16>) outs(%splat_35 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %68 = arith.addf %in, %out : f16
        linalg.yield %68 : f16
      } -> tensor<64xf16>
      %expanded_36 = tensor.expand_shape %55 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_37 = tensor.collapse_shape %expanded_36 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %57 = tensor.empty() : tensor<128x64xf16>
      %58 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_37 : tensor<64xf16>) outs(%57 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<128x64xf16>
      %59 = arith.mulf %arg6, %58 : tensor<128x64xf16>
      %60 = arith.index_cast %arg10 : i32 to index
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %61 = ktdp.construct_access_tile %7[%c0_38, %60, %c0_39] {access_tile_order = #map1, access_tile_set = #set5} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
      %62 = ktdp.load %61 : <128x64x1xindex> -> tensor<128x64x1xf16>
      %63 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62, %53 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%59 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %in_40: f16, %out: f16):
        %68 = arith.mulf %in, %in_40 : f16
        %69 = arith.addf %out, %68 : f16
        linalg.yield %69 : f16
      } -> tensor<128x64xf16>
      %64 = arith.mulf %arg7, %55 : tensor<64xf16>
      %65 = arith.addf %64, %56 : tensor<64xf16>
      %66 = arith.addi %arg9, %c64_i32 : i32
      %67 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %63, %65, %48, %66, %67 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
    }
    %expanded = tensor.expand_shape %26#1 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
    %27 = tensor.empty() : tensor<128x64xf16>
    %28 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed : tensor<64xf16>) outs(%27 : tensor<128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<128x64xf16>
    %29 = arith.divf %26#0, %28 : tensor<128x64xf16>
    %c2_17 = arith.constant 2 : index
    %c0_18 = arith.constant 0 : index
    %c2_19 = arith.constant 2 : index
    %30 = arith.divui %0, %c2_19 : index
    %c4_20 = arith.constant 4 : index
    %31 = arith.divui %30, %c4_20 : index
    %c512_21 = arith.constant 512 : index
    %32 = arith.muli %31, %c512_21 : index
    %c2_22 = arith.constant 2 : index
    %33 = arith.divui %0, %c2_22 : index
    %c4_23 = arith.constant 4 : index
    %34 = arith.remui %33, %c4_23 : index
    %c128_24 = arith.constant 128 : index
    %35 = arith.muli %34, %c128_24 : index
    %36 = arith.addi %32, %35 : index
    %c2_25 = arith.constant 2 : index
    %37 = arith.remui %0, %c2_25 : index
    %c64_26 = arith.constant 64 : index
    %38 = arith.muli %37, %c64_26 : index
    %39 = arith.addi %36, %38 : index
    %c2_27 = arith.constant 2 : index
    %40 = arith.muli %39, %c2_27 : index
    %41 = ktdp.construct_access_tile %8[%40, %c0_18] {access_tile_order = #map, access_tile_set = #set3} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
    ktdp.store %29, %41 : tensor<128x64xf16>, <128x64xindex>
    return
  }
}

