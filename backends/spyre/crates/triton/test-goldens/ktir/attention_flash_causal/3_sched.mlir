#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#set6 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
module attributes {spyre.grid_dead_chain_ops_erased = 8 : i64, spyre.grid_i32_ops_left = 17 : i64, spyre.grid_index_chains_rebuilt = 2 : i64} {
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
    %c1_i32 = arith.constant 1 : i32
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
    %1 = arith.remui %0, %c2 : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.divui %0, %c2 : index
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.divsi %4, %c4_i32 : i32
    %6 = arith.remsi %4, %c4_i32 : i32
    %7 = ktdp.construct_memory_view %arg0, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %8 = ktdp.construct_memory_view %arg1, sizes: [256, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<256x128x64xf16>
    %9 = ktdp.construct_memory_view %arg2, sizes: [128, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<128x256x64xf16>
    %10 = ktdp.construct_memory_view %arg3, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %11 = ktdp.construct_memory_view %arg4, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set3, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64x64xf16>
    %12 = arith.muli %2, %c64_i32 : i32
    %13 = arith.muli %5, %c256_i32 : i32
    %14 = arith.divsi %6, %c2_i32 : i32
    %15 = arith.muli %14, %c128_i32 : i32
    %16 = arith.addi %13, %15 : i32
    %c2_10 = arith.constant 2 : index
    %c0_11 = arith.constant 0 : index
    %c2_12 = arith.constant 2 : index
    %17 = arith.divui %0, %c2_12 : index
    %c4 = arith.constant 4 : index
    %18 = arith.divui %17, %c4 : index
    %c512 = arith.constant 512 : index
    %19 = arith.muli %18, %c512 : index
    %c2_13 = arith.constant 2 : index
    %20 = arith.divui %0, %c2_13 : index
    %c4_14 = arith.constant 4 : index
    %21 = arith.remui %20, %c4_14 : index
    %c128 = arith.constant 128 : index
    %22 = arith.muli %21, %c128 : index
    %23 = arith.addi %19, %22 : index
    %c2_15 = arith.constant 2 : index
    %24 = arith.remui %0, %c2_15 : index
    %c64 = arith.constant 64 : index
    %25 = arith.muli %24, %c64 : index
    %26 = arith.addi %23, %25 : index
    %c2_16 = arith.constant 2 : index
    %27 = arith.muli %26, %c2_16 : index
    %28 = ktdp.construct_access_tile %7[%27, %c0_11] {access_tile_order = #map, access_tile_set = #set4} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
    %29 = ktdp.load %28 : <128x64xindex> -> tensor<128x64xf16>
    %30:5 = scf.for %arg5 = %c0_i32 to %12 step %c64_i32 iter_args(%arg6 = %splat_9, %arg7 = %splat_3, %arg8 = %splat_5, %arg9 = %16, %arg10 = %16) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
      %51 = arith.index_cast %arg9 : i32 to index
      %c0_28 = arith.constant 0 : index
      %c0_29 = arith.constant 0 : index
      %52 = ktdp.construct_access_tile %8[%51, %c0_28, %c0_29] {access_tile_order = #map1, access_tile_set = #set5} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
      %53 = ktdp.load %52 : <64x128x1xindex> -> tensor<64x128x1xf16>
      %54 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%53, %29 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %in_40: f16, %out: f16):
        %77 = arith.mulf %in, %in_40 : f16
        %78 = arith.addf %out, %77 : f16
        linalg.yield %78 : f16
      } -> tensor<64x64xf16>
      %cst_30 = arith.constant 0xFC00 : f16
      %splat_31 = tensor.splat %cst_30 : tensor<64xf16>
      %55 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%54 : tensor<64x64xf16>) outs(%splat_31 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %77 = arith.maxnumf %in, %out : f16
        linalg.yield %77 : f16
      } -> tensor<64xf16>
      %56 = arith.mulf %55, %splat_1 : tensor<64xf16>
      %57 = arith.maxnumf %arg8, %56 : tensor<64xf16>
      %58 = arith.mulf %54, %splat : tensor<64x64xf16>
      %expanded_32 = tensor.expand_shape %57 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_33 = tensor.collapse_shape %expanded_32 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %59 = tensor.empty() : tensor<64x64xf16>
      %60 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_33 : tensor<64xf16>) outs(%59 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<64x64xf16>
      %61 = arith.subf %58, %60 : tensor<64x64xf16>
      %62 = math.exp2 %61 : tensor<64x64xf16>
      %63 = arith.subf %arg8, %57 : tensor<64xf16>
      %64 = math.exp2 %63 : tensor<64xf16>
      %cst_34 = arith.constant 0.000000e+00 : f16
      %splat_35 = tensor.splat %cst_34 : tensor<64xf16>
      %65 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%62 : tensor<64x64xf16>) outs(%splat_35 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %77 = arith.addf %in, %out : f16
        linalg.yield %77 : f16
      } -> tensor<64xf16>
      %expanded_36 = tensor.expand_shape %64 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_37 = tensor.collapse_shape %expanded_36 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %66 = tensor.empty() : tensor<128x64xf16>
      %67 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_37 : tensor<64xf16>) outs(%66 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<128x64xf16>
      %68 = arith.mulf %arg6, %67 : tensor<128x64xf16>
      %69 = arith.index_cast %arg10 : i32 to index
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %70 = ktdp.construct_access_tile %9[%c0_38, %69, %c0_39] {access_tile_order = #map1, access_tile_set = #set6} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
      %71 = ktdp.load %70 : <128x64x1xindex> -> tensor<128x64x1xf16>
      %72 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71, %62 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%68 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %in_40: f16, %out: f16):
        %77 = arith.mulf %in, %in_40 : f16
        %78 = arith.addf %out, %77 : f16
        linalg.yield %78 : f16
      } -> tensor<128x64xf16>
      %73 = arith.mulf %arg7, %64 : tensor<64xf16>
      %74 = arith.addf %73, %65 : tensor<64xf16>
      %75 = arith.addi %arg9, %c64_i32 : i32
      %76 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %72, %74, %57, %75, %76 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
    }
    %31 = arith.muli %2, %c64_i32 : i32
    %32 = arith.addi %2, %c1_i32 : i32
    %33 = arith.muli %32, %c64_i32 : i32
    %34 = arith.addi %16, %31 : i32
    %35:5 = scf.for %arg5 = %31 to %33 step %c64_i32 iter_args(%arg6 = %30#0, %arg7 = %30#1, %arg8 = %30#2, %arg9 = %34, %arg10 = %34) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
      %51 = arith.index_cast %arg9 : i32 to index
      %c0_28 = arith.constant 0 : index
      %c0_29 = arith.constant 0 : index
      %52 = ktdp.construct_access_tile %8[%51, %c0_28, %c0_29] {access_tile_order = #map1, access_tile_set = #set5} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
      %53 = ktdp.load %52 : <64x128x1xindex> -> tensor<64x128x1xf16>
      %54 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%53, %29 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %in_41: f16, %out: f16):
        %79 = arith.mulf %in, %in_41 : f16
        %80 = arith.addf %out, %79 : f16
        linalg.yield %80 : f16
      } -> tensor<64x64xf16>
      %c0_30 = arith.constant 0 : index
      %55 = ktdp.construct_access_tile %11[%c0, %c0_30] {access_tile_order = #map, access_tile_set = #set3} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
      %56 = ktdp.load %55 : <64x64xindex> -> tensor<64x64xf16>
      %57 = arith.mulf %54, %splat : tensor<64x64xf16>
      %58 = arith.addf %57, %56 : tensor<64x64xf16>
      %cst_31 = arith.constant 0xFC00 : f16
      %splat_32 = tensor.splat %cst_31 : tensor<64xf16>
      %59 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%58 : tensor<64x64xf16>) outs(%splat_32 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %79 = arith.maxnumf %in, %out : f16
        linalg.yield %79 : f16
      } -> tensor<64xf16>
      %60 = arith.maxnumf %arg8, %59 : tensor<64xf16>
      %expanded_33 = tensor.expand_shape %60 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_34 = tensor.collapse_shape %expanded_33 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %61 = tensor.empty() : tensor<64x64xf16>
      %62 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_34 : tensor<64xf16>) outs(%61 : tensor<64x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<64x64xf16>
      %63 = arith.subf %58, %62 : tensor<64x64xf16>
      %64 = math.exp2 %63 : tensor<64x64xf16>
      %65 = arith.subf %arg8, %60 : tensor<64xf16>
      %66 = math.exp2 %65 : tensor<64xf16>
      %cst_35 = arith.constant 0.000000e+00 : f16
      %splat_36 = tensor.splat %cst_35 : tensor<64xf16>
      %67 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction", "parallel"]} ins(%64 : tensor<64x64xf16>) outs(%splat_36 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %79 = arith.addf %in, %out : f16
        linalg.yield %79 : f16
      } -> tensor<64xf16>
      %expanded_37 = tensor.expand_shape %66 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
      %collapsed_38 = tensor.collapse_shape %expanded_37 [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
      %68 = tensor.empty() : tensor<128x64xf16>
      %69 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_38 : tensor<64xf16>) outs(%68 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<128x64xf16>
      %70 = arith.mulf %arg6, %69 : tensor<128x64xf16>
      %71 = arith.index_cast %arg10 : i32 to index
      %c0_39 = arith.constant 0 : index
      %c0_40 = arith.constant 0 : index
      %72 = ktdp.construct_access_tile %9[%c0_39, %71, %c0_40] {access_tile_order = #map1, access_tile_set = #set6} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
      %73 = ktdp.load %72 : <128x64x1xindex> -> tensor<128x64x1xf16>
      %74 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73, %64 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%70 : tensor<128x64xf16>) {
      ^bb0(%in: f16, %in_41: f16, %out: f16):
        %79 = arith.mulf %in, %in_41 : f16
        %80 = arith.addf %out, %79 : f16
        linalg.yield %80 : f16
      } -> tensor<128x64xf16>
      %75 = arith.mulf %arg7, %66 : tensor<64xf16>
      %76 = arith.addf %75, %67 : tensor<64xf16>
      %77 = arith.addi %arg9, %c64_i32 : i32
      %78 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %74, %76, %60, %77, %78 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
    }
    %expanded = tensor.expand_shape %35#1 [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
    %36 = tensor.empty() : tensor<128x64xf16>
    %37 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed : tensor<64xf16>) outs(%36 : tensor<128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<128x64xf16>
    %38 = arith.divf %35#0, %37 : tensor<128x64xf16>
    %c2_17 = arith.constant 2 : index
    %c0_18 = arith.constant 0 : index
    %c2_19 = arith.constant 2 : index
    %39 = arith.divui %0, %c2_19 : index
    %c4_20 = arith.constant 4 : index
    %40 = arith.divui %39, %c4_20 : index
    %c512_21 = arith.constant 512 : index
    %41 = arith.muli %40, %c512_21 : index
    %c2_22 = arith.constant 2 : index
    %42 = arith.divui %0, %c2_22 : index
    %c4_23 = arith.constant 4 : index
    %43 = arith.remui %42, %c4_23 : index
    %c128_24 = arith.constant 128 : index
    %44 = arith.muli %43, %c128_24 : index
    %45 = arith.addi %41, %44 : index
    %c2_25 = arith.constant 2 : index
    %46 = arith.remui %0, %c2_25 : index
    %c64_26 = arith.constant 64 : index
    %47 = arith.muli %46, %c64_26 : index
    %48 = arith.addi %45, %47 : index
    %c2_27 = arith.constant 2 : index
    %49 = arith.muli %48, %c2_27 : index
    %50 = ktdp.construct_access_tile %10[%49, %c0_18] {access_tile_order = #map, access_tile_set = #set4} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
    ktdp.store %38, %50 : tensor<128x64xf16>, <128x64xindex>
    return
  }
}

