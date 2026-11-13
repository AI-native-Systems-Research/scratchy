#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#set6 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
module {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) attributes {noinline = false} {
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
    scf.for %arg5 = %0 to %c8 step %c32 {
      ktdf.corelet_plan pattern = "independent_rows" {
        ktdf.corelet 0 {data_bounds = [0, 32]}
        ktdf.corelet 1 {data_bounds = [32, 64]}
      }
      %1 = arith.remui %arg5, %c2 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.divui %arg5, %c2 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.divsi %4, %c4_i32 : i32
      %6 = arith.remsi %4, %c4_i32 : i32
      %7 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
      %8 = ktdp.construct_memory_view %7, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
      %9 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
      %10 = ktdp.construct_memory_view %9, sizes: [256, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<256x128x64xf16>
      %11 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
      %12 = ktdp.construct_memory_view %11, sizes: [128, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<128x256x64xf16>
      %13 = builtin.unrealized_conversion_cast %arg3 : !tt.ptr<f16> to index
      %14 = ktdp.construct_memory_view %13, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
      %15 = builtin.unrealized_conversion_cast %arg4 : !tt.ptr<f16> to index
      %16 = ktdp.construct_memory_view %15, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set3, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64x64xf16>
      %17 = arith.muli %5, %c512_i32 : i32
      %18 = arith.muli %6, %c128_i32 : i32
      %19 = arith.addi %17, %18 : i32
      %20 = arith.muli %2, %c64_i32 : i32
      %21 = arith.addi %19, %20 : i32
      %22 = arith.muli %5, %c256_i32 : i32
      %23 = arith.divsi %6, %c2_i32 : i32
      %24 = arith.muli %23, %c128_i32 : i32
      %25 = arith.addi %22, %24 : i32
      %26 = arith.index_cast %21 : i32 to index
      %c2_10 = arith.constant 2 : index
      %27 = arith.muli %26, %c2_10 : index
      %c0_11 = arith.constant 0 : index
      %28 = ktdp.construct_access_tile %8[%27, %c0_11] {access_tile_order = #map, access_tile_set = #set4} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
      %29 = ktdp.load %28 : <128x64xindex> -> tensor<128x64xf16>
      %30:5 = scf.for %arg6 = %c0_i32 to %20 step %c64_i32 iter_args(%arg7 = %splat_9, %arg8 = %splat_3, %arg9 = %splat_5, %arg10 = %25, %arg11 = %25) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
        %42 = arith.index_cast %arg10 : i32 to index
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %43 = ktdp.construct_access_tile %10[%42, %c0_14, %c0_15] {access_tile_order = #map1, access_tile_set = #set5} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
        %44 = ktdp.load %43 : <64x128x1xindex> -> tensor<64x128x1xf16>
        %45 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%44, %29 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %68 = arith.mulf %in, %in_18 : f16
          %69 = arith.addf %out, %68 : f16
          linalg.yield %69 : f16
        } -> tensor<64x64xf16>
        %46 = "tt.reduce"(%45) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %68 = arith.maxnumf %arg12, %arg13 : f16
          tt.reduce.return %68 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %47 = arith.mulf %46, %splat_1 : tensor<64xf16>
        %48 = arith.maxnumf %arg9, %47 : tensor<64xf16>
        %49 = arith.mulf %45, %splat : tensor<64x64xf16>
        %50 = tt.expand_dims %48 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %51 = tt.broadcast %50 : tensor<1x64xf16> -> tensor<64x64xf16>
        %52 = arith.subf %49, %51 : tensor<64x64xf16>
        %53 = math.exp2 %52 : tensor<64x64xf16>
        %54 = arith.subf %arg9, %48 : tensor<64xf16>
        %55 = math.exp2 %54 : tensor<64xf16>
        %56 = "tt.reduce"(%53) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %68 = arith.addf %arg12, %arg13 : f16
          tt.reduce.return %68 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %57 = tt.expand_dims %55 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %58 = tt.broadcast %57 : tensor<1x64xf16> -> tensor<128x64xf16>
        %59 = arith.mulf %arg7, %58 : tensor<128x64xf16>
        %60 = arith.index_cast %arg11 : i32 to index
        %c0_16 = arith.constant 0 : index
        %c0_17 = arith.constant 0 : index
        %61 = ktdp.construct_access_tile %12[%c0_16, %60, %c0_17] {access_tile_order = #map1, access_tile_set = #set6} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
        %62 = ktdp.load %61 : <128x64x1xindex> -> tensor<128x64x1xf16>
        %63 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62, %53 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%59 : tensor<128x64xf16>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %68 = arith.mulf %in, %in_18 : f16
          %69 = arith.addf %out, %68 : f16
          linalg.yield %69 : f16
        } -> tensor<128x64xf16>
        %64 = arith.mulf %arg8, %55 : tensor<64xf16>
        %65 = arith.addf %64, %56 : tensor<64xf16>
        %66 = arith.addi %arg10, %c64_i32 : i32
        %67 = arith.addi %arg11, %c64_i32 : i32
        scf.yield %63, %65, %48, %66, %67 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
      }
      %31 = arith.muli %2, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
      %32 = arith.addi %2, %c1_i32 : i32
      %33 = arith.muli %32, %c64_i32 : i32
      %34 = arith.addi %25, %31 : i32
      %35:5 = scf.for %arg6 = %31 to %33 step %c64_i32 iter_args(%arg7 = %30#0, %arg8 = %30#1, %arg9 = %30#2, %arg10 = %34, %arg11 = %34) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
        %42 = arith.index_cast %arg10 : i32 to index
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %43 = ktdp.construct_access_tile %10[%42, %c0_14, %c0_15] {access_tile_order = #map1, access_tile_set = #set5} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
        %44 = ktdp.load %43 : <64x128x1xindex> -> tensor<64x128x1xf16>
        %45 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%44, %29 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
        ^bb0(%in: f16, %in_19: f16, %out: f16):
          %70 = arith.mulf %in, %in_19 : f16
          %71 = arith.addf %out, %70 : f16
          linalg.yield %71 : f16
        } -> tensor<64x64xf16>
        %c0_16 = arith.constant 0 : index
        %46 = ktdp.construct_access_tile %16[%c0, %c0_16] {access_tile_order = #map, access_tile_set = #set3} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
        %47 = ktdp.load %46 : <64x64xindex> -> tensor<64x64xf16>
        %48 = arith.mulf %45, %splat : tensor<64x64xf16>
        %49 = arith.addf %48, %47 : tensor<64x64xf16>
        %50 = "tt.reduce"(%49) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %70 = arith.maxnumf %arg12, %arg13 : f16
          tt.reduce.return %70 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %51 = arith.maxnumf %arg9, %50 : tensor<64xf16>
        %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %53 = tt.broadcast %52 : tensor<1x64xf16> -> tensor<64x64xf16>
        %54 = arith.subf %49, %53 : tensor<64x64xf16>
        %55 = math.exp2 %54 : tensor<64x64xf16>
        %56 = arith.subf %arg9, %51 : tensor<64xf16>
        %57 = math.exp2 %56 : tensor<64xf16>
        %58 = "tt.reduce"(%55) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %70 = arith.addf %arg12, %arg13 : f16
          tt.reduce.return %70 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %59 = tt.expand_dims %57 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %60 = tt.broadcast %59 : tensor<1x64xf16> -> tensor<128x64xf16>
        %61 = arith.mulf %arg7, %60 : tensor<128x64xf16>
        %62 = arith.index_cast %arg11 : i32 to index
        %c0_17 = arith.constant 0 : index
        %c0_18 = arith.constant 0 : index
        %63 = ktdp.construct_access_tile %12[%c0_17, %62, %c0_18] {access_tile_order = #map1, access_tile_set = #set6} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
        %64 = ktdp.load %63 : <128x64x1xindex> -> tensor<128x64x1xf16>
        %65 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%64, %55 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%61 : tensor<128x64xf16>) {
        ^bb0(%in: f16, %in_19: f16, %out: f16):
          %70 = arith.mulf %in, %in_19 : f16
          %71 = arith.addf %out, %70 : f16
          linalg.yield %71 : f16
        } -> tensor<128x64xf16>
        %66 = arith.mulf %arg8, %57 : tensor<64xf16>
        %67 = arith.addf %66, %58 : tensor<64xf16>
        %68 = arith.addi %arg10, %c64_i32 : i32
        %69 = arith.addi %arg11, %c64_i32 : i32
        scf.yield %65, %67, %51, %68, %69 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
      }
      %36 = tt.expand_dims %35#1 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
      %37 = tt.broadcast %36 : tensor<1x64xf16> -> tensor<128x64xf16>
      %38 = arith.divf %35#0, %37 : tensor<128x64xf16>
      %39 = arith.index_cast %21 : i32 to index
      %c2_12 = arith.constant 2 : index
      %40 = arith.muli %39, %c2_12 : index
      %c0_13 = arith.constant 0 : index
      %41 = ktdp.construct_access_tile %14[%40, %c0_13] {access_tile_order = #map, access_tile_set = #set4} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
      ktdp.store %38, %41 : tensor<128x64xf16>, <128x64xindex>
    }
    tt.return
  }
}

