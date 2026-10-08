#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#set5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
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
      %15 = arith.muli %5, %c512_i32 : i32
      %16 = arith.muli %6, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.muli %2, %c64_i32 : i32
      %19 = arith.addi %17, %18 : i32
      %20 = arith.muli %5, %c256_i32 : i32
      %21 = arith.divsi %6, %c2_i32 : i32
      %22 = arith.muli %21, %c128_i32 : i32
      %23 = arith.addi %20, %22 : i32
      %24 = arith.index_cast %19 : i32 to index
      %c2_10 = arith.constant 2 : index
      %25 = arith.muli %24, %c2_10 : index
      %c0_11 = arith.constant 0 : index
      %26 = ktdp.construct_access_tile %8[%25, %c0_11] {access_tile_order = #map, access_tile_set = #set3} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
      %27 = ktdp.load %26 : <128x64xindex> -> tensor<128x64xf16>
      %28:5 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %splat_9, %arg8 = %splat_3, %arg9 = %splat_5, %arg10 = %23, %arg11 = %23) -> (tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32)  : i32 {
        %35 = arith.index_cast %arg10 : i32 to index
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %36 = ktdp.construct_access_tile %10[%35, %c0_14, %c0_15] {access_tile_order = #map1, access_tile_set = #set4} : memref<256x128x64xf16> -> !ktdp.access_tile<64x128x1xindex>
        %37 = ktdp.load %36 : <64x128x1xindex> -> tensor<64x128x1xf16>
        %38 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%37, %27 : tensor<64x128x1xf16>, tensor<128x64xf16>) outs(%splat_7 : tensor<64x64xf16>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %61 = arith.mulf %in, %in_18 : f16
          %62 = arith.addf %out, %61 : f16
          linalg.yield %62 : f16
        } -> tensor<64x64xf16>
        %39 = "tt.reduce"(%38) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %61 = arith.maxnumf %arg12, %arg13 : f16
          tt.reduce.return %61 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %40 = arith.mulf %39, %splat_1 : tensor<64xf16>
        %41 = arith.maxnumf %arg9, %40 : tensor<64xf16>
        %42 = arith.mulf %38, %splat : tensor<64x64xf16>
        %43 = tt.expand_dims %41 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %44 = tt.broadcast %43 : tensor<1x64xf16> -> tensor<64x64xf16>
        %45 = arith.subf %42, %44 : tensor<64x64xf16>
        %46 = math.exp2 %45 : tensor<64x64xf16>
        %47 = arith.subf %arg9, %41 : tensor<64xf16>
        %48 = math.exp2 %47 : tensor<64xf16>
        %49 = "tt.reduce"(%46) <{axis = 0 : i32}> ({
        ^bb0(%arg12: f16, %arg13: f16):
          %61 = arith.addf %arg12, %arg13 : f16
          tt.reduce.return %61 : f16
        }) : (tensor<64x64xf16>) -> tensor<64xf16>
        %50 = tt.expand_dims %48 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
        %51 = tt.broadcast %50 : tensor<1x64xf16> -> tensor<128x64xf16>
        %52 = arith.mulf %arg7, %51 : tensor<128x64xf16>
        %53 = arith.index_cast %arg11 : i32 to index
        %c0_16 = arith.constant 0 : index
        %c0_17 = arith.constant 0 : index
        %54 = ktdp.construct_access_tile %12[%c0_16, %53, %c0_17] {access_tile_order = #map1, access_tile_set = #set5} : memref<128x256x64xf16> -> !ktdp.access_tile<128x64x1xindex>
        %55 = ktdp.load %54 : <128x64x1xindex> -> tensor<128x64x1xf16>
        %56 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55, %46 : tensor<128x64x1xf16>, tensor<64x64xf16>) outs(%52 : tensor<128x64xf16>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %61 = arith.mulf %in, %in_18 : f16
          %62 = arith.addf %out, %61 : f16
          linalg.yield %62 : f16
        } -> tensor<128x64xf16>
        %57 = arith.mulf %arg8, %48 : tensor<64xf16>
        %58 = arith.addf %57, %49 : tensor<64xf16>
        %59 = arith.addi %arg10, %c64_i32 : i32
        %60 = arith.addi %arg11, %c64_i32 : i32
        scf.yield %56, %58, %41, %59, %60 : tensor<128x64xf16>, tensor<64xf16>, tensor<64xf16>, i32, i32
      }
      %29 = tt.expand_dims %28#1 {axis = 0 : i32} : tensor<64xf16> -> tensor<1x64xf16>
      %30 = tt.broadcast %29 : tensor<1x64xf16> -> tensor<128x64xf16>
      %31 = arith.divf %28#0, %30 : tensor<128x64xf16>
      %32 = arith.index_cast %19 : i32 to index
      %c2_12 = arith.constant 2 : index
      %33 = arith.muli %32, %c2_12 : index
      %c0_13 = arith.constant 0 : index
      %34 = ktdp.construct_access_tile %14[%33, %c0_13] {access_tile_order = #map, access_tile_set = #set3} : memref<1024x64xf16> -> !ktdp.access_tile<128x64xindex>
      ktdp.store %31, %34 : tensor<128x64xf16>, <128x64xindex>
    }
    tt.return
  }
}

