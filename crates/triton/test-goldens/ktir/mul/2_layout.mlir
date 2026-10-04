#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  tt.func public @mul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) attributes {noinline = false} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c64_i32 = arith.constant 64 : i32
    %0 = ktdp.get_compute_tile_id : index
    scf.for %arg4 = %0 to %c16 step %c32 {
      ktdf.corelet_plan pattern = "single_corelet" {
        ktdf.corelet 0 {data_bounds = [0, 1]}
      }
      %1 = arith.index_cast %arg4 : index to i32
      %2 = arith.muli %1, %c64_i32 : i32
      %3 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
      %4 = arith.index_cast %arg3 : i32 to index
      %5 = ktdp.construct_memory_view %3, sizes: [%4], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
      %6 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
      %7 = arith.index_cast %arg3 : i32 to index
      %8 = ktdp.construct_memory_view %6, sizes: [%7], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
      %9 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
      %10 = arith.index_cast %arg3 : i32 to index
      %11 = ktdp.construct_memory_view %9, sizes: [%10], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
      %12 = arith.index_cast %2 : i32 to index
      %13 = ktdp.construct_access_tile %5[%12] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
      %14 = ktdp.load %13 : <64xindex> -> tensor<64xf16>
      %15 = arith.index_cast %2 : i32 to index
      %16 = ktdp.construct_access_tile %8[%15] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
      %17 = ktdp.load %16 : <64xindex> -> tensor<64xf16>
      %18 = arith.mulf %14, %17 : tensor<64xf16>
      %19 = arith.index_cast %2 : i32 to index
      %20 = ktdp.construct_access_tile %11[%19] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
      ktdp.store %18, %20 : tensor<64xf16>, <64xindex>
    }
    tt.return
  }
}

