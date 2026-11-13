#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module attributes {spyre.grid_dead_chain_ops_erased = 5 : i64, spyre.grid_i32_ops_left = 0 : i64, spyre.grid_index_chains_rebuilt = 3 : i64} {
  func.func @mul_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: i32) attributes {grid = [16 : index], spyre.folded_grid_loop = {num_cores = 32 : index, work_items = 16 : index}} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c64_i32 = arith.constant 64 : i32
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = ktdp.construct_memory_view %arg0, sizes: [%1], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = ktdp.construct_memory_view %arg1, sizes: [%3], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = ktdp.construct_memory_view %arg2, sizes: [%5], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %c64 = arith.constant 64 : index
    %7 = arith.muli %0, %c64 : index
    %8 = ktdp.construct_access_tile %2[%7] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    %9 = ktdp.load %8 : <64xindex> -> tensor<64xf16>
    %c64_0 = arith.constant 64 : index
    %10 = arith.muli %0, %c64_0 : index
    %11 = ktdp.construct_access_tile %4[%10] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    %12 = ktdp.load %11 : <64xindex> -> tensor<64xf16>
    %13 = arith.mulf %9, %12 : tensor<64xf16>
    %c64_1 = arith.constant 64 : index
    %14 = arith.muli %0, %c64_1 : index
    %15 = ktdp.construct_access_tile %6[%14] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    ktdp.store %13, %15 : tensor<64xf16>, <64xindex>
    return
  }
}

