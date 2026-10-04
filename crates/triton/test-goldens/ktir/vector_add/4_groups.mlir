#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module attributes {spyre.grid_dead_chain_ops_erased = 3 : i64, spyre.grid_i32_ops_left = 0 : i64, spyre.grid_index_chains_rebuilt = 3 : i64} {
  func.func @vector_add_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: i32) attributes {grid = [16 : index], spyre.folded_grid_loop = {num_cores = 32 : index, work_items = 16 : index}} {
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = ktdp.construct_memory_view %arg0, sizes: [%1], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %c64 = arith.constant 64 : index
    %c64_0 = arith.constant 64 : index
    %3 = arith.muli %0, %c64_0 : index
    %4 = ktdp.construct_access_tile %2[%3] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    %5 = ktdp.load %4 : <64xindex> -> tensor<64xf16>
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = ktdp.construct_memory_view %arg1, sizes: [%6], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %c64_1 = arith.constant 64 : index
    %c64_2 = arith.constant 64 : index
    %8 = arith.muli %0, %c64_2 : index
    %9 = ktdp.construct_access_tile %7[%8] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    %10 = ktdp.load %9 : <64xindex> -> tensor<64xf16>
    %11 = ktdp.get_compute_tile_id : index
    %12 = tensor.empty() : tensor<64xf16>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %10 : tensor<64xf16>, tensor<64xf16>) outs(%12 : tensor<64xf16>) {
    ^bb0(%in: f16, %in_5: f16, %out: f16):
      %18 = arith.addf %in, %in_5 : f16
      linalg.yield %18 : f16
    } -> tensor<64xf16>
    %14 = arith.index_cast %arg3 : i32 to index
    %15 = ktdp.construct_memory_view %arg2, sizes: [%14], strides: [1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?xf16>
    %c64_3 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %16 = arith.muli %11, %c64_4 : index
    %17 = ktdp.construct_access_tile %15[%16] {access_tile_order = #map, access_tile_set = #set1} : memref<?xf16> -> !ktdp.access_tile<64xindex>
    ktdp.store %13, %17 : tensor<64xf16>, <64xindex>
    return
  }
}

