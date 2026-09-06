func.func @dataflowProgram() attributes {grid = [1]} {

  %0 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
  %1 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
  %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
  %3 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-constant-CL0", type = "constant"} : index
  %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
  %5 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
  %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
  dataflow.program_unit %6 : {
    %7 = arith.constant 3346112 : index
    %8 = dataflow.get_logical_memory_view %0, %7 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
    %9 = arith.constant 0 : index
    %10 = dataflow.get_logical_memory_view %1, %9 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
    agen.composite_load_and_store src:%8[0, 0] dst:%10[0, 0]
     time_symbols(), load_iv(%11:vector<64xf16>)
     {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, 0)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, 0)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 == 0)>}
    {
      agen.yield
    } : memref<1x64xf16>, memref<1x64xf16>
  }
  dataflow.program_unit %4 : {
    %12 = arith.constant 0 : index
    %13 = dataflow.get_logical_memory_view %1, %12 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
    %14 = agen.vector_load %13[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
    dataflow.send %2, %14 : vector<64xf16>
  }
  dataflow.program_unit %2 {precision = "fp16"} : {
    %15 = dataflow.receive %4 : vector<64xf16>
    %16 = dataflow.receive %4 : vector<64xf16>
    %17 = vectorchain.binary %15, %16 {binary_op = #vectorchain<binary_operator add>, op_specific_map = affine_map<(d0) -> (d0)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>
    dataflow.send %5, %17 : vector<64xf16>
  }
  dataflow.program_unit %5 : {
    %18 = arith.constant 3346176 : index
    %19 = dataflow.get_logical_memory_view %1, %18 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
    %20 = dataflow.receive %2 : vector<64xf16>
    agen.vector_store %20, %19[0, 0] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
  }
  return
}
