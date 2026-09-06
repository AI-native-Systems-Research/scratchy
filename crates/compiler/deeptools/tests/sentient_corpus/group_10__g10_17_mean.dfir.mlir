func.func @dataflowProgram() attributes {grid = [1]} {

  %0 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
  %1 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
  %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-constant-CL0", type = "constant"} : index
  %3 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-pe-CL0", type = "pe"} : index
  %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
  %5 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
  %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
  %7 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
  dataflow.program_unit %7 : {
    %8 = arith.constant 3348352 : index
    %9 = dataflow.get_logical_memory_view %0, %8 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %10 = arith.constant 0 : index
    %11 = dataflow.get_logical_memory_view %1, %10 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    agen.composite_load_and_store src:%9[0, 0] dst:%11[0, 0]
     time_symbols(), load_iv(%12:vector<64xf16>)
     {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>}
    {
      agen.yield
    } : memref<1x2048xf16>, memref<1x2048xf16>
  }
  dataflow.program_unit %5 : {
    %13 = arith.constant 0 : index
    %14 = dataflow.get_logical_memory_view %1, %13 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %15 = agen.vector_load %14[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x2048xf16>, vector<64xf16>
    dataflow.send %4, %15 : vector<64xf16>
  }
  dataflow.program_unit %4 {precision = "fp16"} : {
    %16 = dataflow.receive %5 : vector<64xf16>
    %17 = dataflow.receive %5 : vector<64xf16>
    %18 = vectorchain.binary %16, %17 {binary_op = #vectorchain<binary_operator add>, op_specific_map = affine_map<(d0) -> (d0)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>
    dataflow.send %6, %18 : vector<64xf16>
  }
  dataflow.program_unit %6 : {
    %19 = arith.constant 3350400 : index
    %20 = dataflow.get_logical_memory_view %1, %19 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
    %21 = dataflow.receive %4 : vector<64xf16>
    agen.vector_store %21, %20[0, 0] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
  }
  return
}
