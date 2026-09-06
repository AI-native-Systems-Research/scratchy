func.func @dataflowProgram() attributes {grid = [1]} {

  %0 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
  %1 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
  %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
  %3 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow0-CL0", type = "ptrow0"} : index
  %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow1-CL0", type = "ptrow1"} : index
  %5 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow2-CL0", type = "ptrow2"} : index
  %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow3-CL0", type = "ptrow3"} : index
  %7 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow4-CL0", type = "ptrow4"} : index
  %8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow5-CL0", type = "ptrow5"} : index
  %9 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow6-CL0", type = "ptrow6"} : index
  %10 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-ptrow7-CL0", type = "ptrow7"} : index
  %11 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l0su-CL0", type = "l0su"} : index
  %12 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-pe-CL0", type = "pe"} : index
  %13 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
  %14 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l0lu-CL0", type = "l0lu"} : index
  %15 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
  %16 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
  dataflow.program_unit %16 : {
    %17 = arith.constant 0 : index
    %18 = dataflow.get_logical_memory_view %0, %17 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %19 = arith.constant 0 : index
    %20 = dataflow.get_logical_memory_view %1, %19 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    agen.composite_load_and_store src:%18[0, 0] dst:%20[0, 0]
     time_symbols(), load_iv(%21:vector<64xf16>)
     {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>}
    {
      agen.yield
    } : memref<1x2048xf16>, memref<1x2048xf16>
    %22 = arith.constant 2048 : index
    %23 = dataflow.get_logical_memory_view %0, %22 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<2048x2048xf16>
    %24 = arith.constant 2048 : index
    %25 = dataflow.get_logical_memory_view %1, %24 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<2048x2048xf16>
    agen.composite_load_and_store src:%23[0, 0] dst:%25[0, 0]
     time_symbols(), load_iv(%26:vector<64xf16>)
     {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>}
    {
      agen.yield
    } : memref<2048x2048xf16>, memref<2048x2048xf16>
    %27 = arith.constant 2099200 : index
    %28 = dataflow.get_logical_memory_view %0, %27 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %29 = arith.constant 4196352 : index
    %30 = dataflow.get_logical_memory_view %1, %29 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    agen.composite_load_and_store src:%28[0, 0] dst:%30[0, 0]
     time_symbols(), load_iv(%31:vector<64xf16>)
     {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, d0 * 64)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>}
    {
      agen.yield
    } : memref<1x2048xf16>, memref<1x2048xf16>
  }
  dataflow.program_unit %2 : {
    %32 = arith.constant 0 : index
    %33 = dataflow.get_logical_memory_view %1, %32 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %34 = arith.constant 2048 : index
    %35 = dataflow.get_logical_memory_view %1, %34 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<2048x2048xf16>
    %36 = arith.constant 4196352 : index
    %37 = dataflow.get_logical_memory_view %1, %36 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %38 = agen.vector_load %33[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x2048xf16>, vector<64xf16>
    dataflow.send %13, %38 : vector<64xf16>
    %39 = agen.vector_load %35[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<2048x2048xf16>, vector<64xf16>
    dataflow.send %13, %39 : vector<64xf16>
    %40 = agen.vector_load %37[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x2048xf16>, vector<64xf16>
    dataflow.send %13, %40 : vector<64xf16>
  }
  dataflow.program_unit %13 {precision = "fp16"} : {
    %41 = dataflow.receive %2 : vector<64xf16>
    %42 = dataflow.receive %2 : vector<64xf16>
    %43 = vectorchain.binary %41, %42 {binary_op = #vectorchain<binary_operator mul>, op_specific_map = affine_map<(d0) -> (d0)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>
    dataflow.send %15, %43 : vector<64xf16>
  }
  dataflow.program_unit %15 : {
    %44 = arith.constant 0 : index
    %45 = dataflow.get_logical_memory_view %1, %44 {layout_map = affine_map<(d0, d1) -> (d0 * 2048 + d1)>} : index, index, memref<1x2048xf16>
    %46 = dataflow.receive %13 : vector<64xf16>
    agen.vector_store %46, %45[0, 0] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x2048xf16>, vector<64xf16>
  }
  return
}
