IR Dump After DataflowToSentientLoweringPass (dcc-dataflow-to-sentient) //----- //
module {
  func.func @dataflowProgram() attributes {grid = [1]} {
    %0 = sentient.scalar_constant {value = 4096 : si64} : index
    %1 = sentient.scalar_constant {value = 0 : si64} : index
    %2 = sentient.scalar_constant {value = 2048 : si64} : index
    %3 = sentient.scalar_constant {value = 4196352 : si64} : index
    %4 = sentient.scalar_constant {value = 3150848 : si64} : index
    %5 = sentient.scalar_constant {value = 5248000 : si64} : index
    %6 = sentient.scalar_constant {value = 0 : si64} : index
    %7 = sentient.scalar_constant {value = 2048 : si64} : index
    %8 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
    %9 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
    %10 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
    %11 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
    %12 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
    %13 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
    dataflow.program_unit iter_arg : %arg0 -> (%13) : {
      %src_res, %dst_res = sentient.load_and_store src(%8), dst(%9), src_mutable_addr(%1), src_immutable_addr(%2), src_inc(%7), dst_mutable_addr(%1), dst_immutable_addr(%1), dst_inc(%7) {burst_size = 32 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_0, %dst_res_1 = sentient.load_and_store src(%8), dst(%9), src_mutable_addr(%1), src_immutable_addr(%4), src_inc(%7), dst_mutable_addr(%1), dst_immutable_addr(%2), dst_inc(%7) {burst_size = 32 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_2, %dst_res_3 = sentient.load_and_store src(%8), dst(%9), src_mutable_addr(%1), src_immutable_addr(%5), src_inc(%7), dst_mutable_addr(%1), dst_immutable_addr(%3), dst_inc(%7) {burst_size = 32 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
    }
    dataflow.program_unit %10 : {
      %14 = sentient.load_and_send mutable_addr(%1), immutable_addr(%6), increment(%6), consumer(%11)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %15 = sentient.load_and_send mutable_addr(%2), immutable_addr(%6), increment(%6), consumer(%11)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %16 = sentient.load_and_send mutable_addr(%3), immutable_addr(%6), increment(%6), consumer(%11)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
    }
    dataflow.program_unit iter_arg : %arg0 -> (%11) {precision = "fp16"} : {
      sentient.vector_mac mask(%6) {ComputePrecision = #sentient<precision fp16>, DataTransferOnly = true, ResultForwarding = [], ResultPrecision = #sentient<precision none>, fold_mode = #sentient<fold_mode fold_A>, mode = #sentient<fma_mode_op fused_mul_add>, opA = #sentient<compute_port lx>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 2 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, opC = #sentient<compute_port lx>, opCDataID = 3 : si32, opCForwarding = [], opCPortID = -1 : si32, opCPrecision = #sentient<precision fp16>, operandSegmentSizes = array<i32: 1, 0>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrOpC = false, unrollIncrResult = false, xrfReadIncr = 0 : i32, xrfWriteIncr = 0 : i32} : index
      sentient.vector_binary mask(%6) {ComputePrecision = #sentient<precision fp16>, ResultForwarding = [#sentient<compute_port lx>], ResultPrecision = #sentient<precision fp16>, binaryOp = #sentient<binary_operator mul>, fold_mode = #sentient<fold_mode fold_A>, opA = #sentient<compute_port latch>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 1 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrLogicalResult = false, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrResult = false} : index
    }
    dataflow.program_unit %12 : {
      %14 = sentient.receive_and_store mutable_addr(%0), immutable_addr(%6), increment(%6), producer(%11) {burst_size = 0 : i32, coalesce = false, element_size = 16 : i32, interleaved_group = 0 : i32, permute = false, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, stride = 1 : i32, subword_length = 1 : i32, total_elements = 64 : i32} : index, index, index, index : index
    }
    return
  }
}


Program verification failed for core 0 node default_prog_name
Error message: Register initialization out of boundary:
l3lu : LBR2 : 65568

