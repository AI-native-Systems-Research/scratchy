IR Dump After DataflowToSentientLoweringPass (dcc-dataflow-to-sentient) //----- //
module {
  func.func @dataflowProgram() attributes {grid = [1]} {
    %0 = sentient.scalar_constant {value = 3307520 : si64} : index
    %1 = sentient.scalar_constant {value = 0 : si64} : index
    %2 = sentient.scalar_constant {value = 3305472 : si64} : index
    %3 = sentient.scalar_constant {value = 0 : si64} : index
    %4 = sentient.scalar_constant {value = 2048 : si64} : index
    %5 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
    %6 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
    %7 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
    %8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
    %9 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
    %10 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
    dataflow.program_unit iter_arg : %arg0 -> (%10) : {
      %src_res, %dst_res = sentient.load_and_store src(%5), dst(%6), src_mutable_addr(%1), src_immutable_addr(%2), src_inc(%4), dst_mutable_addr(%1), dst_immutable_addr(%1), dst_inc(%4) {burst_size = 32 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
    }
    dataflow.program_unit %8 : {
      %11 = sentient.load_and_send mutable_addr(%1), immutable_addr(%3), increment(%3), consumer(%7)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
    }
    dataflow.program_unit iter_arg : %arg0 -> (%7) {precision = "fp16"} : {
      sentient.vector_mac mask(%3) {ComputePrecision = #sentient<precision fp16>, DataTransferOnly = true, ResultForwarding = [], ResultPrecision = #sentient<precision none>, fold_mode = #sentient<fold_mode fold_A>, mode = #sentient<fma_mode_op fused_mul_add>, opA = #sentient<compute_port lx>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 2 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, opC = #sentient<compute_port lx>, opCDataID = 3 : si32, opCForwarding = [], opCPortID = -1 : si32, opCPrecision = #sentient<precision fp16>, operandSegmentSizes = array<i32: 1, 0>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrOpC = false, unrollIncrResult = false, xrfReadIncr = 0 : i32, xrfWriteIncr = 0 : i32} : index
      sentient.vector_binary mask(%3) {ComputePrecision = #sentient<precision fp16>, ResultForwarding = [#sentient<compute_port lx>], ResultPrecision = #sentient<precision fp16>, binaryOp = #sentient<binary_operator add>, fold_mode = #sentient<fold_mode fold_A>, opA = #sentient<compute_port latch>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 1 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrLogicalResult = false, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrResult = false} : index
    }
    dataflow.program_unit %9 : {
      %11 = sentient.receive_and_store mutable_addr(%0), immutable_addr(%3), increment(%3), producer(%7) {burst_size = 0 : i32, coalesce = false, element_size = 16 : i32, interleaved_group = 0 : i32, permute = false, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, stride = 1 : i32, subword_length = 1 : i32, total_elements = 64 : i32} : index, index, index, index : index
    }
    return
  }
}


Program verification failed for core 0 node default_prog_name
Error message: Register initialization out of boundary:
lxsu0 : LRF0 : 6615040

