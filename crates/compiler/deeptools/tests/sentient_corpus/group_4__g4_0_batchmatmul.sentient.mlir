IR Dump After DataflowToSentientLoweringPass (dcc-dataflow-to-sentient) //----- //
module {
  func.func @dataflowProgram() attributes {grid = [1]} {
    %0 = sentient.scalar_constant {value = 10240 : si64} : index
    %1 = sentient.scalar_constant {value = 0 : si64} : index
    %2 = sentient.scalar_constant {value = 2048 : si64} : index
    %3 = sentient.scalar_constant {value = 132608 : si64} : index
    %4 = sentient.scalar_constant {value = 263168 : si64} : index
    %5 = sentient.scalar_constant {value = 263680 : si64} : index
    %6 = sentient.scalar_constant {value = 25600 : si64} : index
    %7 = sentient.scalar_constant {value = 393216 : si64} : index
    %8 = sentient.scalar_constant {value = 524288 : si64} : index
    %9 = sentient.scalar_constant {value = 749568 : si64} : index
    %10 = sentient.scalar_constant {value = 782336 : si64} : index
    %11 = sentient.scalar_constant {value = 4 : si64} : index
    %12 = sentient.scalar_constant {value = 0 : si64} : index
    %13 = sentient.scalar_constant {value = 512 : si64} : index
    %14 = sentient.scalar_constant {value = 2048 : si64} : index
    %15 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
    %16 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
    %17 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
    %18 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
    %19 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
    %20 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
    dataflow.program_unit iter_arg : %arg0 -> (%20) : {
      %src_res, %dst_res = sentient.load_and_store src(%15), dst(%16), src_mutable_addr(%1), src_immutable_addr(%6), src_inc(%14), dst_mutable_addr(%1), dst_immutable_addr(%1), dst_inc(%14) {burst_size = 32 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_0, %dst_res_1 = sentient.load_and_store src(%15), dst(%16), src_mutable_addr(%1), src_immutable_addr(%7), src_inc(%13), dst_mutable_addr(%1), dst_immutable_addr(%2), dst_inc(%13) {burst_size = 8 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_2, %dst_res_3 = sentient.load_and_store src(%15), dst(%16), src_mutable_addr(%1), src_immutable_addr(%8), src_inc(%13), dst_mutable_addr(%1), dst_immutable_addr(%3), dst_inc(%13) {burst_size = 8 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_4, %dst_res_5 = sentient.load_and_store src(%15), dst(%16), src_mutable_addr(%1), src_immutable_addr(%9), src_inc(%13), dst_mutable_addr(%1), dst_immutable_addr(%4), dst_inc(%13) {burst_size = 8 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
      %src_res_6, %dst_res_7 = sentient.load_and_store src(%15), dst(%16), src_mutable_addr(%1), src_immutable_addr(%10), src_inc(%13), dst_mutable_addr(%1), dst_immutable_addr(%5), dst_inc(%13) {burst_size = 8 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, regIndices = [-1 : i32, -1 : i32], regLocales = [#sentient<reg_type unknown>, #sentient<reg_type unknown>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
    }
    dataflow.program_unit %17 : {
      sentient.for %arg0 = %11 {programHeader = [], regIndices = [], regLocales = []}{
        %21 = sentient.load_and_send mutable_addr(%1), immutable_addr(%12), increment(%12), consumer(%18)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %22 = sentient.load_and_send mutable_addr(%2), immutable_addr(%12), increment(%12), consumer(%18)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %23 = sentient.load_and_send mutable_addr(%3), immutable_addr(%12), increment(%12), consumer(%18)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %24 = sentient.load_and_send mutable_addr(%4), immutable_addr(%12), increment(%12), consumer(%18)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %25 = sentient.load_and_send mutable_addr(%5), immutable_addr(%12), increment(%12), consumer(%18)  {burst_size = 0 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      }
    }
    dataflow.program_unit iter_arg : %arg0 -> (%18) {precision = "fp16"} : {
      sentient.for %arg1 = %11 {programHeader = [], regIndices = [], regLocales = []}{
        sentient.vector_mac mask(%12) {ComputePrecision = #sentient<precision fp16>, DataTransferOnly = true, ResultForwarding = [], ResultPrecision = #sentient<precision none>, fold_mode = #sentient<fold_mode fold_A>, mode = #sentient<fma_mode_op fused_mul_add>, opA = #sentient<compute_port lx>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 2 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, opC = #sentient<compute_port lx>, opCDataID = 3 : si32, opCForwarding = [], opCPortID = -1 : si32, opCPrecision = #sentient<precision fp16>, operandSegmentSizes = array<i32: 1, 0>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrOpC = false, unrollIncrResult = false, xrfReadIncr = 0 : i32, xrfWriteIncr = 0 : i32} : index
        sentient.vector_binary mask(%12) {ComputePrecision = #sentient<precision fp16>, ResultForwarding = [#sentient<compute_port lx>], ResultPrecision = #sentient<precision fp16>, binaryOp = #sentient<binary_operator mul>, fold_mode = #sentient<fold_mode fold_A>, opA = #sentient<compute_port latch>, opADataID = 0 : si32, opAForwarding = [], opAPortID = -1 : si32, opAPrecision = #sentient<precision fp16>, opB = #sentient<compute_port lx>, opBDataID = 1 : si32, opBForwarding = [], opBPortID = -1 : si32, opBPrecision = #sentient<precision fp16>, unrollFactor = #sentient<unroll_factor x1>, unrollIncrLogicalResult = false, unrollIncrOpA = false, unrollIncrOpB = false, unrollIncrResult = false} : index
      }
    }
    dataflow.program_unit %19 : {
      sentient.for %arg0 = %11 {programHeader = [], regIndices = [], regLocales = []}{
        %21 = sentient.receive_and_store mutable_addr(%0), immutable_addr(%12), increment(%12), producer(%18) {burst_size = 0 : i32, coalesce = false, element_size = 16 : i32, interleaved_group = 0 : i32, permute = false, regIndex = -1 : i32, regLocale = #sentient<reg_type unknown>, stride = 1 : i32, subword_length = 1 : i32, total_elements = 64 : i32} : index, index, index, index : index
      }
    }
    return
  }
}


