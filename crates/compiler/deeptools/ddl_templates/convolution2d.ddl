//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// dimensions..
%krd:3 = ddl.dimension{} : index, index, index // kernel reuse dimension -- j, i, mb
%ki, %kj = ddl.dimension{dim_property = "window"} : index, index // kernel window dims ki, kj
%in, %out = ddl.dimension{} : index, index // channels 
%zf:2 = ddl.dimension{dim_property="pad_front"} : index, index // padding dimensions
%zb:2 = ddl.dimension{dim_property="pad_back"} : index, index // padding dimensions
%stride:2 = ddl.dimension{dim_property="stride"} : index, index // stride
%dilation:2 = ddl.dimension{dim_property="dilation"} : index, index // dilation
%padvalid:2 = ddl.dimension{dim_property="pad_valid"} : index, index // pad valid before front and back paddings
%krdpad0 = ddl.padded_dimension(primary=%krd#0, padding=[%zf#0, %zb#0, %padvalid#0], window=[%kj, %stride#0, %dilation#0]) // could be c
%krdpad1 = ddl.padded_dimension(primary=%krd#1, padding=[%zf#1, %zb#1, %padvalid#1], window=[%ki, %stride#1, %dilation#1]) // could be r


// layouts..
%slice_layout_input = ddl.layout(%in,%krdpad0) {is_order_fixed=true} 
%slice_layout_input_16bit = ddl.layout(%in) {is_order_fixed=true} 
%stick_layout_input = ddl.layout(%in) {is_order_fixed=true} 
%global_layout_input = ddl.layout(%in, %krdpad0, %krdpad1, %krd#2) {}//in, c, r, mb

%slice_layout_kernel = ddl.layout(%in,%out) {is_order_fixed=true} 
%slice_layout_kernel_16bit = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_kernel = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_kernel = ddl.layout(%ki, %kj, %in, %out) {}

%slice_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_output = ddl.layout(%out, %krd#0, %krd#1, %krd#2) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_int8 = ddl.type {data_type="SENINT8"}
%type_fp8 = ddl.type {data_type="SEN143_FP8"}
%type_int4 = ddl.type {data_type="SENINT4"}
%type_int24 = ddl.type {data_type="SENINT24"}

// tensors.. 
%inptensor_int8 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_int8]) : index
%kertensor_int8 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_int8]) : index
%inptensor_fp16 = ddl.tensor(%slice_layout_input_16bit, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%kertensor_fp16 = ddl.tensor(%slice_layout_kernel_16bit, %stick_layout_kernel, %global_layout_kernel, [%type_fp16]) : index
%inptensor_fp8 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_fp8]) : index
%kertensor_fp8 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_fp8]) : index
%inptensor_int4 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_int4]) : index
%kertensor_int4 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_int4]) : index
%outtensor, %psum, %bnA, %bnB, %bnA2, %bnB2, %bias, %resadd = ddl.tensor(%slice_layout_output, %stick_layout_output, %global_layout_output, [%type_fp16]) : index, index, index, index, index, index, index, index
%ptsum_fp = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%ptsum_int = ddl.internal_tensor(%outtensor, [%type_int24]) : index
%pesum = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%leakyrelu_intermediate = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

// operation..
%conv2d_int8_op = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="conv2dint8", required=false}
%conv2d_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %kertensor_fp16], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="conv2d", required=false}
%conv2d_fp8_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8, %kertensor_fp8], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="conv2dfp8", required=false}
%conv2d_int4_op = ddl.operation_bind([%type_int4], [%inptensor_int4, %kertensor_int4], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="conv2dint4", required=false}

%fp_conv2d = ddl.condition_or(%conv2d_fp16_op, %conv2d_fp8_op)
%int_conv2d = ddl.condition_or(%conv2d_int8_op, %conv2d_int4_op)

%psum_op = ddl.operation_bind([], [%psum, %outtensor], [%outtensor]) {opFuncName="add", required=false}
%bn_op = ddl.operation_bind([], [%outtensor, %bnA, %bnB], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%bn2_op = ddl.operation_bind([], [%outtensor, %bnA2, %bnB2], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%bias_op = ddl.operation_bind([], [%outtensor, %bias], [%outtensor]) {opFuncName="biasadd", required=false}
%relu_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="relufwd", required=false}
%relu6_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="relu6fwd", required=false}
%leakyrelu_op = ddl.operation_bind([], [%outtensor], [%outtensor], [%leakyrelu_intermediate]) {opFuncName="leakyrelufwd", required=false}
%stradd_op = ddl.operation_bind([], [%outtensor, %resadd], [%outtensor]) {opFuncName="stridedadd", required=false}

// constraints.. 
// no need to add stick/slice "size" constraint when there is one  dimension in stick
ddl.constraint(%inptensor_int8, %inptensor_fp8) {property = "slice", dim_idx = 0, cmp = "equal", value = 8}
ddl.constraint(%inptensor_int4) {property = "slice", dim_idx = 0, cmp = "equal", value = 16}
ddl.constraint(%inptensor_int8, %inptensor_fp8, %inptensor_int4) {property = "slice", dim_idx = 1, cmp = "equal", value = 2}
ddl.constraint(%kertensor_int8, %kertensor_fp8) {property = "slice", dim_idx = 0, cmp = "equal", value = 2}
ddl.constraint(%kertensor_int4) {property = "slice", dim_idx = 0, cmp = "equal", value = 4}
ddl.constraint(%kertensor_int8, %kertensor_fp8, %kertensor_int4) {property = "slice", dim_idx = 1, cmp = "equal", value = 8}
ddl.constraint(%conv2d_int8_op, %conv2d_fp16_op, %conv2d_fp8_op, %conv2d_int4_op) {min_num_valid = 1, max_num_valid = 1}
ddl.constraint(%relu_op, %relu6_op, %leakyrelu_op) {max_num_valid = 1}
ddl.constraint(%conv2d_int8_op, %conv2d_fp16_op, %conv2d_fp8_op, %conv2d_int4_op,
               %psum_op, %bn_op, %bn2_op, %stradd_op, %bias_op, %relu_op, %relu6_op, %leakyrelu_op) {relative_op_order=true}
ddl.constraint() {min_num_cores = 1}

// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}
%leak_const = ddl.define_constant(%type_fp16){value=[0x3733], name="leakconst"}
%clip_const = ddl.define_constant(%type_fp16){value=[0x4300], name="clipconst"}

// alias input, kernel, ptsum tensor
%inptensor = ddl.alias_one_tensor_of(%inptensor_int8, %inptensor_fp16, %inptensor_fp8, %inptensor_int4)
%kertensor = ddl.alias_one_tensor_of(%kertensor_int8, %kertensor_fp16, %kertensor_fp8, %kertensor_int4)
%ptsum = ddl.alias_one_tensor_of(%ptsum_fp, %ptsum_int)

// lx space allocation -- psum and ptsum does not enter lx..
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"}
%kertensor_lx_allocation = ddl.get_external_data_transfer_allocation (%kertensor) {memory="lx", data_connect="l3_lx_kernel"}
%bnA_lx_allocation = ddl.get_external_data_transfer_allocation (%bnA) {memory="lx", data_connect="l3_lx_bnA"}
%bnB_lx_allocation = ddl.get_external_data_transfer_allocation (%bnB) {memory="lx", data_connect="l3_lx_bnB"}
%bnA2_lx_allocation = ddl.get_external_data_transfer_allocation (%bnA2) {memory="lx", data_connect="l3_lx_bnA2"}
%bnB2_lx_allocation = ddl.get_external_data_transfer_allocation (%bnB2) {memory="lx", data_connect="l3_lx_bnB2"}
%bias_lx_allocation = ddl.get_external_data_transfer_allocation (%bias) {memory="lx", data_connect="l3_lx_bias"}
%resadd_lx_allocation = ddl.get_external_data_transfer_allocation (%resadd) {memory="lx", data_connect="l3_lx_resadd"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_output"}
%kertensor_xrf_ext_allocation = ddl.get_external_data_transfer_allocation (%kertensor) {memory="ptxrf", data_connect="xrf_kernel"}

// ddl.constraint(%kertensor_lx_allocation, %kertensor_xrf_ext_allocation) {min_num_valid = 1, max_num_valid = 1}  // re-enable after above-lx scheduler is implemented

ddl.dataflow {
  // datastages
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %blk_datastage = ddl.datastage {strategy="maximize"}
  %blk_rows_iter_datastage = ddl.datastage {strategy="minimize"}
  %blk_kij_datastage = ddl.datastage {strategy="minimize"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %accum2_datastage = ddl.datastage {strategy="minimize"}
  %input_stick_accum_datastage = ddl.datastage {strategy="minimize"} // represent the IN number per stick
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  // datastage constraints
  ddl.datastage_constraint(%bottom_datastage, %outtensor, %krd#0) {values=["1"]}
  ddl.datastage_constraint(%blk_kij_datastage, %kertensor, %ki) {values=["1"]}
  ddl.datastage_constraint(%blk_kij_datastage, %kertensor, %kj) {values=["1"]}
  // a full input slice
  ddl.datastage_constraint(%input_stick_accum_datastage, %inptensor, %in) {values=["0.125"]}
  ddl.if(%fp_conv2d) {
    ddl.if(%conv2d_fp16_op) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["8"]}
      ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {values = ["1","2","4"]}
    } 
    ddl.if(%conv2d_fp8_op) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["4"]}
      ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#0) {values = ["1","2","4"]}
      ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#1, %krd#2) {max="1"}
    }
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["1"]}
  }
  ddl.if(%int_conv2d) {
    ddl.if(%conv2d_int8_op) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["4"]}
    } 
    ddl.if(%conv2d_int4_op) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["2"]}
    }
    //ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {max="2"}
    ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#0) {max="2"}
    ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#1, %krd#2) {max="1"}
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["2"]}
  }
  ddl.if(%conv2d_fp8_op) {
    ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %in) {values=["4"]}  // "accum4" in fp8 because of break of IN coordinates within slice
  } else {
    ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %in) {values=["2"]}
  }

  %leak_sfp_allocation = ddl.allocate(%leak_const) {memory="sfplrf"} 
  %leak_sfp = ddl.unit(%leak_const, %leak_sfp_allocation) {unit="sfp", data_connect= "leak_sfp_lrf"}
  ddl.if(%leakyrelu_op){
    %src_leak_const = ddl.unit(%leak_const) {unit="constant", data_connect= "leak_const_connect"} 
    ddl.data_transfer(%src_leak_const, [%leak_sfp]) {}
  }
  %clip_sfp_allocation = ddl.allocate(%clip_const) {memory="sfplrf"} 
  %clip_sfp = ddl.unit(%clip_const, %clip_sfp_allocation) {unit="sfp", data_connect= "clip_sfp_lrf"}
  ddl.if(%relu6_op){
    %src_clip_const = ddl.unit(%clip_const) {unit="constant", data_connect= "clip_const_connect"} 
    ddl.data_transfer(%src_clip_const, [%clip_sfp]) {}
  }

  ddl.loop (%d_datastage, %b_datastage, %in, %out, %krd#0, %krd#1, %krd#2, %ki, %kj){label="chunk_loop"} {  
    %cond_first_chunk_in_loop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_first_chunk_ki_loop = ddl.condition(%ki){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_first_chunk_kj_loop = ddl.condition(%kj){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_last_chunk_in_loop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_ki_loop = ddl.condition(%ki){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_kj_loop = ddl.condition(%kj){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_accum_loops = ddl.condition_and(%cond_last_chunk_in_loop, %cond_last_chunk_ki_loop,
                                                %cond_last_chunk_kj_loop)
    ddl.loop (%b_datastage, %bottom_datastage, %out) {} {
      
      // block load bn / biasAdd param to sfp
      %bnA_sfp_allocation = ddl.allocate(%bnA) {memory="sfplrf"}
      %bnB_sfp_allocation = ddl.allocate(%bnB) {memory="sfplrf"}
      %bnA2_sfp_allocation = ddl.allocate(%bnA2) {memory="sfplrf"}
      %bnB2_sfp_allocation = ddl.allocate(%bnB2) {memory="sfplrf"}
      %bias_sfp_allocation = ddl.allocate(%bias) {memory="sfplrf"}
      ddl.if(%cond_last_chunk_accum_loops) {
        ddl.if(%bn_op) {
          %src_bnA_lxsfp = ddl.unit(%bnA, %bnA_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA"}
          %dst_bnA_lxsfp = ddl.unit(%bnA, %bnA_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnA"} 
          ddl.data_transfer(%src_bnA_lxsfp, [%dst_bnA_lxsfp]) {}
          %src_bnB_lxsfp = ddl.unit(%bnB, %bnB_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB"}
          %dst_bnB_lxsfp = ddl.unit(%bnB, %bnB_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnB"} 
          ddl.data_transfer(%src_bnB_lxsfp, [%dst_bnB_lxsfp]) {}
        }
        ddl.if(%bn2_op) {
          %src_bnA2_lxsfp = ddl.unit(%bnA2, %bnA2_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA2"}
          %dst_bnA2_lxsfp = ddl.unit(%bnA2, %bnA2_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnA2"} 
          ddl.data_transfer(%src_bnA2_lxsfp, [%dst_bnA2_lxsfp]) {}
          %src_bnB2_lxsfp = ddl.unit(%bnB2, %bnB2_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB2"}
          %dst_bnB2_lxsfp = ddl.unit(%bnB2, %bnB2_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnB2"} 
          ddl.data_transfer(%src_bnB2_lxsfp, [%dst_bnB2_lxsfp]) {}
        }
        ddl.if(%bias_op) {
          %src_bias_lxsfp = ddl.unit(%bias, %bias_lx_allocation) {unit="lxlu", data_connect="l3_lx_bias"}
          %dst_bias_lxsfp = ddl.unit(%bias, %bias_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bias"} 
          ddl.data_transfer(%src_bias_lxsfp, [%dst_bias_lxsfp]) {}
        }
      }

      ddl.loop (%b_datastage, %blk_datastage, %in, %ki, %kj) {label="blkload_loop_in"} {
        %cond_first_blkload_in_loop = ddl.condition(%in){loop_label="blkload_loop_in", condition="eq", value_expr="first"}
        %cond_first_blkload_ki_loop = ddl.condition(%ki){loop_label="blkload_loop_in", condition="eq", value_expr="first"}
        %cond_first_blkload_kj_loop = ddl.condition(%kj){loop_label="blkload_loop_in", condition="eq", value_expr="first"}
        %cond_last_blkload_in_loop = ddl.condition(%in){loop_label="blkload_loop_in", condition="eq", value_expr="last"}
        %cond_last_blkload_ki_loop = ddl.condition(%ki){loop_label="blkload_loop_in", condition="eq", value_expr="last"}
        %cond_last_blkload_kj_loop = ddl.condition(%kj){loop_label="blkload_loop_in", condition="eq", value_expr="last"}
        %cond_first_accum_loops = ddl.condition_and(%cond_first_chunk_in_loop, %cond_first_chunk_ki_loop,
                                                    %cond_first_chunk_kj_loop, %cond_first_blkload_in_loop,
                                                    %cond_first_blkload_ki_loop, %cond_first_blkload_kj_loop)
        %cond_last_accum_loops = ddl.condition_and(%cond_last_chunk_accum_loops, %cond_last_blkload_in_loop,
                                                    %cond_last_blkload_ki_loop, %cond_last_blkload_kj_loop)
        %cond_not_first_accum_loops = ddl.condition_not(%cond_first_accum_loops)
        // block load into lx-XRF
        // load kernel in an order to ensure loading enough IN at innermost loop to cover a stick of input
        %kertensor_xrf_allocation = ddl.allocate(%kertensor, %kertensor_xrf_ext_allocation) {memory="ptxrf"}
        ddl.force_innermost_dimensions(%kertensor_xrf_allocation, %input_stick_accum_datastage, %in)
        ddl.if (%kertensor_lx_allocation) {
          ddl.loop (%blk_datastage, %input_stick_accum_datastage, %in) {} {
            ddl.loop (%blk_datastage, %blk_kij_datastage, %ki) {} {
              ddl.loop (%blk_datastage, %blk_kij_datastage, %kj) {} {
                ddl.loop (%input_stick_accum_datastage, %blk_rows_iter_datastage, %in) {} {
                  %src_ker_lxpt = ddl.unit(%kertensor, %kertensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_kernel"}
                  %dst_ker_lxpt = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", data_connect="xrf_kernel"} 
                  ddl.data_transfer(%src_ker_lxpt, [%dst_ker_lxpt]) {}
                }
              }
            }
          }
        }
        ddl.if(%cond_not_first_accum_loops) {
          ddl.sync {units=["lxsu"], is_receive=false, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
          ddl.sync {units=["lxlu"], is_receive=true, signal_name="input-lxsu-lxlu-sync", separate_corelets=true} 
        }               
        //ddl.loop (%b_datastage, %bottom_datastage, %krd#1, %krd#2) {} {
          ddl.loop (%b_datastage, %interleave_datastage, %krd#0, %krd#1, %krd#2) {} {
            // load input to lx-l0
            %inptensor_l0_allocation = ddl.allocate(%inptensor, [%krdpad0, %krdpad1]) {memory="l0", num_buffers=-1:si64, 
                              padding_type=["padded_wzeropad", "padded_wzeropad"]}
            ddl.implicit_sync(%inptensor_l0_allocation)
            %src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
            %dst_inp_lxl0 = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0su", data_connect="l0_input"} 
            ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0])

            // load output to lx-pe-fifo
            %outtensor_pe_allocation = ddl.allocate(%outtensor) {memory="pelrf"}
            %pe_out_lrf = ddl.unit(%outtensor, %outtensor_pe_allocation) {unit="pe", data_connect="lxpe_output"}
            ddl.if(%cond_not_first_accum_loops) {
              %src_out_lxpe = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxlu", data_connect="lxsu_output"}
              ddl.data_transfer(%src_out_lxpe, [%pe_out_lrf]) {}
            }
            %pesum_allocation = ddl.allocate(%pesum) {memory="pelrf"}
            %pe_fma_lrf = ddl.unit(%pesum, %pesum_allocation) {unit="pe", data_connect="pe_outtensor"}
            %pe_fma_dst00_sfp = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_outtensor"}

            // for l0lu read, the loop order of in, ki, kj has to be in this way to minimize register modification overhead.
            ddl.loop (%blk_datastage, %input_stick_accum_datastage, %in) {label="inp_l0_fetch_in_loop"} {
              %cond_first_inpfetch_in_loop = ddl.condition(%in){loop_label="inp_l0_fetch_in_loop", condition="eq", value_expr="first"}
              %cond_last_inpfetch_in_loop = ddl.condition(%in){loop_label="inp_l0_fetch_in_loop", condition="eq", value_expr="last"}
              ddl.loop (%blk_datastage, %blk_kij_datastage, %ki) {label="inp_l0_fetch_ki_loop"} {
                %cond_first_inpfetch_ki_loop = ddl.condition(%ki){loop_label="inp_l0_fetch_ki_loop", condition="eq", value_expr="first"}
                %cond_last_inpfetch_ki_loop = ddl.condition(%ki){loop_label="inp_l0_fetch_ki_loop", condition="eq", value_expr="last"}
                ddl.loop (%blk_datastage, %blk_kij_datastage, %kj) {label="inp_l0_fetch_kj_loop"} {
                  %cond_first_inpfetch_kj_loop = ddl.condition(%kj){loop_label="inp_l0_fetch_kj_loop", condition="eq", value_expr="first"}
                  %cond_last_inpfetch_kj_loop = ddl.condition(%kj){loop_label="inp_l0_fetch_kj_loop", condition="eq", value_expr="last"}

                  ddl.loop (%input_stick_accum_datastage, %accum2_datastage, %in) {label="blkaccum2_loop"} {
                    %cond_first_blkaccum2loop = ddl.condition(%in){loop_label="blkaccum2_loop", condition="eq", value_expr="first"}
                    %cond_last_blkaccum2loop = ddl.condition(%in){loop_label="blkaccum2_loop", condition="eq", value_expr="last"}
                    %cond_first_blk_accum = ddl.condition_and(%cond_first_blkaccum2loop, %cond_first_inpfetch_in_loop,
                                                          %cond_first_inpfetch_ki_loop, %cond_first_inpfetch_kj_loop)
                    %cond_last_blk_accum = ddl.condition_and(%cond_last_blkaccum2loop, %cond_last_inpfetch_in_loop,
                                                          %cond_last_inpfetch_ki_loop, %cond_last_inpfetch_kj_loop)
                    ddl.loop (%accum2_datastage, %bottom_datastage, %in) {label="accum2bottom_loop"} {
                      %cond_first_accum2bottomloop = ddl.condition(%in){loop_label="accum2bottom_loop", condition="eq", value_expr="first"}
                      %cond_last_accum2bottomloop = ddl.condition(%in){loop_label="accum2bottom_loop", condition="eq", value_expr="last"}
                      %ptsum_arf_allocation = ddl.allocate(%ptsum) {memory="ptarf"}
                      ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                        // load l0-pt
                        %src_inp_l0pt = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0lu", data_connect="l0_input"}
                        %dst_inp_l0pt = ddl.unit(%inptensor) {unit="pt", data_connect="pt_input"} 
                        ddl.data_transfer(%src_inp_l0pt, [%dst_inp_l0pt]) {}

                        // do IMA/FMA 
                        %pt_src00 = ddl.unit(%inptensor) {unit="l0lu", data_connect="pt_input"}
                        %pt_src01 = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", data_connect="xrf_kernel"}
                        %pt_src02_arf = ddl.unit(%ptsum, %ptsum_arf_allocation) {unit="pt", data_connect="arf_ptsum"}
                        %pt_src02_north = ddl.unit(%ptsum) {unit="ptnorth", data_connect="arf_ptsum"}
                        %pt_dst00_arf = ddl.unit(%ptsum, %ptsum_arf_allocation) {unit="pt", data_connect="arf_ptsum"}
                        %pt_dst00_south = ddl.unit(%ptsum) {unit="ptsouth", data_connect="arf_ptsum"}
                        
                        ddl.if(%cond_first_accum2bottomloop) {
                          // flavor1: 0 + store to arf..
                          ddl.compute([%pt_src00, %pt_src01, %zero_const], [%pt_dst00_arf]) {computetype="MACC", unit="ptrow0"}
                          // flavor2: north + store to arf..
                          ddl.compute([%pt_src00, %pt_src01, %pt_src02_north], [%pt_dst00_arf]) {computetype="MACC", unit="ptrow1-7"}
                        } else {
                          ddl.if(%cond_last_accum2bottomloop) {
                            // flavor4: arf + send to south
                            ddl.compute([%pt_src00, %pt_src01, %pt_src02_arf], [%pt_dst00_south]) {computetype="MACC", unit="pt"}
                          } else {
                            // flavor3: arf + store to arf 
                            ddl.compute([%pt_src00, %pt_src01, %pt_src02_arf], [%pt_dst00_arf]) {computetype="MACC", unit="pt"}
                          }
                        }
                        // pt-pe-ptsum
                        ddl.if(%cond_last_accum2bottomloop) {
                          %src_ptsum_ptpe = ddl.unit(%ptsum) {unit="ptrow7", data_connect="arf_ptsum"}
                          %dst_ptsum_ptpe = ddl.unit(%ptsum) {unit="pe", data_connect="pe_ptsum"}
                          ddl.data_transfer(%src_ptsum_ptpe, [%dst_ptsum_ptpe]) {}                            
                        }
                      }
                    }
                    //ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                      // chunk accumulation in PE
                      %pe_fma_src00 = ddl.unit(%ptsum) {unit="pt", data_connect="pe_ptsum"}
                      %cond_last_blk_accum_no_lxlu_ptsum = ddl.condition_and(%cond_last_blk_accum, %cond_first_accum_loops)
                      ddl.if(%cond_first_blk_accum) {
                        ddl.if(%cond_last_blk_accum_no_lxlu_ptsum) {
                          ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                            // chunk accumulation in PE: 0 + ptsum => sfp
                            ddl.compute([%pe_fma_src00, %one_const, %zero_const], [%pe_fma_dst00_sfp]) {computetype="FMA16", unit="pe"}
                          }
                        } else {
                          ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                            // chunk accumulation in PE: 0 + ptsum => lrf
                            ddl.compute([%pe_fma_src00, %one_const, %zero_const], [%pe_fma_lrf]) {computetype="FMA16", unit="pe"}
                          }
                        }
                      } else {
                        ddl.if(%cond_last_blk_accum_no_lxlu_ptsum) {
                          ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                            // chunk accumulation in PE: lrf + ptsum => sfp
                            ddl.compute([%pe_fma_src00, %one_const, %pe_fma_lrf], [%pe_fma_dst00_sfp]) {computetype="FMA16", unit="pe"}
                          }
                        } else {
                          ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                            // chunk accumulation in PE: lrf + ptsum => lrf
                            ddl.compute([%pe_fma_src00, %one_const, %pe_fma_lrf], [%pe_fma_lrf]) {computetype="FMA16", unit="pe"}
                          }
                        }
                      }                            
                    //}
                  }
                }
              }
            }
            ddl.if(%cond_not_first_accum_loops) {
              ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                // chunk accumulation in PE: lrf + lrf => sfp
                ddl.compute([%pe_out_lrf, %one_const, %pe_fma_lrf], [%pe_fma_dst00_sfp]) {computetype="FMA16", unit="pe"}
              }
            }                       
            // pe-sfp-output
            %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
            %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_outtensor"}
            ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}     

            // Aux ops in SFP 
            %outtensor_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
            %is_any_relu = ddl.condition_or(%relu_op, %relu6_op, %leakyrelu_op)
            %is_any_aux_op = ddl.condition_or(%psum_op, %stradd_op, %bn_op, %bn2_op, %bias_op, %is_any_relu)
            %cond_aux_ops_iter = ddl.condition_and(%cond_last_accum_loops, %is_any_aux_op)
            %is_no_aux_op = ddl.condition_not(%is_any_aux_op)
            %is_psum_not_last = ddl.condition_or(%stradd_op, %bn_op, %bn2_op, %bias_op, %is_any_relu)
            %is_bn_not_first = ddl.condition_or(%psum_op)
            %is_bn_not_last = ddl.condition_or(%bn2_op, %bias_op, %stradd_op, %is_any_relu)
            %is_bn2_not_first = ddl.condition_or(%psum_op, %bn_op)
            %is_bn2_not_last = ddl.condition_or(%bias_op, %stradd_op, %is_any_relu)
            %is_bias_not_first = ddl.condition_or(%psum_op, %bn_op, %bn2_op)
            %is_bias_not_last = ddl.condition_or(%stradd_op, %is_any_relu)
            %is_stradd_not_first = ddl.condition_or(%psum_op, %bn_op, %bn2_op, %bias_op)
            %is_stradd_first = ddl.condition_not(%is_stradd_not_first)
            %is_stradd_not_last = ddl.condition_or(%is_any_relu)
            %is_relu_not_first = ddl.condition_or(%psum_op, %stradd_op, %bn_op, %bn2_op, %bias_op)
            %is_relu_first = ddl.condition_not(%is_relu_not_first)
            ddl.if (%cond_aux_ops_iter) {
              // psum
              %psum_start, %psum_end, %next_core, %prev_core = ddl.core_to_core_communication(%in)
              ddl.if (%psum_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_psum_src00 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_psum_src02 = ddl.unit(%outtensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
                  %sfp_psum_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_psum_dst00_sfpring = ddl.unit(%outtensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
                  %sfp_psum_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  ddl.if (%psum_start) {
                    // psum: 0 + pe -> sfpring
                    ddl.compute([%sfp_psum_src00, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                  } else {
                    ddl.if (%psum_end) {
                      ddl.if(%is_psum_not_last) {
                        // psum: sfp-ring + pe -> lrf
                        ddl.compute([%sfp_psum_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_lrf]) {computetype="FMA16", unit="sfp"}                                     
                      } else {
                        // psum: sfp-ring + pe -> lxsu
                        ddl.compute([%sfp_psum_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                      }
                    } else { // middle cores
                      // psum: sfp-ring + pe -> sfpring
                      ddl.compute([%sfp_psum_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                    }
                  }
                }
              }
              // bn
              ddl.if (%bn_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_bn_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_bn_src00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_bn_src01 = ddl.unit(%bnA, %bnA_sfp_allocation) {unit="sfp", data_connect="sfp_bnA"}
                  %sfp_bn_src02 = ddl.unit(%bnB, %bnB_sfp_allocation) {unit="sfp", data_connect="sfp_bnB"}
                  %sfp_bn_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_bn_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  ddl.if(%is_bn_not_first) {
                    ddl.if(%is_bn_not_last) {
                      // bn: lrf * lrf + lrf -> lrf
                      ddl.compute([%sfp_bn_src00_lrf, %sfp_bn_src01, %sfp_bn_src02], [%sfp_bn_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // bn: lrf * lrf + lrf -> lxsu
                      ddl.compute([%sfp_bn_src00_lrf, %sfp_bn_src01, %sfp_bn_src02], [%sfp_bn_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                    }
                  } else {
                    ddl.if(%is_bn_not_last) {
                      // bn: lrf * pe + lrf -> lrf
                      ddl.compute([%sfp_bn_src00_pe, %sfp_bn_src01, %sfp_bn_src02], [%sfp_bn_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // bn: lrf * pe + lrf -> lxsu
                      ddl.compute([%sfp_bn_src00_pe, %sfp_bn_src01, %sfp_bn_src02], [%sfp_bn_dst00_lxsu]) {computetype="FMA16", unit="sfp"}                                        
                    }
                  }
                }
              }
              ddl.if (%bn2_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_bn2_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_bn2_src00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_bn2_src01 = ddl.unit(%bnA2, %bnA2_sfp_allocation) {unit="sfp", data_connect="sfp_bnA2"}
                  %sfp_bn2_src02 = ddl.unit(%bnB2, %bnB2_sfp_allocation) {unit="sfp", data_connect="sfp_bnB2"}
                  %sfp_bn2_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_bn2_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  ddl.if(%is_bn2_not_first) {
                    ddl.if(%is_bn2_not_last) {
                      // bn: lrf * lrf + lrf -> lrf
                      ddl.compute([%sfp_bn2_src00_lrf, %sfp_bn2_src01, %sfp_bn2_src02], [%sfp_bn2_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // bn: lrf * lrf + lrf -> lxsu
                      ddl.compute([%sfp_bn2_src00_lrf, %sfp_bn2_src01, %sfp_bn2_src02], [%sfp_bn2_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                    }
                  } else {
                    ddl.if(%is_bn2_not_last) {
                      // bn: lrf * pe + lrf -> lrf
                      ddl.compute([%sfp_bn2_src00_pe, %sfp_bn2_src01, %sfp_bn2_src02], [%sfp_bn2_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // bn: lrf * pe + lrf -> lxsu
                      ddl.compute([%sfp_bn2_src00_pe, %sfp_bn2_src01, %sfp_bn2_src02], [%sfp_bn2_dst00_lxsu]) {computetype="FMA16", unit="sfp"}                                        
                    }
                  }
                }
              }
              // biasadd
              ddl.if (%bias_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_biasadd_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_biasadd_src00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_biasadd_src02 = ddl.unit(%bias, %bias_sfp_allocation) {unit="sfp", data_connect="sfp_bias"}
                  %sfp_biasadd_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_biasadd_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  ddl.if(%is_bias_not_first) {
                    ddl.if(%is_bias_not_last) {
                      // biasadd: 1.0 * lrf + lrf -> lrf
                      ddl.compute([%sfp_biasadd_src00_lrf, %one_const, %sfp_biasadd_src02], [%sfp_biasadd_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // biasadd: 1.0 * lrf + lrf -> lxsu
                      ddl.compute([%sfp_biasadd_src00_lrf, %one_const, %sfp_biasadd_src02], [%sfp_biasadd_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                    }
                  } else {
                    ddl.if(%is_bias_not_last) {
                      // biasadd: 1.0 * pe + lrf -> lrf
                      ddl.compute([%sfp_biasadd_src00_pe, %one_const, %sfp_biasadd_src02], [%sfp_biasadd_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                    } else {
                      // biasadd: 1.0 * pe + lrf -> lxsu
                      ddl.compute([%sfp_biasadd_src00_pe, %one_const, %sfp_biasadd_src02], [%sfp_biasadd_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                    }
                  }
                }
              }
              // strided add
              ddl.if (%stradd_op) {
                // load stridedAdd input from lxlu to pe-sfp-fifo
                %src_resadd_lxsfp = ddl.unit(%resadd, %resadd_lx_allocation) {unit="lxlu", data_connect="l3_lx_resadd"}
                %dst_resadd_lxsfp = ddl.unit(%resadd) {unit="sfp", vias=["pe"], data_connect="sfp_resadd"} 
                ddl.data_transfer(%src_resadd_lxsfp, [%dst_resadd_lxsfp]) {}

                ddl.if(%is_stradd_first) {
                  ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                    // stridedaddprep (fetch chunk accumulation first): 0 + pe -> lrf
                    %sfp_straddprep_src02 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                    %sfp_straddprep_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                    ddl.compute([%zero_const, %one_const, %sfp_straddprep_src02], [%sfp_straddprep_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                  }
                }
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_stradd_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_stradd_src00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_stradd_src02_pe = ddl.unit(%resadd) {unit="pe", data_connect="sfp_resadd"}
                  %sfp_stradd_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  %sfp_stradd_dst00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  ddl.if(%is_stradd_not_last) {
                    // stridedadd: pe + lrf -> lrf
                    ddl.compute([%sfp_stradd_src00_lrf, %one_const, %sfp_stradd_src02_pe], [%sfp_stradd_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                  } else {
                    // stridedadd: pe + lrf -> lxsu
                    ddl.compute([%sfp_stradd_src00_lrf, %one_const, %sfp_stradd_src02_pe], [%sfp_stradd_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                  }
                }
              }
              // relu
              ddl.if (%relu_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_relu_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  %sfp_relu_src00_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                  %sfp_relu_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  ddl.if(%is_relu_not_first) {
                    // relu: lrf -> lxsu
                    ddl.compute([%sfp_relu_src00_lrf, %zero_const], [%sfp_relu_dst00_lxsu]) {computetype="FMAX", unit="sfp"}
                  } else {
                    // relu: pe -> lxsu
                    ddl.compute([%sfp_relu_src00_pe, %zero_const], [%sfp_relu_dst00_lxsu]) {computetype="FMAX", unit="sfp"}
                  }
                }
              }
              // relu6
              ddl.if (%relu6_op) {
                %sfp_relu_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_relu_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                  ddl.if(%is_relu_not_first) {
                    // relu: lrf -> lrf
                    ddl.compute([%sfp_relu_lrf, %zero_const], [%sfp_relu_lrf]) {computetype="FMAX", unit="sfp"}
                  } else {
                    // relu: pe -> lrf
                    ddl.compute([%sfp_relu_src00_pe, %zero_const], [%sfp_relu_lrf]) {computetype="FMAX", unit="sfp"}
                  }
                }
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_relu_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  // relu: lrf -> lxsu
                  ddl.compute([%sfp_relu_lrf, %clip_sfp], [%sfp_relu_dst00_lxsu]) {computetype="FMIN", unit="sfp"}
                }
              }
              // leakyrelu
              ddl.if (%leakyrelu_op) {
                %leaky_inter_sfp_allocation = ddl.allocate(%leakyrelu_intermediate) {memory="sfplrf"} 
                %sfp_leaky_inter_lrf = ddl.unit(%leakyrelu_intermediate, %leaky_inter_sfp_allocation) {unit="sfp", data_connect="sfp_leak_inter_lrf"}
                %sfp_partialsum_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
                ddl.if(%is_relu_first) {
                  ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                    %sfp_relu_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                    // store partial sum: pe -> lrf
                    ddl.compute([%sfp_relu_src00_pe, %one_const], [%sfp_partialsum_lrf]) {computetype="FMUL", unit="sfp"}
                  }
                }
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  // leaky val: lrf -> lrf
                  ddl.compute([%sfp_partialsum_lrf, %leak_sfp], [%sfp_leaky_inter_lrf]) {computetype="FMUL", unit="sfp"}
                }
                ddl.loop (%interleave_datastage, %bottom_datastage, %krd#0, %krd#1, %krd#2) {} {
                  %sfp_relu_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                  // leakyrelu: lrf -> lxsu
                  ddl.compute([%sfp_partialsum_lrf, %sfp_leaky_inter_lrf], [%sfp_relu_dst00_lxsu]) {computetype="FMAX", unit="sfp"}
                }
              }
              ddl.if(%psum_end) {
                // store output tensor sfp-lx
                %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_lxsu_outtensor"}
                %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
              }
            } else {
              // store output tensor pe-sfp-lx
              %noaux_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
              %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", vias=["sfp"], data_connect="lxsu_output"}
              ddl.data_transfer(%noaux_src00_pe, [%dst_out_sfplx]) {}
            }
          }
        //}
      }
    }
  }
}

ddl.transformations {
}

}
