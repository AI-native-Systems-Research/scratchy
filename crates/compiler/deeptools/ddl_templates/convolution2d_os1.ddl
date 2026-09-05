//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// dimensions..
%krd:2, %j = ddl.dimension{} : index, index, index // kernel reuse dimension -- j, i, mb
%ki, %kj = ddl.dimension{dim_property = "window"} : index, index // kernel window dims ki, kj
%in, %out = ddl.dimension{} : index, index // channels 
%zf:2 = ddl.dimension{dim_property="pad_front"} : index, index // padding dimensions
%zb:2 = ddl.dimension{dim_property="pad_back"} : index, index // padding dimensions
%stride:2 = ddl.dimension{dim_property="stride"} : index, index // stride
%dilation:2 = ddl.dimension{dim_property="dilation"} : index, index // dilation
%padvalid:2 = ddl.dimension{dim_property="pad_valid"} : index, index // pad valid before front and back paddings
%krdpad0 = ddl.padded_dimension(primary=%krd#0, padding=[%zf#0, %zb#0, %padvalid#0], window=[%ki, %stride#0, %dilation#0])
%j_pad = ddl.padded_dimension(primary=%j, padding=[%zf#1, %zb#1, %padvalid#1], window=[%kj, %stride#1, %dilation#1])


// layouts..
%slice_layout_input = ddl.layout(%in,%j_pad) {is_order_fixed=true} 
%slice_layout_input_16bit = ddl.layout(%j_pad) {is_order_fixed=true} 
%stick_layout_input = ddl.layout(%j_pad) {is_order_fixed=true} 
%global_layout_input = ddl.layout(%in, %j_pad, %krdpad0, %krd#1) {}//in, c, r, mb

%slice_layout_kernel = ddl.layout(%in,%out) {is_order_fixed=true} 
%slice_layout_kernel_16bit = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_kernel = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_kernel = ddl.layout(%ki, %kj, %in, %out) {}

%slice_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_output = ddl.layout(%out, %j, %krd#0, %krd#1) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_int8 = ddl.type {data_type="SENINT8"}
%type_int24 = ddl.type {data_type="SENINT24"}

// tensors.. 
%inptensor_int8 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_int8]) : index
%kertensor_int8 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_int8]) : index
%inptensor_fp16 = ddl.tensor(%slice_layout_input_16bit, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%inptensor_fp16_gen = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%kertensor_fp16 = ddl.tensor(%slice_layout_kernel_16bit, %stick_layout_kernel, %global_layout_kernel, [%type_fp16]) : index
%outtensor, %psum, %bnA, %bnB, %bnA2, %bnB2, %bias, %resadd = ddl.tensor(%slice_layout_output, %stick_layout_output, %global_layout_output, [%type_fp16]) : index, index, index, index, index, index, index, index
%ptsum_fp = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%ptsum_int = ddl.internal_tensor(%outtensor, [%type_int24]) : index
%leakyrelu_intermediate = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

// operation..
%conv2d_os1_int8_op_lrf = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int]) {opFuncName="conv2dint8os1", required=false}
%conv2d_os1_int8_op_xrf = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int]) {opFuncName="conv2dxrfint8os1", required=false}
%conv2d_os1_int8_op = ddl.condition_or(%conv2d_os1_int8_op_lrf, %conv2d_os1_int8_op_xrf)
%conv2d_os1_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %kertensor_fp16], [%outtensor], [%ptsum_fp]) {opFuncName="conv2dos1", required=false}
%conv2d_os1_fp16_gen_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16_gen, %kertensor_fp16], [%outtensor], [%ptsum_fp]) {opFuncName="conv2dgenos1", required=false}


%bn_op = ddl.operation_bind([], [%outtensor, %bnA, %bnB], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%bn2_op = ddl.operation_bind([], [%outtensor, %bnA2, %bnB2], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%bias_op = ddl.operation_bind([], [%outtensor, %bias], [%outtensor]) {opFuncName="biasadd", required=false}
%relu_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="relufwd", required=false}
%relu6_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="relu6fwd", required=false}
%leakyrelu_op = ddl.operation_bind([], [%outtensor], [%outtensor], [%leakyrelu_intermediate]) {opFuncName="leakyrelufwd", required=false}
%stradd_op = ddl.operation_bind([], [%outtensor, %resadd], [%outtensor]) {opFuncName="stridedadd", required=false}
%bn_or_bias_op = ddl.condition_or(%bn_op, %bias_op)

// constraints.. 
// no need to add stick/slice "size" constraint when there is one dimension in stick
ddl.constraint(%inptensor_int8) {property = "slice", dim_idx = 0, cmp = "equal", value = 4}
ddl.constraint(%inptensor_int8) {property = "slice", dim_idx = 1, cmp = "equal", value = 4}
ddl.constraint(%kertensor_int8) {property = "slice", dim_idx = 0, cmp = "equal", value = 2}
ddl.constraint(%kertensor_int8) {property = "slice", dim_idx = 1, cmp = "equal", value = 8}
// TODO: add constraint to only allow stride 1, 2, 4 (and 8 in fp16 only)
ddl.constraint(%conv2d_os1_int8_op_lrf, %conv2d_os1_int8_op_xrf, %conv2d_os1_fp16_op, %conv2d_os1_fp16_gen_op) {min_num_valid = 1, max_num_valid = 1}
ddl.constraint(%relu_op, %leakyrelu_op, %relu6_op) {max_num_valid = 1}
ddl.constraint(%conv2d_os1_int8_op_lrf, %conv2d_os1_int8_op_xrf, %conv2d_os1_fp16_op, %conv2d_os1_fp16_gen_op,
               %bn_op, %bn2_op, %stradd_op, %bias_op, %relu_op, %leakyrelu_op, %relu6_op) {relative_op_order=true}
ddl.constraint() {min_num_cores = 1}

// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}
%leak_const = ddl.define_constant(%type_fp16){value=[0x3733], name="leakconst"}
%clip_const = ddl.define_constant(%type_fp16){value=[0x4300], name="clipconst"}

// alias input, kernel, ptsum tensor
%inptensor = ddl.alias_one_tensor_of(%inptensor_int8, %inptensor_fp16, %inptensor_fp16_gen)
%kertensor = ddl.alias_one_tensor_of(%kertensor_int8, %kertensor_fp16)
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
  %blk_kij_datastage = ddl.datastage {strategy="maximize"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=false}  // j epilogue would require unbalanced split across pt rows
  // %accum2_datastage = ddl.datastage {strategy="minimize"}
  %input_stick_accum_datastage = ddl.datastage {strategy="minimize"} // represent the IN number per stick
  %bottom_datastage = ddl.datastage {strategy="minimize"}
  %interleave_datastage_pe = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage_pe = ddl.datastage {strategy="minimize"}

  // datastage constraints
  ddl.datastage_constraint(%bottom_datastage, %outtensor, %j) {values=["1"]}
  ddl.datastage_constraint(%blk_kij_datastage, %kertensor, %ki) {values=["1"]}
  ddl.datastage_constraint(%blk_kij_datastage, %kertensor, %kj) {values=["1"]}
  ddl.datastage_constraint(%blk_datastage, %kertensor, %kj) {values=["1"]}  // to avoid complication caused by lowered reuse from L0 within stick
  // a full input slice
  ddl.datastage_constraint(%input_stick_accum_datastage, %inptensor, %in) {values=["1"]}  // 1 because IN is not split across rows
  ddl.datastage_constraint(%interleave_datastage, %inptensor, %j, %stride#1) {values = ["0.125"]}
  ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %j, %krd#0, %krd#1) {values = ["1","2","4"]}
  %conv2d_os1_fp16 = ddl.condition_or(%conv2d_os1_fp16_op, %conv2d_os1_fp16_gen_op)
  ddl.if(%conv2d_os1_fp16) {
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["1"]}
  } 
  ddl.if(%conv2d_os1_int8_op) {
    // ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %j, %krd#0, %krd#1) {max="2"}
    // ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %j) {max="2"}
    // ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %krd#0, %krd#1) {max="1"}
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["2"]}
  }
  //ddl.datastage_constraint(%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {values = ["1","2","4"]}
  // ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %in) {values=["2"]}

  // Because we are reading from PT through a FIFO, we need to ensure the order of loops in PT and PE are same.
  // So we lock the datastages together.
  ddl.datastage_constraint(%interleave_datastage, %interleave_datastage_pe, %j) {values = ["8"]}  // number of rows
  // Because of the order of datastage exploration, relative constraint above does not work as intended.
  // So adding an absolute contraint below.
  ddl.datastage_constraint(%interleave_datastage_pe, %inptensor, %j, %stride#1) {values = ["0.125"]}
  ddl.datastage_constraint(%interleave_datastage, %interleave_datastage_pe, %krd#0) {values = ["1"]}
  ddl.datastage_constraint(%interleave_datastage, %interleave_datastage_pe, %krd#1) {values = ["1"]}

  %leak_pe_allocation = ddl.allocate(%leak_const) {memory="pelrf"} 
  %leak_pe = ddl.unit(%leak_const, %leak_pe_allocation) {unit="pe", data_connect= "leak_pe_lrf"}
  ddl.if(%leakyrelu_op){
    %src_leak_const = ddl.unit(%leak_const) {unit="constant", data_connect= "leak_const_connect"} 
    ddl.data_transfer(%src_leak_const, [%leak_pe]) {}
  }
  %clip_pe_allocation = ddl.allocate(%clip_const) {memory="pelrf"} 
  %clip_pe = ddl.unit(%clip_const, %clip_pe_allocation) {unit="pe", data_connect= "clip_pe_lrf"}
  ddl.if(%relu6_op){
    %src_clip_const = ddl.unit(%clip_const) {unit="constant", data_connect= "clip_const_connect"} 
    ddl.data_transfer(%src_clip_const, [%clip_pe]) {}
  }

  ddl.loop (%d_datastage, %b_datastage, %in, %out, %j, %krd#0, %krd#1, %ki, %kj){label="chunk_loop"} {  
    %cond_first_chunk_in_loop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_first_chunk_ki_loop = ddl.condition(%ki){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_first_chunk_kj_loop = ddl.condition(%kj){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_last_chunk_in_loop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_ki_loop = ddl.condition(%ki){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_kj_loop = ddl.condition(%kj){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_accum_loops = ddl.condition_and(%cond_last_chunk_ki_loop, %cond_last_chunk_kj_loop)
    ddl.loop (%b_datastage, %bottom_datastage, %out) {} {
      
      // block load bn / biasAdd param to pe
      %bnA_pe_allocation = ddl.allocate(%bnA) {memory="pelrf"}
      %bnB_pe_allocation = ddl.allocate(%bnB) {memory="pelrf"}
      %bnA2_pe_allocation = ddl.allocate(%bnA2) {memory="pelrf"}
      %bnB2_pe_allocation = ddl.allocate(%bnB2) {memory="pelrf"}
      %bias_pe_allocation = ddl.allocate(%bias) {memory="pelrf"}
      ddl.if(%cond_last_chunk_accum_loops) {
        ddl.if(%bn_op) {
          %src_bnA_lxpe = ddl.unit(%bnA, %bnA_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA"}
          %dst_bnA_lxpe = ddl.unit(%bnA, %bnA_pe_allocation) {unit="pe", data_connect="pe_bnA"} 
          ddl.data_transfer(%src_bnA_lxpe, [%dst_bnA_lxpe]) {}
          %src_bnB_lxpe = ddl.unit(%bnB, %bnB_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB"}
          %dst_bnB_lxpe = ddl.unit(%bnB, %bnB_pe_allocation) {unit="pe", data_connect="pe_bnB"} 
          ddl.data_transfer(%src_bnB_lxpe, [%dst_bnB_lxpe]) {}
        }
        ddl.if(%bn2_op) {
          %src_bnA2_lxpe = ddl.unit(%bnA2, %bnA2_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA2"}
          %dst_bnA2_lxpe = ddl.unit(%bnA2, %bnA2_pe_allocation) {unit="pe", data_connect="pe_bnA2"} 
          ddl.data_transfer(%src_bnA2_lxpe, [%dst_bnA2_lxpe]) {}
          %src_bnB2_lxpe = ddl.unit(%bnB2, %bnB2_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB2"}
          %dst_bnB2_lxpe = ddl.unit(%bnB2, %bnB2_pe_allocation) {unit="pe", data_connect="pe_bnB2"} 
          ddl.data_transfer(%src_bnB2_lxpe, [%dst_bnB2_lxpe]) {}
        }
        ddl.if(%bias_op) {
          %src_bias_lxpe = ddl.unit(%bias, %bias_lx_allocation) {unit="lxlu", data_connect="l3_lx_bias"}
          %dst_bias_lxpe = ddl.unit(%bias, %bias_pe_allocation) {unit="pe", data_connect="pe_bias"} 
          ddl.data_transfer(%src_bias_lxpe, [%dst_bias_lxpe]) {}
        }
      }

      ddl.loop (%b_datastage, %blk_datastage, %ki, %kj) {label="blkload_loop"} {
        %cond_first_blkload_ki_loop = ddl.condition(%ki){loop_label="blkload_loop", condition="eq", value_expr="first"}
        %cond_first_blkload_kj_loop = ddl.condition(%kj){loop_label="blkload_loop", condition="eq", value_expr="first"}
        %cond_last_blkload_ki_loop = ddl.condition(%ki){loop_label="blkload_loop", condition="eq", value_expr="last"}
        %cond_last_blkload_kj_loop = ddl.condition(%kj){loop_label="blkload_loop", condition="eq", value_expr="last"}
        %cond_first_accum_loops = ddl.condition_and(%cond_first_chunk_ki_loop, %cond_first_chunk_kj_loop,
                                                    %cond_first_blkload_ki_loop, %cond_first_blkload_kj_loop)
        %cond_last_accum_loops = ddl.condition_and(%cond_last_chunk_accum_loops,
                                                    %cond_last_blkload_ki_loop, %cond_last_blkload_kj_loop)
        %cond_not_first_accum_loops = ddl.condition_not(%cond_first_accum_loops)
        // block load into pt-XRF
        %kertensor_xrf_allocation = ddl.allocate(%kertensor, %kertensor_xrf_ext_allocation) {memory="ptxrf"}
        ddl.force_innermost_dimensions(%kertensor_xrf_allocation, %bottom_datastage, %in)
        ddl.if (%kertensor_lx_allocation) {
          ddl.loop (%b_datastage, %bottom_datastage, %in) {} {
            ddl.loop (%blk_datastage, %blk_kij_datastage, %ki, %kj) {} {
              %src_ker_lxpt = ddl.unit(%kertensor, %kertensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_kernel"}
              %dst_ker_lxpt = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", vias=["sfp"], data_connect="xrf_kernel"} 
              ddl.data_transfer(%src_ker_lxpt, [%dst_ker_lxpt]) {}
            }
          }
        }
        ddl.if(%cond_not_first_accum_loops) {
          ddl.sync {units=["lxsu"], is_receive=false, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
          ddl.sync {units=["lxlu"], is_receive=true, signal_name="input-lxsu-lxlu-sync", separate_corelets=true} 
        }               
        ddl.loop (%b_datastage, %interleave_datastage, %j, %krd#0, %krd#1) {} {
          // load input to lx-l0
          %inptensor_l0_allocation = ddl.allocate(%inptensor, [%j_pad, %krdpad0]) {memory="l0", num_buffers=-1:si64, 
                            padding_type=["padded_wzeropad", "padded_wzeropad"]}
          %src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
          %dst_inp_lxl0 = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0su", vias=["sfp"], data_connect="l0_input"} 
          ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0])

          ddl.sync {units=["l0lu"], is_receive=false, signal_name="input-l0lu-l0su-sync", separate_corelets=true}
          ddl.sync {units=["l0su"], is_receive=true, signal_name="input-l0lu-l0su-sync", separate_corelets=true}

          ddl.sync {units=["l0su"], is_receive=false, signal_name="input-l0su-l0lu-sync", separate_corelets=true}
          ddl.sync {units=["l0lu"], is_receive=true, signal_name="input-l0su-l0lu-sync", separate_corelets=true}

          // for l0lu read, the loop order of in, ki, kj has to be in this way to minimize register modification overhead.
          ddl.loop (%b_datastage, %input_stick_accum_datastage, %in) {label="inp_l0_fetch_in_loop"} {
            %cond_first_inpfetch_in_loop = ddl.condition(%in){loop_label="inp_l0_fetch_in_loop", condition="eq", value_expr="first"}
            %cond_last_inpfetch_in_loop = ddl.condition(%in){loop_label="inp_l0_fetch_in_loop", condition="eq", value_expr="last"}
            ddl.loop (%blk_datastage, %blk_kij_datastage, %ki) {label="inp_l0_fetch_ki_loop"} {
              %cond_first_inpfetch_ki_loop = ddl.condition(%ki){loop_label="inp_l0_fetch_ki_loop", condition="eq", value_expr="first"}
              %cond_last_inpfetch_ki_loop = ddl.condition(%ki){loop_label="inp_l0_fetch_ki_loop", condition="eq", value_expr="last"}
              ddl.loop (%input_stick_accum_datastage, %bottom_datastage, %in) {label="in_accum_loop"} {
                %cond_first_inaccumloop = ddl.condition(%in){loop_label="in_accum_loop", condition="eq", value_expr="first"}
                %cond_last_inaccumloop = ddl.condition(%in){loop_label="in_accum_loop", condition="eq", value_expr="last"}
                ddl.loop (%blk_datastage, %blk_kij_datastage, %kj) {label="inp_l0_fetch_kj_loop"} {
                  %cond_first_inpfetch_kj_loop = ddl.condition(%kj){loop_label="inp_l0_fetch_kj_loop", condition="eq", value_expr="first"}
                  %cond_last_inpfetch_kj_loop = ddl.condition(%kj){loop_label="inp_l0_fetch_kj_loop", condition="eq", value_expr="last"}

                  %cond_first_blk_accum = ddl.condition_and(%cond_first_inaccumloop, %cond_first_inpfetch_in_loop,
                                                        %cond_first_inpfetch_ki_loop, %cond_first_inpfetch_kj_loop)
                  %cond_last_blk_accum = ddl.condition_and(%cond_last_inaccumloop, %cond_last_inpfetch_in_loop,
                                                        %cond_last_inpfetch_ki_loop, %cond_last_inpfetch_kj_loop)
                  %ptsum_arf_allocation = ddl.allocate(%ptsum) {memory="ptarf"}
                  %ptsum_arf = ddl.unit(%ptsum, %ptsum_arf_allocation) {unit="pt", data_connect="arf_ptsum"}
                  ddl.loop (%interleave_datastage, %bottom_datastage, %j, %krd#0, %krd#1) {} {
                    // load l0-pt
                    %src_inp_l0pt = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0lu", data_connect="l0_input"}
                    %dst_inp_l0pt = ddl.unit(%inptensor) {unit="pt", data_connect="pt_input"} 
                    ddl.data_transfer(%src_inp_l0pt, [%dst_inp_l0pt]) {}

                    // do IMA/FMA 
                    %pt_src00 = ddl.unit(%inptensor) {unit="l0lu", data_connect="pt_input"}
                    %pt_src01 = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", data_connect="xrf_kernel"}
                    %pt_dst00_south = ddl.unit(%ptsum) {unit="ptsouth", data_connect="arf_ptsum"}
                    
                    ddl.if(%cond_first_blk_accum) {
                      // 0 + store to arf..
                      ddl.compute([%pt_src00, %pt_src01, %zero_const], [%ptsum_arf]) {computetype="MACC", unit="pt"}
                    } else {
                      // arf + store to arf 
                      ddl.compute([%pt_src00, %pt_src01, %ptsum_arf], [%ptsum_arf]) {computetype="MACC", unit="pt"}
                    }
                  }
                  ddl.if(%cond_last_blk_accum) {
                    // pt-pe-ptsum
                    %dst_ptsum_ptpe = ddl.unit(%ptsum) {unit="pe", data_connect="pe_ptsum"}
                    ddl.data_transfer(%ptsum_arf, [%dst_ptsum_ptpe]) {}                            
                  }
                }
              }
            }
          }

          ddl.loop (%interleave_datastage, %interleave_datastage_pe, %j, %krd#0, %krd#1) {} {
            // load output to lx-pe-fifo
            ddl.if(%cond_not_first_accum_loops) {
              %src_out_lxpe = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxlu", data_connect="lxsu_output"}
              %pe_lxpsum = ddl.unit(%outtensor) {unit="pe", data_connect="lxpe_output"}
              ddl.data_transfer(%src_out_lxpe, [%pe_lxpsum]) {}
            }

            // lx-ptsum accumulation in PE
            %outtensor_pe_allocation = ddl.allocate(%outtensor) {memory="pelrf"}
            %pe_outtensor_lrf = ddl.unit(%outtensor, %outtensor_pe_allocation) {unit="pe", data_connect="pe_outtensor"}
            %pe_dst_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="pe_lxsu_outtensor"}
            %pe_fma_src00 = ddl.unit(%ptsum) {unit="pt", data_connect="pe_ptsum"}
            %is_relu_or_leakyrelu = ddl.condition_or(%relu_op, %leakyrelu_op, %relu6_op)
            %is_any_aux_op = ddl.condition_or(%stradd_op, %bn_op, %bn2_op, %bias_op, %is_relu_or_leakyrelu)
            %cond_aux_ops_iter = ddl.condition_and(%cond_last_accum_loops, %is_any_aux_op)
            ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
              ddl.if(%cond_first_accum_loops) {
                ddl.if (%cond_aux_ops_iter) {
                  // chunk accumulation in PE: 0 + ptsum => lrf
                  ddl.compute([%pe_fma_src00, %one_const, %zero_const], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                } else {
                  // chunk accumulation in PE: 0 + ptsum => lxsu
                  ddl.compute([%pe_fma_src00, %one_const, %zero_const], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}
                }
              } else {
                %pe_lxpsum = ddl.unit(%outtensor) {unit="lxlu", data_connect="lxpe_output"}
                ddl.if (%cond_aux_ops_iter) {
                  // chunk accumulation in PE: lxpsum + ptsum => lrf
                  ddl.compute([%pe_fma_src00, %one_const, %pe_lxpsum], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                } else {
                  // chunk accumulation in PE: lxpsum + ptsum => lxsu
                  ddl.compute([%pe_fma_src00, %one_const, %pe_lxpsum], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}
                }
              }
            }                            

            // Aux ops in PE 
            %is_psum_not_last = ddl.condition_or(%stradd_op, %bn_op, %bn2_op, %bias_op, %is_relu_or_leakyrelu)
            %is_bn_not_last = ddl.condition_or(%bn2_op, %bias_op, %stradd_op, %is_relu_or_leakyrelu)
            %is_bn2_not_last = ddl.condition_or(%bias_op, %stradd_op, %is_relu_or_leakyrelu)
            %is_bias_not_last = ddl.condition_or(%stradd_op, %is_relu_or_leakyrelu)
            %is_stradd_not_last = ddl.condition_or(%is_relu_or_leakyrelu)
            ddl.if (%cond_aux_ops_iter) {
              // bn
              ddl.if (%bn_op) {
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  %pe_bn_src01 = ddl.unit(%bnA, %bnA_pe_allocation) {unit="pe", data_connect="pe_bnA"}
                  %pe_bn_src02 = ddl.unit(%bnB, %bnB_pe_allocation) {unit="pe", data_connect="pe_bnB"}
                  ddl.if(%is_bn_not_last) {
                    // bn: lrf * lrf + lrf -> lrf
                    ddl.compute([%pe_outtensor_lrf, %pe_bn_src01, %pe_bn_src02], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                  } else {
                    // bn: lrf * lrf + lrf -> lxsu
                    ddl.compute([%pe_outtensor_lrf, %pe_bn_src01, %pe_bn_src02], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}                                        
                  }
                }
              }
              ddl.if (%bn2_op) {
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  %pe_bn2_src01 = ddl.unit(%bnA2, %bnA2_pe_allocation) {unit="pe", data_connect="pe_bnA2"}
                  %pe_bn2_src02 = ddl.unit(%bnB2, %bnB2_pe_allocation) {unit="pe", data_connect="pe_bnB2"}
                  ddl.if(%is_bn2_not_last) {
                    // bn: lrf * lrf + lrf -> lrf
                    ddl.compute([%pe_outtensor_lrf, %pe_bn2_src01, %pe_bn2_src02], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                  } else {
                    // bn: lrf * lrf + lrf -> lxsu
                    ddl.compute([%pe_outtensor_lrf, %pe_bn2_src01, %pe_bn2_src02], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}
                  }
                }
              }
              // biasadd
              ddl.if (%bias_op) {
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  %pe_biasadd_src02 = ddl.unit(%bias, %bias_pe_allocation) {unit="pe", data_connect="pe_bias"}
                  ddl.if(%is_bias_not_last) {
                    // biasadd: 1.0 * lrf + lrf -> lrf
                    ddl.compute([%pe_outtensor_lrf, %one_const, %pe_biasadd_src02], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                  } else {
                    // biasadd: 1.0 * lrf + lrf -> lxsu
                    ddl.compute([%pe_outtensor_lrf, %one_const, %pe_biasadd_src02], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}
                  }
                }
              }
              // strided add
              ddl.if (%stradd_op) {
                // load stridedAdd input from lxlu to pe
                %src_resadd_lxpe = ddl.unit(%resadd, %resadd_lx_allocation) {unit="lxlu", data_connect="l3_lx_resadd"}
                %dst_resadd_lxpe = ddl.unit(%resadd) {unit="pe", data_connect="pe_resadd"} 
                ddl.data_transfer(%src_resadd_lxpe, [%dst_resadd_lxpe]) {}

                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  %pe_stradd_src02_pe = ddl.unit(%resadd) {unit="pe", data_connect="pe_resadd"}
                  ddl.if(%is_stradd_not_last) {
                    // stridedadd: pe + lrf -> lrf
                    ddl.compute([%pe_outtensor_lrf, %one_const, %pe_stradd_src02_pe], [%pe_outtensor_lrf]) {computetype="FMA16", unit="pe"}
                  } else {
                    // stridedadd: pe + lrf -> lxsu
                    ddl.compute([%pe_outtensor_lrf, %one_const, %pe_stradd_src02_pe], [%pe_dst_lxsu]) {computetype="FMA16", unit="pe"}
                  }
                }
              }
              // relu
              ddl.if (%relu_op) {
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  // relu: lrf -> lxsu
                  ddl.compute([%pe_outtensor_lrf, %zero_const], [%pe_dst_lxsu]) {computetype="FMAX", unit="pe"}
                }
              }
              // relu6
              ddl.if (%relu6_op) {
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  // relu: lrf -> lrf
                  ddl.compute([%pe_outtensor_lrf, %zero_const], [%pe_outtensor_lrf]) {computetype="FMAX", unit="pe"}
                }
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  // relu: lrf -> lxsu
                  ddl.compute([%pe_outtensor_lrf, %clip_pe], [%pe_dst_lxsu]) {computetype="FMIN", unit="pe"}
                }
              }
              // leakyrelu
              ddl.if (%leakyrelu_op) {
                %leaky_inter_pe_allocation = ddl.allocate(%leakyrelu_intermediate) {memory="pelrf"} 
                %pe_leaky_inter_lrf = ddl.unit(%leakyrelu_intermediate, %leaky_inter_pe_allocation) {unit="pe", data_connect="pe_leak_inter_lrf"}
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  // leaky val: lrf -> lrf
                  ddl.compute([%pe_outtensor_lrf, %leak_pe], [%pe_leaky_inter_lrf]) {computetype="FMUL", unit="pe"}
                }
                ddl.loop (%interleave_datastage_pe, %bottom_datastage_pe, %j, %krd#0, %krd#1) {} {
                  // leakyrelu: lrf -> lxsu
                  ddl.compute([%pe_outtensor_lrf, %pe_leaky_inter_lrf], [%pe_dst_lxsu]) {computetype="FMAX", unit="pe"}
                }
              }
            }
            // store output tensor pe-lx
            // we do it inside the loop to keep the same order as data arrived from the pt rows
            %src_out_pelx = ddl.unit(%outtensor) {unit="pe", data_connect="pe_lxsu_outtensor"}
            %dst_out_pelx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
            ddl.data_transfer(%src_out_pelx, [%dst_out_pelx]) {}
          }
        }
      }
    }
  }
}

ddl.transformations {
}

}
