//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%d:6 = ddl.dimension {} : index, index, index, index, index, index
%slice_layout = ddl.layout() {is_order_fixed=false} 
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}
%type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
%inptensor, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index, index
%temptensor1 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%temptensor2 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%temptensor3 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

%relu6_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="relu6fwd", required=false}
%clip_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="clip", required=false}
%leakyrelu_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="leakyrelufwd", required=false}
%fastexp_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="fastexp", required=false}
%exp_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="exp", required=false}
%fastsigmoid_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="fastsigmoid", required=false} // not possible in fp32 (FEST hardware limitation)
%mish_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="mish", required=false} // not possible in fp32 (FEST hardware limitation)
%log_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="log", required=false}
%softplus_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="softplus", required=false}

ddl.constraint(%inptensor, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor, %outtensor) {property = "stick", cmp = "equal"}
ddl.constraint() {min_num_cores = 1}
ddl.constraint(%relu6_op, %clip_op, %leakyrelu_op, %fastexp_op, %exp_op, %fastsigmoid_op, %mish_op, %log_op, %softplus_op) {min_num_valid = 1, max_num_valid = 1}

%leak_const = ddl.define_constant(%type_fp16){value=[0x3733], name="leakconst"}
%const_6 = ddl.define_constant(%type_fp16){value=[0x4300], name="const_6"}
%clip_min_const = ddl.get_external_constant(%type_fp16){name="clipMin", num_elements=1}
%clip_max_const = ddl.get_external_constant(%type_fp16){name="clipMax", num_elements=1}
%eps_const = ddl.define_constant(%type_fp16){value=[0x1000], name="eps"}
%fastexp_const = ddl.define_constant(%type_fp16){value=[0x54e3], name="fastexpVal"}
%const_46DC = ddl.define_constant(%type_fp16){value=[0x46dc], name="expVal1"}
%const_46E2 = ddl.define_constant(%type_fp16){value=[0x46e2], name="expVal2"}
%const_34C5 = ddl.define_constant(%type_fp16){value=[0x34c5], name="expVal3"}
%const_2121 = ddl.define_constant(%type_fp16){value=[0x2121], name="expVal4"}
%const_3E00 = ddl.define_constant(%type_fp16){value=[0x3e00], name="expVal5"}
%const_3C00 = ddl.define_constant(%type_fp16){value=[0x3c00], name="fastSigmoidConst"}
%const_0 = ddl.define_constant(%type_fp16){value=[0], name="zero", num_elements=1}
%mish_const1 = ddl.define_constant(%type_fp16) {value=[0x3A00], name = "mishConstVal1"}
%mish_const2 = ddl.define_constant(%type_fp16) {value=[0x3AAB], name = "mishConstVal2"}
%mish_const3 = ddl.define_constant(%type_fp16) {value=[0xBF00], name = "mishConstVal3"}
%mish_const4 = ddl.define_constant(%type_fp16) {value=[0x3CC6], name = "mishConstVal4"}
%mish_const5 = ddl.define_constant(%type_fp16) {value=[0x3E00], name = "plus1"}
%mish_const6 = ddl.define_constant(%type_fp16) {value=[0xBE00], name = "minus1"}
%mish_const7 = ddl.define_constant(%type_fp16) {value=[0x54E3], name = "mishConstVal5"}
%ffff_const = ddl.define_constant(%type_fp16) {value=[0xFFFF], name = "ffff"}
%softplus_beta = ddl.get_external_constant(%type_fp16) {name="softplusBeta", num_elements=1}
%softplus_thresh = ddl.get_external_constant(%type_fp16) {name="softplusThresh", num_elements=1}
%softplus_const_eps = ddl.define_constant(%type_fp16) {value=[0x2C8A], name = "logConstVal"}

%zero_const = ddl.operand_constant {name="0.0"}
%one_const = ddl.operand_constant {name="1.0"}



// allocate space: lx
%allocate_handler_input_lx = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"} 
%allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_input"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage = ddl.datastage {strategy="minimize"}
  ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5) {values=["1", "2", "4"]}

  // Constants
  %exp_pe_allocation = ddl.allocate(%fastexp_const) {memory="pelrf"} 
  %const_46DC_pe_alloc = ddl.allocate(%const_46DC) {memory="pelrf"} 
  %const_46E2_pe_alloc = ddl.allocate(%const_46E2) {memory="pelrf"} 
  %const_34C5_pe_alloc = ddl.allocate(%const_34C5) {memory="pelrf"} 
  %const_2121_pe_alloc = ddl.allocate(%const_2121) {memory="pelrf"} 
  %const_3E00_pe_alloc = ddl.allocate(%const_3E00) {memory="pelrf"} 
  %const_3E00_sfp_alloc = ddl.allocate(%const_3E00) {memory="sfplrf"} 
  %eps_sfp_allocation = ddl.allocate(%eps_const) {memory="sfplrf"}
  %const_3C00_sfp_alloc = ddl.allocate(%const_3C00) {memory="sfplrf"} 
  %leak_pe_allocation = ddl.allocate(%leak_const) {memory="pelrf"} 
  %const_6_sfp_allocation = ddl.allocate(%const_6) {memory="sfplrf"}
  %clip_min_const_pe_allocation = ddl.allocate(%clip_min_const) {memory="pelrf"}
  %clip_max_const_sfp_allocation = ddl.allocate(%clip_max_const) {memory="sfplrf"}
  // Mish constant allocations
  %mish_const0_pe_allocation = ddl.allocate(%const_0) {memory="pelrf"}
  %mish_const5_pe_allocation = ddl.allocate(%mish_const5) {memory="pelrf"}
  %mish_const6_pe_allocation = ddl.allocate(%mish_const6) {memory="pelrf"}
  %mish_const7_pe_allocation = ddl.allocate(%mish_const7) {memory="pelrf"}

  %mish_const1_sfp_allocation = ddl.allocate(%mish_const1) {memory="sfplrf"}
  %mish_const2_sfp_allocation = ddl.allocate(%mish_const2) {memory="sfplrf"}
  %mish_const3_sfp_allocation = ddl.allocate(%mish_const3) {memory="sfplrf"}
  %mish_const4_sfp_allocation = ddl.allocate(%mish_const4) {memory="sfplrf"}
  // Log constant allocations
  %log_const0_pe_allocation = ddl.allocate(%const_0) {memory="pelrf"}
  %log_const6_pe_allocation = ddl.allocate(%mish_const6) {memory="pelrf"}
  %log_const8_pe_allocation = ddl.allocate(%ffff_const) {memory="pelrf"}
  %log_const1_sfp_allocation = ddl.allocate(%mish_const1) {memory="sfplrf"}
  %log_const2_sfp_allocation = ddl.allocate(%mish_const2) {memory="sfplrf"}
  %log_const3_sfp_allocation = ddl.allocate(%mish_const3) {memory="sfplrf"}
  %log_const4_sfp_allocation = ddl.allocate(%mish_const4) {memory="sfplrf"}
  %log_const5_sfp_allocation = ddl.allocate(%mish_const5) {memory="sfplrf"}
  // Softplus constant allocation
  %softplus_beta_pe_allocation = ddl.allocate(%softplus_beta) {memory="pelrf"}
  %softplus_zero_pe_allocation = ddl.allocate(%const_0) {memory="pelrf"}
  %softplus_one_pe_allocation = ddl.allocate(%mish_const5) {memory="pelrf"}
  %softplus_minus_pe_allocation = ddl.allocate(%mish_const6) {memory="pelrf"}
  %softplus_ffff_pe_allocation = ddl.allocate(%ffff_const) {memory="pelrf"}
  %softplus_thresh_sfp_allocation = ddl.allocate(%softplus_thresh) {memory="sfplrf"}
  %softplus_const1_sfp_allocation = ddl.allocate(%mish_const1) {memory="sfplrf"}
  %softplus_const2_sfp_allocation = ddl.allocate(%mish_const2) {memory="sfplrf"}
  %softplus_const3_sfp_allocation = ddl.allocate(%mish_const3) {memory="sfplrf"}
  %softplus_const4_sfp_allocation = ddl.allocate(%mish_const4) {memory="sfplrf"}
  %softplus_const5_sfp_allocation = ddl.allocate(%mish_const5) {memory="sfplrf"}

  ddl.if (%mish_op) {
    %src_mish_const0_opaque = ddl.unit(%const_0) {unit="constant", data_connect= "mish_const0_opaque_connect"}
    %dst_mish_const0_pe = ddl.unit(%const_0, %mish_const0_pe_allocation) {unit="pe", data_connect= "mish_const0_pe_lrf"}

    %src_mish_const5_opaque = ddl.unit(%mish_const5) {unit="constant", data_connect= "mish_const5_opaque_connect"}
    %dst_mish_const5_pe = ddl.unit(%mish_const5, %mish_const5_pe_allocation) {unit="pe", data_connect= "mish_const5_pe_lrf"}

    %src_mish_const6_opaque = ddl.unit(%mish_const6) {unit="constant", data_connect= "mish_const6_opaque_connect"}
    %dst_mish_const6_pe = ddl.unit(%mish_const6, %mish_const6_pe_allocation) {unit="pe", data_connect= "mish_const6_pe_lrf"}

    %src_mish_const7_opaque = ddl.unit(%mish_const7) {unit="constant", data_connect= "mish_const7_opaque_connect"}
    %dst_mish_const7_pe = ddl.unit(%mish_const7, %mish_const7_pe_allocation) {unit="pe", data_connect= "mish_const7_pe_lrf"}


    %src_mish_const1_opaque = ddl.unit(%mish_const1) {unit="constant", data_connect= "mish_const1_opaque_connect"}
    %dst_mish_const1_sfp = ddl.unit(%mish_const1, %mish_const1_sfp_allocation) {unit="sfp", data_connect= "mish_const1_sfp_lrf"}

    %src_mish_const2_opaque = ddl.unit(%mish_const2) {unit="constant", data_connect= "mish_const2_opaque_connect"}
    %dst_mish_const2_sfp = ddl.unit(%mish_const2, %mish_const2_sfp_allocation) {unit="sfp", data_connect= "mish_const2_sfp_lrf"}

    %src_mish_const3_opaque = ddl.unit(%mish_const3) {unit="constant", data_connect= "mish_const3_opaque_connect"}
    %dst_mish_const3_sfp = ddl.unit(%mish_const3, %mish_const3_sfp_allocation) {unit="sfp", data_connect= "mish_const3_sfp_lrf"}

    %src_mish_const4_opaque = ddl.unit(%mish_const4) {unit="constant", data_connect= "mish_const4_opaque_connect"}
    %dst_mish_const4_sfp = ddl.unit(%mish_const4, %mish_const4_sfp_allocation) {unit="sfp", data_connect= "mish_const4_sfp_lrf"}


    ddl.data_transfer(%src_mish_const0_opaque, [%dst_mish_const0_pe]) {}
    ddl.data_transfer(%src_mish_const5_opaque, [%dst_mish_const5_pe]) {}
    ddl.data_transfer(%src_mish_const6_opaque, [%dst_mish_const6_pe]) {}
    ddl.data_transfer(%src_mish_const7_opaque, [%dst_mish_const7_pe]) {}

    ddl.data_transfer(%src_mish_const1_opaque, [%dst_mish_const1_sfp]) {}
    ddl.data_transfer(%src_mish_const2_opaque, [%dst_mish_const2_sfp]) {}
    ddl.data_transfer(%src_mish_const3_opaque, [%dst_mish_const3_sfp]) {}
    ddl.data_transfer(%src_mish_const4_opaque, [%dst_mish_const4_sfp]) {}

  }
  ddl.if (%log_op) {
    %src_log_const0_opaque = ddl.unit(%const_0) {unit="constant", data_connect= "log_const0_opaque_connect"}
    %dst_log_const0_pe = ddl.unit(%const_0, %log_const0_pe_allocation) {unit="pe", data_connect= "log_const0_pe_lrf"}
    %src_log_const6_opaque = ddl.unit(%mish_const6) {unit="constant", data_connect= "log_const6_opaque_connect"}
    %dst_log_const6_pe = ddl.unit(%mish_const6, %log_const6_pe_allocation) {unit="pe", data_connect= "log_const6_pe_lrf"}
    %src_log_const8_opaque = ddl.unit(%ffff_const) {unit="constant", data_connect= "log_const8_opaque_connect"}
    %dst_log_const8_pe = ddl.unit(%ffff_const, %log_const8_pe_allocation) {unit="pe", data_connect= "log_const8_pe_lrf"}


    %src_log_const1_opaque = ddl.unit(%mish_const1) {unit="constant", data_connect= "log_const1_opaque_connect"}
    %dst_log_const1_sfp = ddl.unit(%mish_const1, %log_const1_sfp_allocation) {unit="sfp", data_connect= "log_const1_sfp_lrf"}
    %src_log_const2_opaque = ddl.unit(%mish_const2) {unit="constant", data_connect= "log_const2_opaque_connect"}
    %dst_log_const2_sfp = ddl.unit(%mish_const2, %log_const2_sfp_allocation) {unit="sfp", data_connect= "log_const2_sfp_lrf"}
    %src_log_const3_opaque = ddl.unit(%mish_const3) {unit="constant", data_connect= "log_const3_opaque_connect"}
    %dst_log_const3_sfp = ddl.unit(%mish_const3, %log_const3_sfp_allocation) {unit="sfp", data_connect= "log_const3_sfp_lrf"}
    %src_log_const4_opaque = ddl.unit(%mish_const4) {unit="constant", data_connect= "log_const4_opaque_connect"}
    %dst_log_const4_sfp = ddl.unit(%mish_const4, %log_const4_sfp_allocation) {unit="sfp", data_connect= "log_const4_sfp_lrf"}
    %src_log_const5_opaque = ddl.unit(%mish_const5) {unit="constant", data_connect= "log_const5_opaque_connect"}
    %dst_log_const5_sfp = ddl.unit(%mish_const5, %log_const5_sfp_allocation) {unit="sfp", data_connect= "log_const5_sfp_lrf"}

    ddl.data_transfer(%src_log_const0_opaque, [%dst_log_const0_pe]) {}
    ddl.data_transfer(%src_log_const6_opaque, [%dst_log_const6_pe]) {}
    ddl.data_transfer(%src_log_const8_opaque, [%dst_log_const8_pe]) {}

    ddl.data_transfer(%src_log_const1_opaque, [%dst_log_const1_sfp]) {}
    ddl.data_transfer(%src_log_const2_opaque, [%dst_log_const2_sfp]) {}
    ddl.data_transfer(%src_log_const3_opaque, [%dst_log_const3_sfp]) {}
    ddl.data_transfer(%src_log_const4_opaque, [%dst_log_const4_sfp]) {}
    ddl.data_transfer(%src_log_const5_opaque, [%dst_log_const5_sfp]) {}
  }

  ddl.if (%fastexp_op) {
    %src_fastexp_const = ddl.unit(%fastexp_const) {unit="constant", data_connect= "fastexp_const_connect"} 
    %dst_exp_pe = ddl.unit(%fastexp_const, %exp_pe_allocation) {unit="pe", data_connect= "exp_pe_lrf"}
    ddl.data_transfer(%src_fastexp_const, [%dst_exp_pe]) {}
  } 
  ddl.if (%exp_op) {
    %src_const_46DC = ddl.unit(%const_46DC) {unit="constant", data_connect= "const_46DC_connect"} 
    %dst_expconst1_pe = ddl.unit(%const_46DC, %const_46DC_pe_alloc) {unit="pe", data_connect= "exp_pe_const1_lrf"}
    ddl.data_transfer(%src_const_46DC, [%dst_expconst1_pe]) {}
    %src_const_46E2 = ddl.unit(%const_46E2) {unit="constant", data_connect= "const_46E2_connect"} 
    %dst_expconst2_pe = ddl.unit(%const_46E2, %const_46E2_pe_alloc) {unit="pe", data_connect= "exp_pe_const2_lrf"}
    ddl.data_transfer(%src_const_46E2, [%dst_expconst2_pe]) {}
    %src_const_34C5 = ddl.unit(%const_34C5) {unit="constant", data_connect= "const_34C5_connect"} 
    %dst_expconst3_pe = ddl.unit(%const_34C5, %const_34C5_pe_alloc) {unit="pe", data_connect= "exp_pe_const3_lrf"}
    ddl.data_transfer(%src_const_34C5, [%dst_expconst3_pe]) {}
    %src_const_2121 = ddl.unit(%const_2121) {unit="constant", data_connect= "const_2121_connect"} 
    %dst_expconst4_pe = ddl.unit(%const_2121, %const_2121_pe_alloc) {unit="pe", data_connect= "exp_pe_const4_lrf"}
    ddl.data_transfer(%src_const_2121, [%dst_expconst4_pe]) {}
    %src_const_3E00 = ddl.unit(%const_3E00) {unit="constant", data_connect= "const_3E00_connect"} 
    %dst_expconst5_sfp = ddl.unit(%const_3E00, %const_3E00_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const5_lrf"}
    ddl.data_transfer(%src_const_3E00, [%dst_expconst5_sfp]) {}
    %src_eps_const = ddl.unit(%eps_const) {unit="constant", data_connect= "eps_const_connect"}
    %dst_eps_sfp = ddl.unit(%eps_const, %eps_sfp_allocation) {unit="sfp", data_connect= "eps_sfp_lrf"}
    ddl.data_transfer(%src_eps_const, [%dst_eps_sfp]) {}
  }

  ddl.if (%fastsigmoid_op) {
    %src_const_3C00 = ddl.unit(%const_3C00) {unit="constant", data_connect= "const_3C00_connect"} 
    %dst_fastsigmoid_sfp = ddl.unit(%const_3C00, %const_3C00_sfp_alloc) {unit="sfp", data_connect= "fastsigmoid_connect"}
    ddl.data_transfer(%src_const_3C00, [%dst_fastsigmoid_sfp]) {}
  }
  ddl.if(%leakyrelu_op){
    %src_leak_const = ddl.unit(%leak_const) {unit="constant", data_connect= "leak_const_connect"} 
    %dst_leak_pe = ddl.unit(%leak_const, %leak_pe_allocation) {unit="pe", data_connect= "leak_pe_lrf"}
    ddl.data_transfer(%src_leak_const, [%dst_leak_pe]) {}
  }
  ddl.if(%relu6_op){
    %src_const_6 = ddl.unit(%const_6) {unit="constant", data_connect= "const_6_connect"} 
    %dst_const_6_sfp = ddl.unit(%const_6, %const_6_sfp_allocation) {unit="sfp", data_connect= "const_6_sfp_lrf"}
    ddl.data_transfer(%src_const_6, [%dst_const_6_sfp]) {}
  }
  ddl.if(%clip_op){
    %src_clip_min_const = ddl.unit(%clip_min_const) {unit="constant", data_connect= "clip_min_const_connect"} 
    %dst_clip_min_const_pe = ddl.unit(%clip_min_const, %clip_min_const_pe_allocation) {unit="pe", data_connect= "clip_min_const_lrf"}
    ddl.data_transfer(%src_clip_min_const, [%dst_clip_min_const_pe]) {}
    %src_clip_max_const = ddl.unit(%clip_max_const) {unit="constant", data_connect= "clip_max_const_connect"} 
    %dst_clip_max_const_sfp = ddl.unit(%clip_max_const, %clip_max_const_sfp_allocation) {unit="sfp", data_connect= "clip_max_const_lrf"}
    ddl.data_transfer(%src_clip_max_const, [%dst_clip_max_const_sfp]) {}
  }
  ddl.if(%softplus_op) {
    %src_softplus_beta = ddl.unit(%softplus_beta) {unit="constant", data_connect= "softplus_beta_connect"}
    %dst_softplus_pe = ddl.unit(%softplus_beta, %softplus_beta_pe_allocation) {unit="pe", data_connect= "softplus_beta_pe_lrf"}
    ddl.data_transfer(%src_softplus_beta, [%dst_softplus_pe]) {}

    %src_fastexp_const = ddl.unit(%fastexp_const) {unit="constant", data_connect= "fastexp_const_connect"}
    %dst_exp_pe = ddl.unit(%fastexp_const, %exp_pe_allocation) {unit="pe", data_connect= "exp_pe_lrf"}
    ddl.data_transfer(%src_fastexp_const, [%dst_exp_pe]) {}

    %src_zero_const = ddl.unit(%const_0) {unit="constant", data_connect= "zero_const_connect"}
    %dst_zero_const_pe = ddl.unit(%const_0, %softplus_zero_pe_allocation) {unit="pe", data_connect= "zero_const_pe_lrf"}
    ddl.data_transfer(%src_zero_const, [%dst_zero_const_pe]) {}

    %src_one_const = ddl.unit(%mish_const5) {unit="constant", data_connect= "one_const_connect"}
    %dst_one_const_pe = ddl.unit(%mish_const5, %softplus_one_pe_allocation) {unit="pe", data_connect= "one_const_pe_lrf"}
    ddl.data_transfer(%src_one_const, [%dst_one_const_pe]) {}

    %src_ffff_const = ddl.unit(%ffff_const) {unit="constant", data_connect= "ffff_const_connect"}
    %dst_ffff_const_pe = ddl.unit(%ffff_const, %softplus_ffff_pe_allocation) {unit="pe", data_connect= "ffff_const_pe_lrf"}
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_const_pe]) {}

    %src_minus_const = ddl.unit(%mish_const6) {unit="constant", data_connect= "minus_const_connect"}
    %dst_minus_const_pe = ddl.unit(%mish_const6, %softplus_minus_pe_allocation) {unit="pe", data_connect= "minus_const_pe_lrf"}
    ddl.data_transfer(%src_minus_const, [%dst_minus_const_pe]) {}

    %src_log_const1_opaque = ddl.unit(%mish_const1) {unit="constant", data_connect= "log_const1_opaque_connect"}
    %dst_log_const1_sfp = ddl.unit(%mish_const1, %softplus_const1_sfp_allocation) {unit="sfp", data_connect= "softplus_const1_sfp_lrf"}
    ddl.data_transfer(%src_log_const1_opaque, [%dst_log_const1_sfp]) {}

    %src_log_const2_opaque = ddl.unit(%mish_const2) {unit="constant", data_connect= "log_const2_opaque_connect"}
    %dst_log_const2_sfp = ddl.unit(%mish_const2, %softplus_const2_sfp_allocation) {unit="sfp", data_connect= "softplus_const2_sfp_lrf"}
    ddl.data_transfer(%src_log_const2_opaque, [%dst_log_const2_sfp]) {}

    %src_log_const3_opaque = ddl.unit(%mish_const3) {unit="constant", data_connect= "log_const3_opaque_connect"}
    %dst_log_const3_sfp = ddl.unit(%mish_const3, %softplus_const3_sfp_allocation) {unit="sfp", data_connect= "softplus_const3_sfp_lrf"}
    ddl.data_transfer(%src_log_const3_opaque, [%dst_log_const3_sfp]) {}

    %src_log_const4_opaque = ddl.unit(%mish_const4) {unit="constant", data_connect= "log_const4_opaque_connect"}
    %dst_log_const4_sfp = ddl.unit(%mish_const4, %softplus_const4_sfp_allocation) {unit="sfp", data_connect= "softplus_const4_sfp_lrf"}
    ddl.data_transfer(%src_log_const4_opaque, [%dst_log_const4_sfp]) {}

    %src_log_const5_opaque = ddl.unit(%mish_const5) {unit="constant", data_connect= "log_const5_opaque_connect"}
    %dst_log_const5_sfp = ddl.unit(%mish_const5, %softplus_const5_sfp_allocation) {unit="sfp", data_connect= "softplus_const5_sfp_lrf"}
    ddl.data_transfer(%src_log_const5_opaque, [%dst_log_const5_sfp]) {}

    %src_softplus_thresh = ddl.unit(%softplus_thresh) {unit="constant", data_connect= "softplus_thresh_connect"}
    %dst_softplus_thresh_sfp = ddl.unit(%softplus_thresh, %softplus_thresh_sfp_allocation) {unit="sfp", data_connect= "softplus_thresh_sfp_lrf"}
    ddl.data_transfer(%src_softplus_thresh, [%dst_softplus_thresh_sfp]) {}
  }
  // main dataflow
  ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
    ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 

      %pe_lrf_allocation = ddl.allocate(%inptensor) {memory="pelrf"} 
     
      // send input
      // lx-pe 
      %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
      ddl.if(%fastsigmoid_op) {
        %dst_inp_pelrf = ddl.unit(%inptensor, %pe_lrf_allocation) {unit="pe", data_connect="pe_lx_lrf_input"}  // allocate input in pe-register for multiple use
        ddl.data_transfer(%src_inp_lx, [%dst_inp_pelrf]) {}
      } else {
        %dst_inp_pe = ddl.unit(%inptensor) {unit="pe", data_connect="pe_lx_input"}  // no need to allocate input in pe-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_pe]) {}
      }

      %input_to_sfp = ddl.condition_or(%leakyrelu_op, %fastsigmoid_op, %mish_op, %softplus_op)

      ddl.if(%input_to_sfp){
        // lx to sfp
        %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"} // no need to allocate input in pe-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}
      }

      // compute in PE
      ddl.if (%fastexp_op) {
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // FMUL mask=255 be=be src0=lxlu src1=R1 src2=0.0 tgtsfp=result
          %pe_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="pe_lx_input"}
          %pe_src01 = ddl.unit(%fastexp_const, %exp_pe_allocation) {unit="pe", data_connect="exp_pe_lrf"}
          %pe_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_output"}
          ddl.compute([%pe_src00, %pe_src01], [%pe_dst00]) {computetype="FMUL", unit="pe"}
        }
      }
      ddl.if (%exp_op) {
        ddl.opaque(%outtensor, %const_46DC_pe_alloc, %const_46E2_pe_alloc, %const_34C5_pe_alloc, %const_2121_pe_alloc)
            {unit="pe", op="EXP_P1", input_output_registers=["c0", "c1", "c2", "c3"], internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
            max_unroll_factor=4, params={"in0"="lxlu", "out0"="result"},
            input_data_connects=["exp_pe_const1_lrf", "exp_pe_const2_lrf", "exp_pe_const3_lrf", "exp_pe_const4_lrf", "pe_lx_input"],
            output_data_connects=["pe_output", "pe_out1", "pe_out3"]}
      }

      ddl.if (%mish_op) {
        ddl.opaque(%outtensor, %mish_const0_pe_allocation, %mish_const5_pe_allocation, %mish_const6_pe_allocation, %mish_const7_pe_allocation)
                {unit="pe", op="MISH_P1", input_output_registers=["c0", "c5", "c6", "c7"],
                internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
                max_unroll_factor=4, params={"in0"="lxlu", "out0"="result"},
                input_data_connects=["mish_const0_pe_lrf", "mish_const5_pe_lrf", "mish_const6_pe_lrf", "mish_const7_pe_lrf", "pe_lx_input"],
                output_data_connects=["pe_output", "pe_out1", "pe_out3"]}
      }
      ddl.if (%log_op) {
        ddl.opaque(%outtensor, %log_const0_pe_allocation, %log_const6_pe_allocation, %log_const8_pe_allocation)
                {unit="pe", op="LOG_P1", input_output_registers=["c0", "c6", "c8"],
                internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
                max_unroll_factor=1, params={"in0"="lxlu", "out0"="result"},
                input_data_connects=["log_const0_pe_lrf", "log_const6_pe_lrf", "log_const8_pe_lrf", "pe_lx_input"],
                output_data_connects=["pe_out1", "pe_out3"]}
      }

      ddl.if (%fastsigmoid_op) {
        // FAST SIGMOID
        %pe_src00 = ddl.unit(%inptensor, %pe_lrf_allocation) {unit="pe", data_connect="pe_lx_lrf_input"}        
        %pe_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_out1"}
        %pe_dst20 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_out3"}
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // FEST    mask=255 be=0  src0=lrf  imm=0x7 src2=0.0 peer=result  // sigmoid offset, send to PE
          ddl.compute([%pe_src00], [%pe_dst00]) {computetype="FEST", unit="pe", mode=7}        
        }        
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // FEST    mask=255 be=be src0=lrf imm=0x6 src2=0.0 peer=result  // sigmoid slope, send to PE
          ddl.compute([%pe_src00], [%pe_dst20]) {computetype="FEST", unit="pe", mode=6}
        }        

      }
      ddl.if(%leakyrelu_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          %pe_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="pe_lx_input"}
          %pe_src01 = ddl.unit(%leak_const, %leak_pe_allocation) {unit="pe", data_connect="leak_pe_lrf"}
          %pe_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_output"}
          ddl.compute([%pe_src00, %pe_src01], [%pe_dst00]) {computetype="FMUL", unit="pe"}
        }
      }

      ddl.if(%relu6_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          %pe_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="pe_lx_input"}
          %pe_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_output"}
          ddl.compute([%pe_src00, %zero_const], [%pe_dst00]) {computetype="FMAX", unit="pe"}
        }
      }
      ddl.if(%clip_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          %pe_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="pe_lx_input"}
          %pe_src02 = ddl.unit(%clip_min_const, %clip_min_const_pe_allocation) {unit="pe", data_connect="clip_min_const_lrf"}
          %pe_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_output"}
          ddl.compute([%pe_src00, %pe_src02], [%pe_dst00]) {computetype="FMAX", unit="pe"}
        }
      }

      ddl.if(%softplus_op) {
        ddl.opaque(%outtensor, %exp_pe_allocation, %softplus_one_pe_allocation, %softplus_ffff_pe_allocation,
                   %softplus_zero_pe_allocation, %softplus_beta_pe_allocation, %softplus_minus_pe_allocation)
            {unit="pe", op="SOFTPLUS_P1", input_output_registers=["c0", "c1", "c2", "c3", "c4", "c5"],
            internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
            max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
            input_data_connects=["exp_pe_lrf", "softplus_beta_pe_lrf", "one_const_pe_lrf", "ffff_const_pe_lrf",
                                 "zero_const_pe_lrf", "pe_lx_input", "minus_const_pe_lrf"],
            output_data_connects=["pe_output"]}
      }

      // pe-sfp
      %double_transfer = ddl.condition_or(%exp_op, %fastsigmoid_op, %mish_op, %log_op)
      ddl.if (%double_transfer) { // 2 transfers
        %src_out1_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_out1"} 
        %dst_out1_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_input1"}
        ddl.data_transfer(%src_out1_pesfp, [%dst_out1_pesfp]) {}
        %src_out3_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_out3"} 
        %dst_out3_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_input3"}
        ddl.data_transfer(%src_out3_pesfp, [%dst_out3_pesfp]) {}
      } else {
        %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_output"} 
        %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_input"}
        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
      }

      // compute in sfp
      ddl.if (%fastexp_op) {
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // ICVT  mask=255 be=be src0=0xf imm=0x7 src2=pe tgtlx=result
          %sfp_src02 = ddl.unit(%inptensor) {unit="pe", data_connect="sfp_input"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src02], [%sfp_dst00]) {computetype="ICVT", unit="sfp", mode=7}
        }
      }
      ddl.if (%exp_op) {
        ddl.opaque(%outtensor, %const_3E00_sfp_alloc, %eps_sfp_allocation)
            {unit="sfp", op="EXP_P2", input_output_registers=["c0", "c1"], internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
            max_unroll_factor=4, params={"in0"="pe", "out0"="result"}, input_data_connects=["exp_sfp_const5_lrf", "pe_output", "eps_sfp_lrf"],
            output_data_connects=["sfp_output"]}
      }
      ddl.if(%mish_op) {
        ddl.opaque(%outtensor, %mish_const1_sfp_allocation, %mish_const2_sfp_allocation, %mish_const3_sfp_allocation, %mish_const4_sfp_allocation)
                {unit="sfp", op="MISH_P2", input_output_registers=["c1", "c2", "c3", "c4"],
                internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
                max_unroll_factor=4, params={"in0"="lxlu", "in1"="pe", "out0"="result"},
                input_data_connects=["mish_const1_sfp_lrf", "mish_const2_sfp_lrf", "mish_const3_sfp_lrf", "mish_const4_sfp_lrf",  "sfp_lx_input", "pe_output"],
                output_data_connects=["sfp_output"]}
      }
      ddl.if(%log_op) {
        ddl.opaque(%outtensor, %log_const1_sfp_allocation, %log_const2_sfp_allocation, %log_const3_sfp_allocation, %log_const4_sfp_allocation, %log_const5_sfp_allocation)
                {unit="sfp", op="LOG_P2", input_output_registers=["c1", "c2", "c3", "c4", "c5"],
                internal_registers=["t0_unroll", "t1_unroll", "t2_unroll"],
                max_unroll_factor=1, params={"in0"="pe", "out0"="result"},
                input_data_connects=["log_const1_sfp_lrf", "log_const2_sfp_lrf", "log_const3_sfp_lrf", "log_const4_sfp_lrf", "pe_out1", "pe_out3"],
                output_data_connects=["sfp_output"]}
      }

      ddl.if (%fastsigmoid_op) {
        // FAST SIGMOID
        %sfp_src00 = ddl.unit(%inptensor) {unit="pe", data_connect="sfp_input1"}
        %sfp_src10 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
        %sfp_src20 = ddl.unit(%inptensor) {unit="pe", data_connect="sfp_input3"}
        %sfp_offset_alloc = ddl.allocate(%outtensor) {memory="sfplrf"}
        %sfp_dst00 = ddl.unit(%outtensor, %sfp_offset_alloc) {unit="sfp", data_connect="sfp_offset"}
        %sfp_dst10 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
        %sigmoid_sfp_const = ddl.unit(%const_3C00, %const_3C00_sfp_alloc) {unit="sfp", data_connect="fastsigmoid_connect"}
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // FMA      mask=255 be=0  src0=sfp    src1=1.0  src2=R1  tgtrf=R12 // 0.5 * 1.0 + offset
          ddl.compute([%sfp_src00, %one_const, %sigmoid_sfp_const], [%sfp_dst00]) {computetype="FMA16", unit="sfp"}        
        }
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          // FMA      mask=255 be=be src0=lxlu  src1=sfp  src2=R12 tgtlx=result // sigmoid = ss * x + offset + 0.5
          ddl.compute([%sfp_src10, %sfp_src20, %sfp_dst00], [%sfp_dst10]) {computetype="FMA16", unit="sfp"}        
        }
      }
      ddl.if(%leakyrelu_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          %sfp_src00 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_input"}
          %sfp_src02 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src00, %sfp_src02], [%sfp_dst00]) {computetype="FMAX", unit="sfp"}
        }
      }
      ddl.if(%relu6_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
        //   FMINMAX be=be mask=255 mode=0 src0=peer imm=4 src2=R0 tgtlx=result
          %sfp_src00 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_input"}
          %sfp_src02 = ddl.unit(%const_6, %const_6_sfp_allocation) {unit="sfp", data_connect="const_6_sfp_lrf"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src00, %sfp_src02], [%sfp_dst00]) {computetype="FMIN", unit="sfp"}
        }
      }
      ddl.if(%clip_op){
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
          %sfp_src00 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_input"}
          %sfp_src02 = ddl.unit(%clip_max_const, %clip_max_const_sfp_allocation) {unit="sfp", data_connect="clip_max_const_lrf"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src00, %sfp_src02], [%sfp_dst00]) {computetype="FMIN", unit="sfp"}
        }
      }
      ddl.if(%softplus_op) {
        ddl.opaque(%outtensor, %softplus_const1_sfp_allocation, %softplus_const2_sfp_allocation, %softplus_const3_sfp_allocation,
                   %softplus_const4_sfp_allocation, %softplus_const5_sfp_allocation, %softplus_thresh_sfp_allocation)
          {unit="sfp", op="SOFTPLUS_P2", input_output_registers=["c1", "c2", "c3", "c4", "c5", "c9"],
          internal_registers=["t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll"],
          max_unroll_factor=2, params={"in0"="lxlu", "in1_unroll"="pe", "out0"="result"},
          input_data_connects=["softplus_const1_sfp_lrf", "softplus_const2_sfp_lrf", "softplus_const3_sfp_lrf",
                               "softplus_const4_sfp_lrf", "softplus_const5_sfp_lrf", "softplus_thresh_sfp_lrf",
                               "sfp_lx_input", "pe_output"],
          output_data_connects=["sfp_output"]}
      }

      // sfp-lx
      %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
      %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
      ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
    }  
  }
}

ddl.transformations {
}

}
