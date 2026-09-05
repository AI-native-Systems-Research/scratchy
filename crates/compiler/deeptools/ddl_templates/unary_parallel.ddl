//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%d:6 = ddl.dimension {} : index, index, index, index, index, index
%slice_layout = ddl.layout() {is_order_fixed=false} 
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_fp32 = ddl.type {data_type="IEEE_FP32"}
%type_uint32 = ddl.type {data_type="SENUINT32"}

%inptensor, %inptensor2, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16, %type_fp32]) : index, index, index
%inptensor_uint32, %outtensor_uint32 = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_uint32]) : index, index
%interim_tensor0 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%interim_tensor1 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

%gelu_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="gelufwd", required=false} // not possible in fp32 (FEST hardware limitation)
%relu_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="relufwd", required=false}
%clip_op = ddl.operation_bind([%type_fp32], [%inptensor], [%outtensor]) {opFuncName="clip", required=false}
%reciprocal_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="reciprocal", required=false}
%layernormscale_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="layernormscale", required=false}
// TEMP: disabling stickpacking opt
// %layernormscale_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor, %inptensor2], [%outtensor]) {opFuncName="layernormscale", required=false}
%tanh_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor], [%interim_tensor0, %interim_tensor1]) {opFuncName="tanh", required=false} // not possible in fp32 (FEST hardware limitation)
%abs_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="abs", required=false}
%neg_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="neg", required=false}
%silu_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="silu", required=false}
%exp_op = ddl.operation_bind([%type_fp32], [%inptensor], [%outtensor]) {opFuncName="exp", required=false}
// %fastexp_op = ddl.operation_bind([%type_fp32], [%inptensor], [%outtensor]) {opFuncName="fastexp", required=false} // ICVT imm=7 is not possible in fp32 mode
%sigmoid_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="sigmoid", required=false}
%sqrt_op = ddl.operation_bind([%type_fp16, %type_fp32],[%inptensor], [%outtensor]) {opFuncName="sqrt", required=false}
%rsqrt_op = ddl.operation_bind([%type_fp16, %type_fp32],[%inptensor], [%outtensor]) {opFuncName="rsqrt", required=false}
%identity_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="identity", required=false}
%shuffle_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="shuffle", required=false} // alias of identity
%floor_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inptensor], [%outtensor]) {opFuncName="floor", required=false}
%int32idxtoaddr_op = ddl.operation_bind([%type_fp32], [%inptensor_uint32], [%outtensor_uint32]) {opFuncName="int32idxtoaddr", required=false}


ddl.constraint(%inptensor, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor, %outtensor) {property = "stick", cmp = "equal"}
ddl.constraint(%inptensor_uint32, %outtensor_uint32) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor_uint32, %outtensor_uint32) {property = "stick", cmp = "equal"}
ddl.constraint(%inptensor2, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor2, %outtensor) {property = "stick", cmp = "equal"}
ddl.constraint(%relu_op, %clip_op, %reciprocal_op, %layernormscale_op, %tanh_op, %sqrt_op, %rsqrt_op, %gelu_op, %abs_op, %neg_op, %silu_op, %sigmoid_op, %exp_op, %identity_op, %shuffle_op, %floor_op, %int32idxtoaddr_op) {min_num_valid = 1, max_num_valid = 1} // , %fastexp_op
ddl.constraint() {min_num_cores = 1}

%ffff_const_fp16 = ddl.define_constant(%type_fp16) {value=[0xFFFF], name="ffff"}
%ffff_const_fp32 = ddl.define_constant(%type_fp32) {value=[0xFFFFFFFF], name="ffff"}
%ffff_const = ddl.alias_one_constant_of(%ffff_const_fp16, %ffff_const_fp32)
%zero_const_fp16 = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
%zero_const_fp32 = ddl.define_constant(%type_fp32) {value=[0], name="zero"}
%zero_const_opaque = ddl.alias_one_constant_of(%zero_const_fp16, %zero_const_fp32)
%clip_min_const = ddl.get_external_constant(%type_fp32){name="clipMin", num_elements=1}
%clip_max_const = ddl.get_external_constant(%type_fp32){name="clipMax", num_elements=1}
%plus1_const_fp16 = ddl.define_constant(%type_fp16) {value=[0x3E00], name="plus1"}
%plus1_const_fp32 = ddl.define_constant(%type_fp32) {value=[0x3F800000], name="plus1"}
%plus1_const = ddl.alias_one_constant_of(%plus1_const_fp16, %plus1_const_fp32)
%minus1_const_fp16 = ddl.define_constant(%type_fp16) {value=[0xBE00], name="minus1"}
%minus1_const_fp32 = ddl.define_constant(%type_fp32) {value=[0xBF800000], name="minus1"}
%minus1_const = ddl.alias_one_constant_of(%minus1_const_fp16, %minus1_const_fp32)
%eps_const = ddl.get_external_constant(%type_fp16, %type_fp32){name="eps", num_elements=1}
%eps_const_internal_fp32 = ddl.define_constant(%type_fp32){value=[0x36fffff6], name="eps_fp32"}
%maskone_const_fp16 = ddl.define_constant(%type_fp16){value=[0x0001], name="maskone"}
%maskone_const_fp32 = ddl.define_constant(%type_fp32){value=[0x00000001], name="maskone"}
%maskone_const = ddl.alias_one_constant_of(%maskone_const_fp16, %maskone_const_fp32)
%gelu_const_A = ddl.define_constant(%type_fp16){value=[0x3D31], name="geluVal1"}
%gelu_const_B = ddl.define_constant(%type_fp16){value=[0x3448], name="geluVal2"}
%fastexp_const = ddl.define_constant(%type_fp16){value=[0x54e3], name="fastexpVal"}
// idx32toaddr
%stride_byte0_const = ddl.get_external_constant(%type_fp32){name="stride_byte0", num_elements=1}
%stride_byte1_const = ddl.get_external_constant(%type_fp32){name="stride_byte1", num_elements=1}
%stride_byte2_const = ddl.get_external_constant(%type_fp32){name="stride_byte2", num_elements=1}
%stride_byte3_const = ddl.get_external_constant(%type_fp32){name="stride_byte3", num_elements=1}

%addr_byte0_const = ddl.get_external_constant(%type_fp32){name="addr_byte0", num_elements=1}
%addr_byte1_const = ddl.get_external_constant(%type_fp32){name="addr_byte1", num_elements=1}
%addr_byte2_const = ddl.get_external_constant(%type_fp32){name="addr_byte2", num_elements=1}
%addr_byte3_const = ddl.get_external_constant(%type_fp32){name="addr_byte3", num_elements=1}

// %lsbgate_const = ddl.define_constant(%type_fp32){value=[0x0000FFFF], name="lsbgate"}
// %carrydiv_const = ddl.define_constant(%type_fp32){value=[0x37800000], name="carrydiv"}
// %carrymul_const = ddl.define_constant(%type_fp32){value=[0x47800000], name="carrymul"}

%lsbgate_const = ddl.define_constant(%type_fp32){value=[0x000000FF], name="lsbgate"}
%carrydiv_const = ddl.define_constant(%type_fp32){value=[0x3B800000], name="carrydiv"}
%carrymul_const = ddl.define_constant(%type_fp32){value=[0x43800000], name="carrymul"}

%zero_const = ddl.operand_constant {name="0.0"}
%one_const = ddl.operand_constant {name="1.0"}

// allocate space: lx
%allocate_handler_input_lx = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"} 
%allocate_handler_input2_lx = ddl.get_external_data_transfer_allocation (%inptensor2) {memory="lx", data_connect="l3_lx_input2"}
%allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_input"}
%allocate_handler_input_lx_uint32 = ddl.get_external_data_transfer_allocation (%inptensor_uint32) {memory="lx", data_connect="l3_lx_input"} 
%allocate_handler_output_lx_uint32 = ddl.get_external_data_transfer_allocation (%outtensor_uint32) { memory="lx", data_connect="lxsu_input"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  %ffff_sfp_allocation = ddl.allocate(%ffff_const) {memory="sfplrf"} 
  %src_ffff_const = ddl.unit(%ffff_const) {unit="constant", data_connect= "ffff_const_connect"} 
  %dst_ffff_sfp = ddl.unit(%ffff_const, %ffff_sfp_allocation) {unit="sfp", data_connect= "ffff_sfp_lrf"}

  %zero_sfp_allocation = ddl.allocate(%zero_const_opaque) {memory="sfplrf"} 
  %src_zero_const_opaque = ddl.unit(%zero_const_opaque) {unit="constant", data_connect= "zero_const_opaque_connect"} 
  %dst_zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}

  %maskone_sfp_allocation = ddl.allocate(%maskone_const) {memory="sfplrf"} 
  %src_maskone_const = ddl.unit(%maskone_const) {unit="constant", data_connect= "maskone_const_connect"} 
  %dst_maskone_sfp = ddl.unit(%maskone_const, %maskone_sfp_allocation) {unit="sfp", data_connect= "maskone_sfp_lrf"}

  %gelu_const_A_allocation = ddl.allocate(%gelu_const_A) {memory="sfplrf"}
  %gelu_const_B_allocation = ddl.allocate(%gelu_const_B) {memory="sfplrf"}

  %clip_min_const_sfp_allocation = ddl.allocate(%clip_min_const) {memory="sfplrf"}
  %clip_max_const_sfp_allocation = ddl.allocate(%clip_max_const) {memory="sfplrf"}

  // idx32toaddr
  %src_stride_byte0_const = ddl.unit(%stride_byte0_const) {unit="constant", data_connect="stride_byte0_const_connect"}
  %src_stride_byte1_const = ddl.unit(%stride_byte1_const) {unit="constant", data_connect="stride_byte1_const_connect"}
  %src_stride_byte2_const = ddl.unit(%stride_byte2_const) {unit="constant", data_connect="stride_byte2_const_connect"}
  %src_stride_byte3_const = ddl.unit(%stride_byte3_const) {unit="constant", data_connect="stride_byte3_const_connect"}
  %src_addr_byte0_const = ddl.unit(%addr_byte0_const) {unit="constant", data_connect="addr_byte0_cons_connect"}
  %src_addr_byte1_const = ddl.unit(%addr_byte1_const) {unit="constant", data_connect="addr_byte1_cons_connect"}
  %src_addr_byte2_const = ddl.unit(%addr_byte2_const) {unit="constant", data_connect="addr_byte2_cons_connect"}
  %src_addr_byte3_const = ddl.unit(%addr_byte3_const) {unit="constant", data_connect="addr_byte3_cons_connect"}
  %src_lsbgate_const = ddl.unit(%lsbgate_const) {unit="constant", data_connect="lsbgate_const_connect"}
  %src_carrydiv_const = ddl.unit(%carrydiv_const) {unit="constant", data_connect="carrydiv_const_connect"}
  %src_carrymul_const = ddl.unit(%carrymul_const) {unit="constant", data_connect="carrymul_const_connect"}

  %stride_byte0_const_sfp_allocation = ddl.allocate(%stride_byte0_const) {memory="sfplrf"}
  %stride_byte1_const_sfp_allocation = ddl.allocate(%stride_byte1_const) {memory="sfplrf"}
  %stride_byte2_const_sfp_allocation = ddl.allocate(%stride_byte2_const) {memory="sfplrf"}
  %stride_byte3_const_sfp_allocation = ddl.allocate(%stride_byte3_const) {memory="sfplrf"}

  %addr_byte0_const_pe_allocation = ddl.allocate(%addr_byte0_const) {memory="pelrf"}
  %addr_byte1_const_pe_allocation = ddl.allocate(%addr_byte1_const) {memory="pelrf"}
  %addr_byte2_const_pe_allocation = ddl.allocate(%addr_byte2_const) {memory="pelrf"}
  %addr_byte3_const_pe_allocation = ddl.allocate(%addr_byte3_const) {memory="pelrf"}

  %lsbgate_const_sfp_allocation = ddl.allocate(%lsbgate_const) {memory="sfplrf"}
  %carrydiv_const_sfp_allocation = ddl.allocate(%carrydiv_const) {memory="sfplrf"}
  %carrymul_const_sfp_allocation = ddl.allocate(%carrymul_const) {memory="sfplrf"}

  %stride_byte0_const_sfp = ddl.unit(%stride_byte0_const, %stride_byte0_const_sfp_allocation) {unit="sfp", data_connect="stride_byte0_sfp_lrf"}
  %stride_byte1_const_sfp = ddl.unit(%stride_byte1_const, %stride_byte1_const_sfp_allocation) {unit="sfp", data_connect="stride_byte1_sfp_lrf"}
  %stride_byte2_const_sfp = ddl.unit(%stride_byte2_const, %stride_byte2_const_sfp_allocation) {unit="sfp", data_connect="stride_byte2_sfp_lrf"}
  %stride_byte3_const_sfp = ddl.unit(%stride_byte3_const, %stride_byte3_const_sfp_allocation) {unit="sfp", data_connect="stride_byte3_sfp_lrf"}

  %addr_byte0_const_pe = ddl.unit(%addr_byte0_const, %addr_byte0_const_pe_allocation) {unit="pe", data_connect="addr_byte0_const_pe_lrf"}
  %addr_byte1_const_pe = ddl.unit(%addr_byte1_const, %addr_byte1_const_pe_allocation) {unit="pe", data_connect="addr_byte1_const_pe_lrf"}
  %addr_byte2_const_pe = ddl.unit(%addr_byte2_const, %addr_byte2_const_pe_allocation) {unit="pe", data_connect="addr_byte2_const_pe_lrf"}
  %addr_byte3_const_pe = ddl.unit(%addr_byte3_const, %addr_byte3_const_pe_allocation) {unit="pe", data_connect="addr_byte3_const_pe_lrf"}

  %lsbgate_const_sfp = ddl.unit(%lsbgate_const, %lsbgate_const_sfp_allocation) {unit="sfp", data_connect="lsbgate_sfp_lrf"}
  %carrydiv_const_sfp = ddl.unit(%carrydiv_const, %carrydiv_const_sfp_allocation) {unit="sfp", data_connect="carrydiv_sfp_lrf"}
  %carrymul_const_sfp = ddl.unit(%carrymul_const, %carrymul_const_sfp_allocation) {unit="sfp", data_connect="carrymul_sfp_lrf"}

  ddl.if (%exp_op) {
    %const_46DC = ddl.define_constant(%type_fp16){value=[0x46dc], name="expVal1"}
    %const_46E2 = ddl.define_constant(%type_fp16){value=[0x46e2], name="expVal2"}
    %const_34C5 = ddl.define_constant(%type_fp16){value=[0x34c5], name="expVal3"}
    %const_2121 = ddl.define_constant(%type_fp16){value=[0x2121], name="expVal4"}
    %const_3E00 = ddl.define_constant(%type_fp16){value=[0x3e00], name="expVal5"}

    %const_46DC_sfp_alloc = ddl.allocate(%const_46DC) {memory="sfplrf"} 
    %const_46E2_sfp_alloc = ddl.allocate(%const_46E2) {memory="sfplrf"} 
    %const_34C5_sfp_alloc = ddl.allocate(%const_34C5) {memory="sfplrf"} 
    %const_2121_sfp_alloc = ddl.allocate(%const_2121) {memory="sfplrf"} 
    %const_3E00_sfp_alloc = ddl.allocate(%const_3E00) {memory="sfplrf"} 


    %src_const_46DC = ddl.unit(%const_46DC) {unit="constant", data_connect= "const_46DC_connect"} 
    %dst_expconst1_sfp = ddl.unit(%const_46DC, %const_46DC_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const1_lrf"}
    ddl.data_transfer(%src_const_46DC, [%dst_expconst1_sfp]) {}
    %src_const_46E2 = ddl.unit(%const_46E2) {unit="constant", data_connect= "const_46E2_connect"} 
    %dst_expconst2_sfp = ddl.unit(%const_46E2, %const_46E2_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const2_lrf"}
    ddl.data_transfer(%src_const_46E2, [%dst_expconst2_sfp]) {}
    %src_const_34C5 = ddl.unit(%const_34C5) {unit="constant", data_connect= "const_34C5_connect"} 
    %dst_expconst3_sfp = ddl.unit(%const_34C5, %const_34C5_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const3_lrf"}
    ddl.data_transfer(%src_const_34C5, [%dst_expconst3_sfp]) {}
    %src_const_2121 = ddl.unit(%const_2121) {unit="constant", data_connect= "const_2121_connect"} 
    %dst_expconst4_sfp = ddl.unit(%const_2121, %const_2121_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const4_lrf"}
    ddl.data_transfer(%src_const_2121, [%dst_expconst4_sfp]) {}
    %src_const_3E00 = ddl.unit(%const_3E00) {unit="constant", data_connect= "const_3E00_connect"} 
    %dst_expconst5_sfp = ddl.unit(%const_3E00, %const_3E00_sfp_alloc) {unit="sfp", data_connect= "exp_sfp_const5_lrf"}
    ddl.data_transfer(%src_const_3E00, [%dst_expconst5_sfp]) {}

    %eps_sfp_allocation = ddl.allocate(%eps_const_internal_fp32) {memory="sfplrf"}
    %src_eps_const_internal_fp32 = ddl.unit(%eps_const_internal_fp32) {unit="constant", data_connect= "eps_const_connect_internal_fp32"}
    %dst_eps_sfp = ddl.unit(%eps_const_internal_fp32, %eps_sfp_allocation) {unit="sfp", data_connect= "eps_sfp_lrf"}
    ddl.data_transfer(%src_eps_const_internal_fp32, [%dst_eps_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

        ddl.opaque(%outtensor, %const_46DC_sfp_alloc, %const_46E2_sfp_alloc, %const_34C5_sfp_alloc, %const_2121_sfp_alloc, %const_3E00_sfp_alloc, %eps_sfp_allocation)
          {unit="sfp", op="EXP", input_output_registers=["c0", "c1", "c2", "c3", "c4", "c5"], internal_registers=["t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll"],
          max_unroll_factor=4, params={"in0"="lxlu", "out0"="result"},
          input_data_connects=["exp_sfp_const1_lrf", "exp_sfp_const2_lrf", "exp_sfp_const3_lrf", "exp_sfp_const4_lrf", "exp_sfp_const5_lrf", "sfp_lx_input", "eps_sfp_lrf"],
          output_data_connects=["sfp_output"]}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  %silu_or_sigmoid = ddl.condition_or(%silu_op, %sigmoid_op)

  ddl.if(%silu_or_sigmoid) {
    %const_46DC = ddl.define_constant(%type_fp16){value=[0x46dc], name="expVal1"}
    %const_46E2 = ddl.define_constant(%type_fp16){value=[0x46e2], name="expVal2"}
    %const_34C5 = ddl.define_constant(%type_fp16){value=[0x34c5], name="expVal3"}
    %const_2121 = ddl.define_constant(%type_fp16){value=[0x2121], name="expVal4"}
    %const_3E00 = ddl.define_constant(%type_fp16){value=[0x3e00], name="expVal5"}
    %const_0 = ddl.define_constant(%type_fp16){value=[0], name="zero", num_elements=1}

    %const_46DC_sfp_alloc = ddl.allocate(%const_46DC) {memory="sfplrf"} 
    %const_46E2_sfp_alloc = ddl.allocate(%const_46E2) {memory="sfplrf"} 
    %const_34C5_sfp_alloc = ddl.allocate(%const_34C5) {memory="sfplrf"} 
    %const_2121_sfp_alloc = ddl.allocate(%const_2121) {memory="sfplrf"} 
    %const_3E00_sfp_alloc = ddl.allocate(%const_3E00) {memory="sfplrf"} 
    %const_0_sfp_alloc = ddl.allocate(%const_0) {memory="sfplrf"}

    %src_const_46DC = ddl.unit(%const_46DC) {unit="constant", data_connect= "const_46DC_connect"} 
    %dst_sigmoidconst1_sfp = ddl.unit(%const_46DC, %const_46DC_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const1_lrf"}
    ddl.data_transfer(%src_const_46DC, [%dst_sigmoidconst1_sfp]) {}
    %src_const_46E2 = ddl.unit(%const_46E2) {unit="constant", data_connect= "const_46E2_connect"} 
    %dst_sigmoidconst2_sfp = ddl.unit(%const_46E2, %const_46E2_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const2_lrf"}
    ddl.data_transfer(%src_const_46E2, [%dst_sigmoidconst2_sfp]) {}
    %src_const_34C5 = ddl.unit(%const_34C5) {unit="constant", data_connect= "const_34C5_connect"} 
    %dst_sigmoidconst3_sfp = ddl.unit(%const_34C5, %const_34C5_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const3_lrf"}
    ddl.data_transfer(%src_const_34C5, [%dst_sigmoidconst3_sfp]) {}
    %src_const_2121 = ddl.unit(%const_2121) {unit="constant", data_connect= "const_2121_connect"} 
    %dst_sigmoidconst4_sfp = ddl.unit(%const_2121, %const_2121_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const4_lrf"}
    ddl.data_transfer(%src_const_2121, [%dst_sigmoidconst4_sfp]) {}
    %src_const_3E00 = ddl.unit(%const_3E00) {unit="constant", data_connect= "const_3E00_connect"} 
    %dst_sigmoidconst5_sfp = ddl.unit(%const_3E00, %const_3E00_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const5_lrf"}
    ddl.data_transfer(%src_const_3E00, [%dst_sigmoidconst5_sfp]) {}
    %src_const_0 = ddl.unit(%const_0) {unit="constant", data_connect= "const_0_connect"} 
    %dst_sigmoidconst0_sfp = ddl.unit(%const_0, %const_0_sfp_alloc) {unit="sfp", data_connect= "sigmoid_sfp_const0_lrf"}
    ddl.data_transfer(%src_const_0, [%dst_sigmoidconst0_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 

        %sfp_lrf_allocation = ddl.allocate(%inptensor) {memory="sfplrf"} 
        %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

        %sig_result_alloc = ddl.allocate(%outtensor){memory = "sfplrf"}
        ddl.if(%sigmoid_op) {
          ddl.opaque(%outtensor, %const_46DC_sfp_alloc, %const_46E2_sfp_alloc, %const_34C5_sfp_alloc, %const_2121_sfp_alloc,  %const_3E00_sfp_alloc, %const_0_sfp_alloc, %sig_result_alloc)
              {unit="sfp", op="SIGMOID", input_output_registers=["c0", "c1", "c2", "c3", "c4", "c5", "t1_unroll"], internal_registers=["p0_unroll", "p1_unroll",  "t0_unroll", "t2_unroll"],
              max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
              input_data_connects=["sigmoid_sfp_const1_lrf", "sigmoid_sfp_const2_lrf", "sigmoid_sfp_const3_lrf", "sigmoid_sfp_const4_lrf", "sigmoid_sfp_const5_lrf", "sigmoid_sfp_const0_lrf", "sfp_lx_input"],
              output_data_connects=["sfp_output", "sig_sfp_lrf"]}
        } else {  // silu
          ddl.opaque(%outtensor, %const_46DC_sfp_alloc, %const_46E2_sfp_alloc, %const_34C5_sfp_alloc, %const_2121_sfp_alloc,  %const_3E00_sfp_alloc, %const_0_sfp_alloc, %sig_result_alloc)
              {unit="sfp", op="SIGMOID", input_output_registers=["c0", "c1", "c2", "c3", "c4", "c5", "t1_unroll"], internal_registers=["p0_unroll", "p1_unroll",  "t0_unroll", "t2_unroll"],
              max_unroll_factor=2, params={"in0"="lxlu", "out0"="no"},
              input_data_connects=["sigmoid_sfp_const1_lrf", "sigmoid_sfp_const2_lrf", "sigmoid_sfp_const3_lrf", "sigmoid_sfp_const4_lrf", "sigmoid_sfp_const5_lrf", "sigmoid_sfp_const0_lrf", "sfp_lx_input"],
              output_data_connects=["sfp_output", "sig_sfp_lrf"]}
            
          // send another copy of the input
          ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

          ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
            %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
            %sfp_src01 = ddl.unit(%outtensor, %sig_result_alloc) {unit="sfp", data_connect="sig_sfp_lrf"}
            %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
            ddl.compute([%sfp_src00, %sfp_src01], [%sfp_dst00]) {computetype="FMUL", unit="sfp"}
          }
        }

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
  ddl.if(%abs_op) {
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 

        %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

        // neg -(A*1-0) = -A
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src00, %zero_const], [%sfp_dst00]) {computetype="FABSMAX", unit="sfp"}
        }

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      } 
    }
  }
  ddl.if(%neg_op) {
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {

        %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

        // neg -(A*1-0) = -A
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_src00, %one_const, %zero_const], [%sfp_dst00]) {computetype="FNMS", unit="sfp"}
        }

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
  // ddl.if(%fastexp_op) {
  //   %exp_sfp_allocation = ddl.allocate(%fastexp_const) {memory="sfplrf"}
  //   %src_fastexp_const = ddl.unit(%fastexp_const) {unit="constant", data_connect= "fastexp_const_connect"} 
  //   %exp_sfp_lrf = ddl.unit(%fastexp_const, %exp_sfp_allocation) {unit="sfp", data_connect= "exp_sfp_lrf"}
  //   ddl.data_transfer(%src_fastexp_const, [%exp_sfp_lrf]) {}
  //   ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
  //     ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 

  //       %src_inp_lx = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
  //       %dst_inp_sfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}
  //       ddl.data_transfer(%src_inp_lx, [%dst_inp_sfp]) {}

  //       %sfp_lrf_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
  //       %outp_sfp_lrf = ddl.unit(%outtensor, %sfp_lrf_allocation) {unit="sfp", data_connect="sfp_output_intermediate"}
  //       ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2, %d#3, %d#4, %d#5){} { 
  //         %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
  //         ddl.compute([%sfp_src00, %exp_sfp_lrf], [%outp_sfp_lrf]) {computetype="FMUL", unit="sfp"}
  //       }

  //       ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2, %d#3, %d#4, %d#5){} { 
  //         %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
  //         ddl.compute([%outp_sfp_lrf], [%sfp_dst00]) {computetype="ICVT", unit="sfp", mode=7}
  //       }

  //       // sfp-lx
  //       %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
  //       %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
  //       ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
  //     } 
  //   }
  // }
  ddl.if (%gelu_op) {
    %src_gelu_const_A = ddl.unit(%gelu_const_A) {unit="constant", data_connect= "gelu_const_A_connect"}
    %src_gelu_const_B = ddl.unit(%gelu_const_B) {unit="constant", data_connect= "gelu_const_B_connect"}
    %dst_gelu_const_A = ddl.unit(%gelu_const_A, %gelu_const_A_allocation) {unit="sfp", data_connect= "gelu_const_A_lrf"}
    %dst_gelu_const_B = ddl.unit(%gelu_const_B, %gelu_const_B_allocation) {unit="sfp", data_connect= "gelu_const_B_lrf"}
    ddl.data_transfer(%src_gelu_const_A, [%dst_gelu_const_A]) {}
    ddl.data_transfer(%src_gelu_const_B, [%dst_gelu_const_B]) {}

    // Compute:
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        // lx-sfp
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        ddl.opaque(%outtensor, %gelu_const_A_allocation, %gelu_const_B_allocation)
            {unit="sfp", op="GELU", input_output_registers=["c1", "c2"], internal_registers=["p0_unroll", "t0_unroll", "t1_unroll"],
            max_unroll_factor=4, params={"in0"="lxlu", "out0"="result"},
            input_data_connects=["gelu_const_A_lrf", "gelu_const_B_lrf", "sfp_lx_input"],
            output_data_connects=["sfp_output"]}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  // Tanh
  ddl.if(%tanh_op) {
    // Compute:
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        // lx-sfp
        %sfp_inp_lrf_alloc = ddl.allocate(%inptensor) {memory = "sfplrf"}
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %sfp_src00 = ddl.unit(%inptensor, %sfp_inp_lrf_alloc) {unit="sfp", data_connect="sfp_lx_input"}
        ddl.data_transfer(%src_inp_lxsfp, [%sfp_src00]) {}

        %sfp_allocation00 = ddl.allocate(%interim_tensor0) {memory="sfplrf"}
        %sfp_allocation11 = ddl.allocate(%interim_tensor1) {memory="sfplrf"}
        %sfp_out00 = ddl.unit(%interim_tensor0, %sfp_allocation00) {unit="sfp", data_connect="sfp_outtensor_lrf"}
        %sfp_out11 = ddl.unit(%interim_tensor1, %sfp_allocation11) {unit="sfp", data_connect="sfp_outtensor_lrf"}
        %sfp_dst = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}

        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          ddl.compute([%sfp_src00], [%sfp_out00]) {computetype="FEST", unit="sfp", mode=8}
        }
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          ddl.compute([%sfp_src00], [%sfp_out11]) {computetype="FEST", unit="sfp", mode=9}
        }
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          ddl.compute([%sfp_out00, %sfp_src00, %sfp_out11], [%sfp_dst]) {computetype="MACC", unit="sfp"}
        }
        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
  // Reciprocal
  ddl.if (%reciprocal_op) {
    // Constants:
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}

    %plus1_sfp_allocation = ddl.allocate(%plus1_const) {memory="sfplrf"} 
    %src_plus1_const = ddl.unit(%plus1_const) {unit="constant", data_connect= "plus1_const_connect"} 
    %dst_plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
    ddl.data_transfer(%src_plus1_const, [%dst_plus1_sfp]) {}

    %minus1_sfp_allocation = ddl.allocate(%minus1_const) {memory="sfplrf"} 
    %src_minus1_const = ddl.unit(%minus1_const) {unit="constant", data_connect= "minus1_const_connect"} 
    %dst_minus1_sfp = ddl.unit(%minus1_const, %minus1_sfp_allocation) {unit="sfp", data_connect= "minus1_sfp_lrf"}
    ddl.data_transfer(%src_minus1_const, [%dst_minus1_sfp]) {}

    // Compute:
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
        
        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        ddl.opaque(%outtensor, %zero_sfp_allocation, %ffff_sfp_allocation, %minus1_sfp_allocation, %plus1_sfp_allocation)
            {unit="sfp", op="RECIPROCAL", input_output_registers=["c1", "c2", "c3", "c4"], internal_registers=["p0_unroll", "t0_unroll", "t2_unroll", "t4_unroll", "t6_unroll"],
            max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
            input_data_connects=["zero_sfp_lrf", "ffff_sfp_lrf", "minus1_sfp_lrf", "sfp_lx_input"],
            output_data_connects=["sfp_output"]}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  ddl.if (%layernormscale_op) {
    // Constants:
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
    ddl.data_transfer(%src_maskone_const, [%dst_maskone_sfp]) {}

    %eps_sfp_allocation = ddl.allocate(%eps_const) {memory="sfplrf"} 
    %src_eps_const = ddl.unit(%eps_const) {unit="constant", data_connect= "eps_const_connect"} 
    %src_eps_const_internal_fp32 = ddl.unit(%eps_const_internal_fp32) {unit="constant", data_connect= "eps_const_connect_internal_fp32"} 
    %dst_eps_sfp = ddl.unit(%eps_const, %eps_sfp_allocation) {unit="sfp", data_connect= "eps_sfp_lrf"}
    ddl.data_transfer(%src_eps_const, [%dst_eps_sfp]) {}

    // Compute:
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
    ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
    
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
        // send E[x]
        %src_ex_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_ex_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_ex"}
        ddl.data_transfer(%src_ex_lxsfp, [%dst_ex_lxsfp]) {}
        // send E[x^2]
        %src_ex2_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input", stick_replicated_dim_offset_elements=8}
        %dst_ex2_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_ex2"}
        // TEMP: disabling stickpacking opt
        // %src_ex2_lxsfp = ddl.unit(%inptensor2, %allocate_handler_input2_lx) {unit="lxlu", data_connect="l3_lx_input2"}
        // %dst_ex2_lxsfp = ddl.unit(%inptensor2) {unit="sfp", data_connect="sfp_lx_ex2"}
        ddl.data_transfer(%src_ex2_lxsfp, [%dst_ex2_lxsfp]) {}
      }

      ddl.opaque(%outtensor, %eps_sfp_allocation, %zero_sfp_allocation, %maskone_sfp_allocation, %ffff_sfp_allocation)
          {unit="sfp", op="LAYERNORMSCALE", input_output_registers=["c1", "c2", "c3", "c4"], internal_registers=["t0_unroll", "t2_unroll", "t3_unroll", "t4_unroll", "t5_unroll", "t6_unroll"],
          max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
          input_data_connects=["eps_sfp_lrf", "zero_sfp_lrf", "maskone_sfp_lrf", "ffff_sfp_lrf", "sfp_lx_ex", "sfp_lx_ex2" ],
          output_data_connects=["sfp_output"]}

      // sfp-lx
      %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
      %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
      ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
  
  ddl.if(%sqrt_op) {
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
    ddl.data_transfer(%src_maskone_const, [%dst_maskone_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {

        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        ddl.opaque(%outtensor, %zero_sfp_allocation, %maskone_sfp_allocation, %ffff_sfp_allocation)
                {unit="sfp", op="SQRT", input_output_registers=["c0", "c1", "c2"],
                internal_registers=["p0_unroll", "t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll", "t4_unroll"],
                max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
                input_data_connects=["zero_sfp_lrf", "maskone_sfp_lrf", "ffff_sfp_lrf", "sfp_lx_input"],
                output_data_connects=["sfp_output"] }

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
  
  ddl.if(%rsqrt_op) {
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
    ddl.data_transfer(%src_maskone_const, [%dst_maskone_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        ddl.opaque(%outtensor, %zero_sfp_allocation, %maskone_sfp_allocation, %ffff_sfp_allocation)
                {unit="sfp", op="RSQRT", input_output_registers=["c0", "c1", "c2"],
                internal_registers=["p0_unroll", "t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll", "t4_unroll"],
                max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"},
                input_data_connects=["zero_sfp_lrf", "maskone_sfp_lrf", "ffff_sfp_lrf", "sfp_lx_input"],
                output_data_connects=["sfp_output"]}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  ddl.if(%int32idxtoaddr_op) {
    ddl.data_transfer(%src_stride_byte0_const, [%stride_byte0_const_sfp]) {}
    ddl.data_transfer(%src_stride_byte1_const, [%stride_byte1_const_sfp]) {}
    ddl.data_transfer(%src_stride_byte2_const, [%stride_byte2_const_sfp]) {}
    ddl.data_transfer(%src_stride_byte3_const, [%stride_byte3_const_sfp]) {}

    ddl.data_transfer(%src_addr_byte0_const, [%addr_byte0_const_pe]) {}
    ddl.data_transfer(%src_addr_byte1_const, [%addr_byte1_const_pe]) {}
    ddl.data_transfer(%src_addr_byte2_const, [%addr_byte2_const_pe]) {}
    ddl.data_transfer(%src_addr_byte3_const, [%addr_byte3_const_pe]) {}

    ddl.data_transfer(%src_lsbgate_const, [%lsbgate_const_sfp]) {}
    ddl.data_transfer(%src_carrydiv_const, [%carrydiv_const_sfp]) {}
    ddl.data_transfer(%src_carrymul_const, [%carrymul_const_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor_uint32, %allocate_handler_input_lx_uint32) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor_uint32) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}
       
        %dst_addr_byte0_sfp_via_pe = ddl.unit(%addr_byte0_const) {unit="sfp", data_connect="addr_byte_input"}
        ddl.data_transfer(%addr_byte0_const_pe, [%dst_addr_byte0_sfp_via_pe]) {}
        %dst_addr_byte1_sfp_via_pe = ddl.unit(%addr_byte1_const) {unit="sfp", data_connect="addr_byte_input"}
        ddl.data_transfer(%addr_byte1_const_pe, [%dst_addr_byte1_sfp_via_pe]) {}
        %dst_addr_byte2_sfp_via_pe = ddl.unit(%addr_byte2_const) {unit="sfp", data_connect="addr_byte_input"}
        ddl.data_transfer(%addr_byte2_const_pe, [%dst_addr_byte2_sfp_via_pe]) {}
        %dst_addr_byte3_sfp_via_pe = ddl.unit(%addr_byte3_const) {unit="sfp", data_connect="addr_byte_input"}
        ddl.data_transfer(%addr_byte3_const_pe, [%dst_addr_byte3_sfp_via_pe]) {}

        ddl.opaque(%outtensor_uint32, %stride_byte0_const_sfp_allocation, %stride_byte1_const_sfp_allocation, %stride_byte2_const_sfp_allocation, %stride_byte3_const_sfp_allocation,
                   %lsbgate_const_sfp_allocation, %carrydiv_const_sfp_allocation, %carrymul_const_sfp_allocation)
                {unit="sfp", op="idx32toaddr", input_output_registers=["stride_byte0", "stride_byte1", "stride_byte2", "stride_byte3", "lsbgate", "carrydiv", "carrymul"],
                internal_registers=["carry_unroll", "a0_unroll", "a1_unroll", "a2_unroll", "a3_unroll", "t_result_unroll", "t_curr_unroll"],
                max_unroll_factor=2, params={"in0"="lxlu", "in2"="pe", "out0"="result"},
                input_data_connects=["sfp_lx_input", "addr_byte_input", "stride_byte0_sfp_lrf", "stride_byte1_sfp_lrf", "stride_byte2_sfp_lrf", "stride_byte3_sfp_lrf",
                                     "lsbgate_sfp_lrf", "carrydiv_sfp_lrf", "carrymul_sfp_lrf"],
                output_data_connects=["sfp_output"]}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor_uint32) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor_uint32, %allocate_handler_output_lx_uint32) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  ddl.if (%relu_op){
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {

        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
        %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
        ddl.compute([%sfp_src00, %zero_const], [%sfp_dst00]) {computetype="FMAX", unit="sfp"}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}

      }
    }
  }

  %identity_or_shuffle = ddl.condition_or(%identity_op, %shuffle_op)  // shuffle is an alias of identity
  ddl.if (%identity_or_shuffle){
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      ddl.loop (%b_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {

        // lx-sfp
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
        %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
        ddl.compute([%sfp_src00, %zero_const], [%sfp_dst00]) {computetype="OR", unit="sfp"}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  ddl.if (%floor_op){
    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {

        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
        %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="FLOOR", unit="sfp"}

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }

  ddl.if (%clip_op){
    %src_clip_min_const = ddl.unit(%clip_min_const) {unit="constant", data_connect= "clip_min_const_connect"} 
    %clip_min_const_sfp = ddl.unit(%clip_min_const, %clip_min_const_sfp_allocation) {unit="sfp", data_connect= "clip_min_const_lrf"}
    ddl.data_transfer(%src_clip_min_const, [%clip_min_const_sfp]) {}
    %src_clip_max_const = ddl.unit(%clip_max_const) {unit="constant", data_connect= "clip_max_const_connect"} 
    %clip_max_const_sfp = ddl.unit(%clip_max_const, %clip_max_const_sfp_allocation) {unit="sfp", data_connect= "clip_max_const_lrf"}
    ddl.data_transfer(%src_clip_max_const, [%clip_max_const_sfp]) {}

    ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
        // lx-sfp 
        %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxsfp = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_lx_input"}  // no need to allocate input in sfp-register as its done on the fly
        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

        %output_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
        %sfp_out_lrf = ddl.unit(%outtensor, %output_sfp_allocation) {unit="sfp", data_connect="sfp_lrf_out"}

        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          %sfp_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_lx_input"}
          ddl.compute([%sfp_src00, %clip_min_const_sfp], [%sfp_out_lrf]) {computetype="FMAX", unit="sfp"}
        }
        ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          %sfp_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}
          ddl.compute([%sfp_out_lrf, %clip_max_const_sfp], [%sfp_dst00]) {computetype="FMIN", unit="sfp"}
        }

        // sfp-lx
        %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
        %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
      }
    }
  }
}

ddl.transformations {
  ddl.if(%layernormscale_op) {
    ddl.disable_transfer_promotion()
  }
}

}
