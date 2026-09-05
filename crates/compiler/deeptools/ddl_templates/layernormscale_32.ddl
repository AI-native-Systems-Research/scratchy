//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%d:6 = ddl.dimension {} : index, index, index, index, index, index
%slice_layout = ddl.layout() {is_order_fixed=false} 
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}
%type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
%type_fp32 = ddl.type {data_type="IEEE_FP32", bit_width=32}
%inptensor, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index, index
%interim_tensor0 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%interim_tensor1 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

%layernormscale_op = ddl.operation_bind([%type_fp16], [%inptensor], [%outtensor]) {opFuncName="layernormscale", required=false}

ddl.constraint(%inptensor, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor, %outtensor) {property = "stick", cmp = "equal"}
ddl.constraint(%layernormscale_op) {min_num_valid = 1, max_num_valid = 1}
ddl.constraint() {min_num_cores = 1}

%ffff_const = ddl.define_constant(%type_fp16){value=[0xFFFF], name="ffff"}
%zero_const_opaque = ddl.define_constant(%type_fp16){value=[0], name="zero"}
%plus1_const = ddl.define_constant(%type_fp16){value=[0x3E00], name="plus1"}
%minus1_const = ddl.define_constant(%type_fp16){value=[0xBE00], name="minus1"}
%eps_const = ddl.get_external_constant(%type_fp16){name="eps", num_elements=1}
%maskone_const = ddl.define_constant(%type_fp16){value=[0x0001], name="maskone"}

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

  %ffff_sfp_allocation = ddl.allocate(%ffff_const) {memory="sfplrf"} 
  %src_ffff_const = ddl.unit(%ffff_const) {unit="constant", data_connect= "ffff_const_connect"} 
  %dst_ffff_sfp = ddl.unit(%ffff_const, %ffff_sfp_allocation) {unit="sfp", data_connect= "ffff_sfp_lrf"}

  %zero_sfp_allocation = ddl.allocate(%zero_const_opaque) {memory="sfplrf"} 
  %src_zero_const_opaque = ddl.unit(%zero_const_opaque) {unit="constant", data_connect= "zero_const_opaque_connect"} 
  %dst_zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}

  %maskone_sfp_allocation = ddl.allocate(%maskone_const) {memory="sfplrf"} 
  %src_maskone_const = ddl.unit(%maskone_const) {unit="constant", data_connect= "maskone_const_connect"} 
  %dst_maskone_sfp = ddl.unit(%maskone_const, %maskone_sfp_allocation) {unit="sfp", data_connect= "maskone_sfp_lrf"}

  ddl.if (%layernormscale_op) {
    // Constants:
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
    ddl.data_transfer(%src_maskone_const, [%dst_maskone_sfp]) {}

    %eps_sfp_allocation = ddl.allocate(%eps_const) {memory="sfplrf"} 
    %src_eps_const = ddl.unit(%eps_const) {unit="constant", data_connect= "eps_const_connect"} 
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
        ddl.data_transfer(%src_ex2_lxsfp, [%dst_ex2_lxsfp]) {}
      }

      ddl.opaque(%outtensor, %eps_sfp_allocation, %zero_sfp_allocation, %maskone_sfp_allocation, %ffff_sfp_allocation)
          {unit="sfp", op="LAYERNORMSCALE32", input_output_registers=["c1", "c2", "c3", "c4"], internal_registers=["t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll", "t4_unroll", "t5_unroll", "t6_unroll"],
          max_unroll_factor=1, params={"in0"="lxlu", "out0"="result"},
          input_data_connects=["eps_sfp_lrf", "zero_sfp_lrf", "maskone_sfp_lrf", "ffff_sfp_lrf", "sfp_lx_ex", "sfp_lx_ex2" ],
          output_data_connects=["sfp_output"]}

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
