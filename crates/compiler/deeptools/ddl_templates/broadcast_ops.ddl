//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%d:6 = ddl.dimension {} : index, index, index, index, index, index
%slice_layout = ddl.layout() {is_order_fixed=false}
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
%type_fp32 = ddl.type {data_type="IEEE_FP32"} 
%type_bool = ddl.type {data_type="BOOL"}
%type_uint32 = ddl.type {data_type="SENUINT32"}

%inp1tensor, %inp2tensor, %inp3tensor, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16, %type_fp32]) : index, index, index, index
%inp1tensor_uint32, %inp2tensor_uint32, %outtensor_uint32 = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_uint32]) : index, index, index
%state_reg = ddl.internal_tensor(%outtensor, [%type_bool]) : index
// for sinkcorrectionfactor
%internal_tensor1, %internal_tensor2, %internal_tensor3 = ddl.internal_tensor(%inp1tensor, [%type_fp16, %type_fp32]) : index, index, index

%add_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="add", required=false}
%sub_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="sub", required=false}
%mul_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="mul", required=false}
%mul_i32_to_i32_op = ddl.operation_bind([%type_fp32], [%inp1tensor_uint32, %inp2tensor_uint32], [%outtensor_uint32]) {opFuncName="muli32toi32", required=false}
%revsub_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="revsub", required=false}
%biasadd_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="biasadd", required=false}
%stridedadd_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="stridedadd", required=false}
%bn_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor, %inp3tensor], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%where_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor, %inp3tensor], [%outtensor], [%state_reg]) {opFuncName="where3", required=false}
%ge_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="greaterequal", required=false}
%gt_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="greaterthan", required=false}
%le_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="lesserequal", required=false}
%lt_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="lesserthan", required=false}
%eq_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="equal", required=false}
%ne_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor], [%state_reg]) {opFuncName="notequal", required=false}
%fnms_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor, %inp3tensor], [%outtensor]) {opFuncName="fnms", required=false}
%realdiv_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="realdiv", required=false}
%maximum_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="maximum", required=false}
%minimum_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1tensor, %inp2tensor], [%outtensor]) {opFuncName="minimum", required=false}
%sinkcorr_op = ddl.operation_bind([%type_fp16], [%inp1tensor, %inp2tensor, %inp3tensor], [%outtensor], [%internal_tensor1, %internal_tensor2, %internal_tensor3]) {opFuncName="sinkcorrectionfactor", required=false}


ddl.constraint(%inp1tensor, %inp2tensor, %inp3tensor, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inp1tensor, %inp2tensor, %inp3tensor, %outtensor) {property = "stick", cmp = "equal"}
ddl.constraint(%inp1tensor_uint32, %inp2tensor_uint32, %outtensor_uint32) {property = "slice", cmp = "equal"}
ddl.constraint(%inp1tensor_uint32, %inp2tensor_uint32, %outtensor_uint32) {property = "stick", cmp = "equal"}
ddl.constraint(%add_op, %sub_op, %mul_op, %mul_i32_to_i32_op, %revsub_op, %biasadd_op, %stridedadd_op, %bn_op, %where_op, %ge_op, %gt_op, %le_op, %lt_op, %eq_op, %ne_op, %fnms_op, %realdiv_op, %maximum_op, %minimum_op, %sinkcorr_op) {min_num_valid = 1, max_num_valid = 1}
ddl.constraint() {min_num_cores = 1}

// allocate space: lx
%inp1tensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inp1tensor) {memory="lx", data_connect="l3_lx_input1"}
%inp2tensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inp2tensor) {memory="lx", data_connect="l3_lx_input2"}
%inp3tensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inp3tensor) {memory="lx", data_connect="l3_lx_input3"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) {memory="lx", data_connect="lxsu_input"}
%inp1tensor_uint32_lx_allocation = ddl.get_external_data_transfer_allocation (%inp1tensor_uint32) {memory="lx", data_connect="l3_lx_input1"}
%inp2tensor_uint32_lx_allocation = ddl.get_external_data_transfer_allocation (%inp2tensor_uint32) {memory="lx", data_connect="l3_lx_input2"}
%outtensor_uint32_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor_uint32) {memory="lx", data_connect="lxsu_input"}

%zero_const_opaque = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
%plus1_const = ddl.define_constant(%type_fp16) {value=[0x3E00], name="plus1"}
%minus1_const = ddl.define_constant(%type_fp16) {value=[0xBE00], name="minus1"}
%ffff_const_fp16 = ddl.define_constant(%type_fp16) {value=[0xFFFF], name="ffff"}
%ffff_const_fp32 = ddl.define_constant(%type_fp32) {value=[0xFFFFFFFF], name="ffff"}
%ffff_const = ddl.alias_one_constant_of(%ffff_const_fp16, %ffff_const_fp32)
%fastexp_const = ddl.define_constant(%type_fp16){value=[0x54e3], name="fastexpVal"}
%zero_const = ddl.operand_constant {name="0.0"}
%one_const = ddl.operand_constant {name="1.0"}

// mul_i32_to_i32
%lsbgate_const = ddl.define_constant(%type_fp32){value=[0x000000FF], name="lsbgate"}
%carrydiv_8_const = ddl.define_constant(%type_fp32){value=[0x3B800000], name="carrydiv_8"}
%carrymul_8_const = ddl.define_constant(%type_fp32){value=[0x43800000], name="carrymul_8"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleaving_datastage = ddl.datastage {strategy="maximize"}
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  %op_with_interleaving = ddl.condition_or(%realdiv_op, %sinkcorr_op)
  %op_without_interleaving = ddl.condition_not(%op_with_interleaving)

  ddl.if(%op_without_interleaving) {
    // force no interleaving
    ddl.datastage_constraint(%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5) {values=["1"]}
  }

  %zero_sfp_allocation = ddl.allocate(%zero_const_opaque) {memory="sfplrf"}
  %plus1_sfp_allocation = ddl.allocate(%plus1_const) {memory="sfplrf"}
  %ffff_sfp_allocation = ddl.allocate(%ffff_const) {memory="sfplrf"}
  %minus1_sfp_allocation = ddl.allocate(%minus1_const) {memory="sfplrf"}
  %exp_sfp_allocation = ddl.allocate(%fastexp_const) {memory="sfplrf"}
  %fastexp_const_sfp = ddl.unit(%fastexp_const, %exp_sfp_allocation) {unit="sfp", data_connect= "exp_sfp_lrf"}

  // mul_i32_to_i32
  %src_lsbgate_const = ddl.unit(%lsbgate_const) {unit="constant", data_connect="lsbgate_const_connect"}
  %src_carrydiv_8_const = ddl.unit(%carrydiv_8_const) {unit="constant", data_connect="carrydiv_8_const_connect"}
  %src_carrymul_8_const = ddl.unit(%carrymul_8_const) {unit="constant", data_connect="carrymul_8_const_connect"}
  %lsbgate_const_sfp_allocation = ddl.allocate(%lsbgate_const) {memory="sfplrf"}
  %carrydiv_8_const_sfp_allocation = ddl.allocate(%carrydiv_8_const) {memory="sfplrf"}
  %carrymul_8_const_sfp_allocation = ddl.allocate(%carrymul_8_const) {memory="sfplrf"}
  %lsbgate_const_sfp = ddl.unit(%lsbgate_const, %lsbgate_const_sfp_allocation) {unit="sfp", data_connect="lsbgate_sfp_lrf"}
  %carrydiv_8_const_sfp = ddl.unit(%carrydiv_8_const, %carrydiv_8_const_sfp_allocation) {unit="sfp", data_connect="carrydiv_8_sfp_lrf"}
  %carrymul_8_const_sfp = ddl.unit(%carrymul_8_const, %carrymul_8_const_sfp_allocation) {unit="sfp", data_connect="carrymul_8_sfp_lrf"}

  %uses_realdiv = ddl.condition_or(%realdiv_op, %sinkcorr_op)
  %conditional_op = ddl.condition_or(%eq_op, %ge_op, %gt_op, %le_op, %lt_op, %ne_op)

  ddl.if(%conditional_op) {
    %src_zero_const_opaque = ddl.unit(%zero_const_opaque) {unit="constant", data_connect= "zero_const_opaque_connect"}
    %dst_zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
    %src_plus1_const = ddl.unit(%plus1_const) {unit="constant", data_connect= "plus1_const_connect"}
    %dst_plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
    ddl.data_transfer(%src_plus1_const, [%dst_plus1_sfp]) {}
  }
  ddl.if(%uses_realdiv) {
    %src_zero_const_opaque = ddl.unit(%zero_const_opaque) {unit="constant", data_connect= "zero_const_opaque_connect"} 
    %dst_zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
    ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}

    %src_ffff_const = ddl.unit(%ffff_const) {unit="constant", data_connect= "ffff_const_connect"} 
    %dst_ffff_sfp = ddl.unit(%ffff_const, %ffff_sfp_allocation) {unit="sfp", data_connect= "ffff_sfp_lrf"}
    ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}

    %src_plus1_const = ddl.unit(%plus1_const) {unit="constant", data_connect= "plus1_const_connect"} 
    %dst_plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
    ddl.data_transfer(%src_plus1_const, [%dst_plus1_sfp]) {}

    %src_minus1_const = ddl.unit(%minus1_const) {unit="constant", data_connect= "minus1_const_connect"} 
    %dst_minus1_sfp = ddl.unit(%minus1_const, %minus1_sfp_allocation) {unit="sfp", data_connect= "minus1_sfp_lrf"}
    ddl.data_transfer(%src_minus1_const, [%dst_minus1_sfp]) {}
  }
  ddl.if (%sinkcorr_op) {
    %src_fastexp_const = ddl.unit(%fastexp_const) {unit="constant", data_connect= "fastexp_const_connect"} 
    ddl.data_transfer(%src_fastexp_const, [%fastexp_const_sfp]) {}
  }
  ddl.if(%mul_i32_to_i32_op) {
    ddl.data_transfer(%src_lsbgate_const, [%lsbgate_const_sfp]) {}
    ddl.data_transfer(%src_carrydiv_8_const, [%carrydiv_8_const_sfp]) {}
    ddl.data_transfer(%src_carrymul_8_const, [%carrymul_8_const_sfp]) {}
  }
  ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
    ddl.loop (%b_datastage, %interleaving_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      %inp1tensor_sfp_allocation = ddl.allocate(%inp1tensor) {memory="sfplrf"}
      %inp2tensor_sfp_allocation = ddl.allocate(%inp2tensor) {memory="sfplrf"}
      %inp3tensor_sfp_allocation = ddl.allocate(%inp3tensor) {memory="sfplrf"}
      %inp2tensor_uint32_sfp_allocation = ddl.allocate(%inp2tensor_uint32) {memory="sfplrf"}
      %src_inp1_lxsfp = ddl.unit(%inp1tensor, %inp1tensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input1"}
      %src_inp2_lxsfp = ddl.unit(%inp2tensor, %inp2tensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input2"}
      %src_inp3_lxsfp = ddl.unit(%inp3tensor, %inp3tensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input3"}
      %sfp_inp1_fifo = ddl.unit(%inp1tensor) {unit="lxlu", data_connect="sfp_lx_input1"}
      %sfp_inp1 = ddl.unit(%inp1tensor, %inp1tensor_sfp_allocation) {unit="sfp", data_connect="sfp_lx_input1"}
      %sfp_inp2 = ddl.unit(%inp2tensor, %inp2tensor_sfp_allocation) {unit="sfp", data_connect="sfp_lx_input2"}
      %sfp_inp3 = ddl.unit(%inp3tensor, %inp3tensor_sfp_allocation) {unit="sfp", data_connect="sfp_lx_input3"}
      ddl.if (%realdiv_op) {
      	// lx-sfp - inp2tensor - always stream
        %dst_inp2_lxsfp = ddl.unit(%inp2tensor) {unit="sfp", data_connect="sfp_lx_input2"}
        ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          ddl.data_transfer(%src_inp2_lxsfp, [%dst_inp2_lxsfp]) {}
        }
      } else {
        ddl.if(%mul_i32_to_i32_op) {
          // lx-sfp - inp2tensor -- could be migrated up for broadcast reuse via. transformations
          %src_inp2_uint32_lxsfp = ddl.unit(%inp2tensor_uint32, %inp2tensor_uint32_lx_allocation) {unit="lxlu", data_connect="l3_lx_input2"}
          %dst_inp2_uint32_lxsfp = ddl.unit(%inp2tensor_uint32, %inp2tensor_uint32_sfp_allocation) {unit="sfp", data_connect="sfp_lx_input2"} 
          ddl.data_transfer(%src_inp2_uint32_lxsfp, [%dst_inp2_uint32_lxsfp]) {}
        } else {
          // lx-sfp - inp2tensor -- could be migrated up for broadcast reuse via. transformations
          ddl.data_transfer(%src_inp2_lxsfp, [%sfp_inp2]) {}
        }
      }
      // lx-sfp - inp3tensor  -- could be migrated up for broadcast reuse via. transformations
      %op_with_3inputs = ddl.condition_or(%bn_op, %fnms_op, %where_op, %sinkcorr_op)
      ddl.if(%op_with_3inputs) {
          ddl.data_transfer(%src_inp3_lxsfp, [%sfp_inp3]) {}
      }
      
      ddl.if(%sinkcorr_op) {
        // lx-sfp - inp1tensor -- transfer to sfp register
        ddl.data_transfer(%src_inp1_lxsfp, [%sfp_inp1]) {}
      } else {      
        // lx-sfp - inp1tensor -- always stream -- dont allocate in sfp register
        ddl.if (%mul_i32_to_i32_op) {
          %src_inp1_uint32_lxsfp = ddl.unit(%inp1tensor_uint32,  %inp1tensor_uint32_lx_allocation) {unit="lxlu", data_connect="l3_lx_input1"}
          %dst_inp1_uint32_lxsfp = ddl.unit(%inp1tensor_uint32) {unit="sfp", data_connect="sfp_lx_input1"}  
          ddl.data_transfer(%src_inp1_uint32_lxsfp, [%dst_inp1_uint32_lxsfp]) {}
        } else {
          %dst_inp1_lxsfp = ddl.unit(%inp1tensor) {unit="sfp", data_connect="sfp_lx_input1"}
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.data_transfer(%src_inp1_lxsfp, [%dst_inp1_lxsfp]) {}
          }
        }
      }

      // compute in sfp.. use conditional for each operation.. 
      %sfp_dst = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}

      ddl.if(%op_without_interleaving) {
        ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
          // bn_op (fma)
          ddl.if(%bn_op) {
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2, %sfp_inp3], [%sfp_dst]) {computetype="MACC", unit="sfp"}
          }
          // fnms_op (fnms)
          ddl.if(%fnms_op) {
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2, %sfp_inp3], [%sfp_dst]) {computetype="FNMS", unit="sfp"}
          }
          // biasadd_op or add or strided add (fma) // B*1+A
          %addition_ops = ddl.condition_or(%biasadd_op, %add_op, %stridedadd_op)
          ddl.if(%addition_ops) {
            ddl.compute([%sfp_inp1_fifo, %one_const, %sfp_inp2], [%sfp_dst]) {computetype="MACC", unit="sfp"}
          }
          // mul // B*A+0
          ddl.if(%mul_op) {
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2, %zero_const], [%sfp_dst]) {computetype="MACC", unit="sfp"}
          }
          ddl.if(%mul_i32_to_i32_op) {
            ddl.opaque(%outtensor_uint32, %lsbgate_const_sfp_allocation, %carrydiv_8_const_sfp_allocation, %carrymul_8_const_sfp_allocation, %inp2tensor_uint32_sfp_allocation) {
              unit="sfp", op="MULI32TOI32",
              input_output_registers=["lsbgate", "carrydiv_8", "carrymul_8", "in1_unroll"],
              internal_registers=["carry_unroll", "a0_unroll", "a1_unroll", "a2_unroll", "a3_unroll", "b0_unroll", "b1_unroll", "b2_unroll", "b3_unroll", "t_result_unroll", "t_curr_unroll"],
              max_unroll_factor=1, params={"in0_unroll"="lxlu", "out0"="result"},
              input_data_connects=[  "sfp_lx_input1", "lsbgate_sfp_lrf", "carrydiv_8_sfp_lrf", "carrymul_8_sfp_lrf", "sfp_lx_input2" ],
              output_data_connects=["sfp_output_uint32"]
            }
          }
          // sub -(B*1-A) = A-B
          ddl.if(%sub_op) {
            ddl.compute([%sfp_inp2, %one_const, %sfp_inp1_fifo], [%sfp_dst]) {computetype="FNMS", unit="sfp"}
          }
          // rev-sub -(A*1-B) = B-A
          ddl.if(%revsub_op) {
            ddl.compute([%sfp_inp1_fifo, %one_const, %sfp_inp2], [%sfp_dst]) {computetype="FNMS", unit="sfp"}
          }
          ddl.if(%maximum_op) {
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_dst]) {computetype="FMAX", unit="sfp"}
          }
          ddl.if(%minimum_op) {
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_dst]) {computetype="FMIN", unit="sfp"}
          }
          ddl.if(%where_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            ddl.compute([%sfp_inp1_fifo, %zero_const], [%sfp_state]) {computetype="NOTEQUAL", unit="sfp"}
            ddl.compute([%sfp_state, %sfp_inp2, %sfp_inp3], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%eq_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="EQUAL", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%ge_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="GREATEREQUAL", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%le_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="LESSEREQUAL", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%gt_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="GREATERTHAN", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%lt_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="LESSERTHAN", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
          ddl.if(%ne_op) {
            %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
            %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
            %zero_sfp = ddl.unit(%zero_const_opaque, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            %plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.compute([%sfp_inp1_fifo, %sfp_inp2], [%sfp_state]) {computetype="NOTEQUAL", unit="sfp"}
            ddl.compute([%sfp_state, %plus1_sfp, %zero_sfp], [%sfp_dst]) {computetype="SELECT", unit="sfp"}
          }
        }
      } else {  // op_with_interleaving
        // realdiv // B/A (1/input2 * input1)
        ddl.if(%realdiv_op) {
          ddl.opaque(%outtensor, %zero_sfp_allocation, %ffff_sfp_allocation, %minus1_sfp_allocation, %plus1_sfp_allocation)
          {unit="sfp", op="REALDIV", input_output_registers=["c1", "c2", "c3", "c4"], internal_registers=["p0_unroll", "t0_unroll", "t2_unroll", "t4_unroll", "t6_unroll"],
          max_unroll_factor=2, params={"in0_unroll"="lxlu", "in1_unroll"="lxlu", "out0"="result", "outreg_unroll"="no"},
          input_data_connects=["zero_sfp_lrf", "ffff_sfp_lrf", "minus1_sfp_lrf", "plus1_sfp_lrf", "sfp_lx_input1", "sfp_lx_input2"],
          output_data_connects=["sfp_output"]}
        }
        ddl.if(%sinkcorr_op) {
          %internal1_allocation = ddl.allocate(%internal_tensor1) {memory="sfplrf"}
          %internal2_allocation = ddl.allocate(%internal_tensor2) {memory="sfplrf"}
          %internal3_allocation = ddl.allocate(%internal_tensor3) {memory="sfplrf"}
          %sfp_internal1 = ddl.unit(%internal_tensor1, %internal1_allocation) {unit="sfp", data_connect="sfp_internal1"}
          %sfp_internal2 = ddl.unit(%internal_tensor2, %internal2_allocation) {unit="sfp", data_connect="sfp_internal2"}
          %sfp_internal3 = ddl.unit(%internal_tensor3, %internal3_allocation) {unit="sfp", data_connect="sfp_internal3"}

          // m1 = maximum(inp1(AttnMax), inp3(Sink))
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_inp1, %sfp_inp3], [%sfp_internal1]) {computetype="FMAX", unit="sfp"}
          }
          // s1 = inp1 - m1     realized as -(m1*1-inp1)
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal1, %one_const, %sfp_inp1], [%sfp_internal2]) {computetype="FNMS", unit="sfp"}
          }
          // e1 = exp(s1)     realized as fast exponent
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal2, %fastexp_const_sfp], [%sfp_internal2]) {computetype="FMUL", unit="sfp"}
          }
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal2], [%sfp_internal2]) {computetype="ICVT", unit="sfp", mode=7}
          }
          // s2 = inp3 - m1     realized as -(m1*1-inp3)
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal1, %one_const, %sfp_inp3], [%sfp_internal3]) {computetype="FNMS", unit="sfp"}
          }
          // e2 = exp(s2)     realized as fast exponent
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal3, %fastexp_const_sfp], [%sfp_internal3]) {computetype="FMUL", unit="sfp"}
          }
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal3], [%sfp_internal3]) {computetype="ICVT", unit="sfp", mode=7}
          }
          // m2 = inp2(AttnSum) * e1
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_inp2, %sfp_internal2], [%sfp_internal1]) {computetype="FMUL", unit="sfp"}
          }
          // a1 = m2 + e2
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal1, %one_const, %sfp_internal3], [%sfp_internal3]) {computetype="MACC", unit="sfp"}
          }
          // d1 = inp2 / a1       realized as (1/input2 * input1)
          ddl.opaque(%inp2tensor, %zero_sfp_allocation, %ffff_sfp_allocation, %minus1_sfp_allocation, %plus1_sfp_allocation, %internal3_allocation, %inp2tensor_sfp_allocation, %internal3_allocation)
          {unit="sfp", op="REALDIV", input_output_registers=["c1", "c2", "c3", "c4", "in0_unroll", "in1_unroll", "outreg_unroll"], internal_registers=["p0_unroll", "t0_unroll", "t2_unroll", "t4_unroll", "t6_unroll"],
          max_unroll_factor=2, params={"out0"="no"},
          input_data_connects=["zero_sfp_lrf", "ffff_sfp_lrf", "minus1_sfp_lrf", "plus1_sfp_lrf", "sfp_lx_input2", "sfp_internal3"],
          output_data_connects=["sfp_internal3"]}
          // res = e1 * d1
          ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            ddl.compute([%sfp_internal2, %sfp_internal3], [%sfp_dst]) {computetype="FMUL", unit="sfp"}
          }
        }
      }

      // sfp-lx
      ddl.loop (%interleaving_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.if(%mul_i32_to_i32_op) {
          %src_out_sfplx = ddl.unit(%outtensor_uint32) {unit="sfp", data_connect="sfp_output_uint32"}
          %dst_out_sfplx = ddl.unit(%outtensor_uint32, %outtensor_uint32_lx_allocation) {unit="lxsu", data_connect="lxsu_input"}
          ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
        } else {
          %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
          %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_input"}
          ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
        }
      }
    }
  }
}

ddl.transformations {
}

}
