//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    %d:6 = ddl.dimension {} : index, index, index, index, index, index
    %slice_layout = ddl.layout() {is_order_fixed=false} 
    %stick_layout = ddl.layout() {is_order_fixed=false}
    %global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}
    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
    // inptensor1: forward input (gelu'(x)) inptensor2: backward input (y) outtensor: backward output (gelu'(x)*y)
    %inptensor1, %inptensor2, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index, index, index
    %interim_tensor0 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
    %interim_tensor1 = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
    
    %gelu_bwd_op = ddl.operation_bind([%type_fp16], [%inptensor1, %inptensor2], [%outtensor]) {opFuncName="gelubackward", required=false}
    %tanh_bwd_op = ddl.operation_bind([%type_fp16], [%inptensor1, %inptensor2], [%outtensor], [%interim_tensor0, %interim_tensor1]) {opFuncName="tanhbackward", required=false}
  
    ddl.constraint(%inptensor1, %inptensor2, %outtensor) {property = "slice", cmp = "equal"}
    ddl.constraint(%inptensor1, %inptensor2, %outtensor) {property = "stick", cmp = "equal"}
    ddl.constraint() {min_num_cores = 1}
    ddl.constraint(%gelu_bwd_op, %tanh_bwd_op) {min_num_valid = 1, max_num_valid = 1}
    
    %gelu_bwd_const_C1 = ddl.define_constant(%type_fp16){value=[0x3449], name="geluBwdVal1"}  // 0.0356774
    %gelu_bwd_const_C2 = ddl.define_constant(%type_fp16){value=[0x3D31], name="geluBwdVal2"}  // 0.797885
    // %gelu_bwd_const_C3 = ddl.define_constant(%type_fp16){value=[0xBE00], name="geluBwdVal3"}  // -1.0
    %gelu_bwd_const_C4 = ddl.define_constant(%type_fp16){value=[0x356D], name="geluBwdVal4"}  // 0.0535161
    %gelu_bwd_const_C5 = ddl.define_constant(%type_fp16){value=[0x3B31], name="geluBwdVal5"}  // 0.398942
    %gelu_bwd_const_C6 = ddl.define_constant(%type_fp16){value=[0x3C00], name="geluBwdVal6"}  // 0.5
    %gelu_bwd_const_C7 = ddl.define_constant(%type_fp16){value=[0x4000], name="geluBwdVal7"}  // 2
    %gelu_bwd_const_exp = ddl.define_constant(%type_fp16){value=[0x54E3], name="fastexpVal"}  // 0x54E354E354E354E354E354E354E354E3
    %const_0 = ddl.define_constant(%type_fp16){value=[0], name="zero"} // 0
    %const_1 = ddl.define_constant(%type_fp16){value=[0x3E00], name="plus1"} // 1
    
    
    // allocate space: lx
    %allocate_handler_input1_lx = ddl.get_external_data_transfer_allocation (%inptensor1) {memory="lx", data_connect="l3_lx_input1"} 
    %allocate_handler_input2_lx = ddl.get_external_data_transfer_allocation (%inptensor2) {memory="lx", data_connect="l3_lx_input2"} 
    %allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_input"}
    
    ddl.dataflow {
      %d_datastage = ddl.get_external_datastage{property = "core"}
      %b_datastage = ddl.get_external_datastage {property = "chunk"}
      %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
      %bottom_datastage = ddl.datastage {strategy="minimize"}
      ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5) {values=["1", "2", "4"]}
    
      // Constants
      %gelu_pe_const_C1_allocation = ddl.allocate(%gelu_bwd_const_C1) {memory="pelrf"}
      %gelu_pe_const_C2_allocation = ddl.allocate(%gelu_bwd_const_C2) {memory="pelrf"}
      %gelu_pe_const_C4_allocation = ddl.allocate(%gelu_bwd_const_C4) {memory="pelrf"}
      %gelu_pe_const_C5_allocation = ddl.allocate(%gelu_bwd_const_C5) {memory="pelrf"}
      %gelu_sfp_const_C6_allocation = ddl.allocate(%gelu_bwd_const_C6) {memory="sfplrf"}
      %gelu_pe_const_exp_allocation = ddl.allocate(%gelu_bwd_const_exp) {memory="pelrf"}
      %gelu_sfp_const_C7_allocation = ddl.allocate(%gelu_bwd_const_C7) {memory="sfplrf"}
      %const_0_sfp_allocation = ddl.allocate(%const_0) {memory="sfplrf"}
      %const_1_sfp_allocation = ddl.allocate(%const_1) {memory="sfplrf"}

      ddl.if (%tanh_bwd_op) {
        %src_plus1 = ddl.unit(%const_1) {unit="constant", data_connect= "tanh_const_1_connect"}
        %dst_plus1_sfp = ddl.unit(%const_1, %const_1_sfp_allocation) {unit="sfp", data_connect= "tanh_const_1_sfp_lrf"}
        ddl.data_transfer(%src_plus1, [%dst_plus1_sfp]) {}
        // main dataflow
        ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
          ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
            // lx-sfp - inptensor2
            %sfp_inp2_lrf_alloc = ddl.allocate(%inptensor2) {memory = "sfplrf"}
            %src_inp2_lx = ddl.unit(%inptensor2, %allocate_handler_input2_lx) {unit="lxlu", data_connect="l3_lx_input2"}
            %sfp_src00 = ddl.unit(%inptensor2, %sfp_inp2_lrf_alloc) {unit="sfp", data_connect="sfp_lx_input2"}
            ddl.data_transfer(%src_inp2_lx, [%sfp_src00]) {}

            %sfp_allocation00 = ddl.allocate(%interim_tensor0) {memory="sfplrf"}
            %sfp_allocation11 = ddl.allocate(%interim_tensor1) {memory="sfplrf"}
            %sfp_out00 = ddl.unit(%interim_tensor0, %sfp_allocation00) {unit="sfp", data_connect="sfp_outtensor_lrf"}
            %sfp_out11 = ddl.unit(%interim_tensor1, %sfp_allocation11) {unit="sfp", data_connect="sfp_outtensor_lrf"}

            // lx-sfp - inptensor1
            %sfp_inp1_lrf_alloc = ddl.allocate(%inptensor1) {memory = "sfplrf"}
            %src_inp1_lx = ddl.unit(%inptensor1, %allocate_handler_input1_lx) {unit="lxlu", data_connect="l3_lx_input1"}
            %sfp_src10 = ddl.unit(%inptensor1, %sfp_inp1_lrf_alloc) {unit="sfp", data_connect="sfp_lx_input1"}
            ddl.data_transfer(%src_inp1_lx, [%sfp_src10]) {}
            
            %sfp_dst = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_output"}

            // ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            //   ddl.compute([%sfp_src00], [%sfp_out00]) {computetype="FEST", unit="sfp", mode=8}
            // }
            // ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            //   ddl.compute([%sfp_src00], [%sfp_out11]) {computetype="FEST", unit="sfp", mode=9}
            // }
            // ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
            //   ddl.compute([%sfp_out00, %sfp_src00, %sfp_out11], [%sfp_out00]) {computetype="FMA16", unit="sfp"}
            // }
            ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
              // ddl.compute([%sfp_out00, %sfp_out00, %dst_plus1_sfp], [%sfp_out00]) {computetype="FNMS", unit="sfp"}
              ddl.compute([%sfp_src00, %sfp_src00, %dst_plus1_sfp], [%sfp_out00]) {computetype="FNMS", unit="sfp"}
            }
            ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
              ddl.compute([%sfp_out00, %sfp_src10], [%sfp_dst]) {computetype="FMUL", unit="sfp"}
            }
            // sfp-lx
            %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_output"}
            %dst_out_sfplx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lxsu_input"}
            ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
          }
        }
      }
      ddl.if (%gelu_bwd_op) {
        %src_gelu_const_C1 = ddl.unit(%gelu_bwd_const_C1) {unit="constant", data_connect= "gelu_const_C1_connect"}
        %src_gelu_const_C2 = ddl.unit(%gelu_bwd_const_C2) {unit="constant", data_connect= "gelu_const_C2_connect"}
        %src_gelu_const_C4 = ddl.unit(%gelu_bwd_const_C4) {unit="constant", data_connect= "gelu_const_C4_connect"}
        %src_gelu_const_C5 = ddl.unit(%gelu_bwd_const_C5) {unit="constant", data_connect= "gelu_const_C5_connect"}
        %src_gelu_const_C6 = ddl.unit(%gelu_bwd_const_C6) {unit="constant", data_connect= "gelu_const_C6_connect"}
        %src_gelu_const_exp = ddl.unit(%gelu_bwd_const_exp) {unit="constant", data_connect= "gelu_const_exp_connect"}
        %src_gelu_const_C7 = ddl.unit(%gelu_bwd_const_C7) {unit="constant", data_connect= "gelu_const_C7_connect"}
        %src_gelu_const_0 = ddl.unit(%const_0) {unit="constant", data_connect= "gelu_const_0_connect"}
       
        %dst_gelu_const_C1_pe = ddl.unit(%gelu_bwd_const_C1, %gelu_pe_const_C1_allocation) {unit="pe", data_connect= "gelu_const_C1_pe_lrf"}
        %dst_gelu_const_C2_pe = ddl.unit(%gelu_bwd_const_C2, %gelu_pe_const_C2_allocation) {unit="pe", data_connect= "gelu_const_C2_pe_lrf"}
        %dst_gelu_const_C4_pe = ddl.unit(%gelu_bwd_const_C4, %gelu_pe_const_C4_allocation) {unit="pe", data_connect= "gelu_const_C4_pe_lrf"}
        %dst_gelu_const_C5_pe = ddl.unit(%gelu_bwd_const_C5, %gelu_pe_const_C5_allocation) {unit="pe", data_connect= "gelu_const_C5_pe_lrf"}
        %dst_gelu_const_C6_sfp = ddl.unit(%gelu_bwd_const_C6, %gelu_sfp_const_C6_allocation) {unit="sfp", data_connect= "gelu_const_C6_sfp_lrf"}
        %dst_gelu_const_exp_pe = ddl.unit(%gelu_bwd_const_exp, %gelu_pe_const_exp_allocation) {unit="pe", data_connect= "gelu_const_exp_pe_lrf"}
        %dst_gelu_const_C7_sfp = ddl.unit(%gelu_bwd_const_C7, %gelu_sfp_const_C7_allocation) {unit="sfp", data_connect= "gelu_const_C7_sfp_lrf"}
        %dst_gelu_const_0_sfp = ddl.unit(%const_0, %const_0_sfp_allocation) {unit="sfp", data_connect= "gelu_const_0_sfp_lrf"}

        ddl.data_transfer(%src_gelu_const_C1, [%dst_gelu_const_C1_pe]) {}
        ddl.data_transfer(%src_gelu_const_C2, [%dst_gelu_const_C2_pe]) {}
        ddl.data_transfer(%src_gelu_const_C4, [%dst_gelu_const_C4_pe]) {}
        ddl.data_transfer(%src_gelu_const_C5, [%dst_gelu_const_C5_pe]) {}
        ddl.data_transfer(%src_gelu_const_C6, [%dst_gelu_const_C6_sfp]) {}
        ddl.data_transfer(%src_gelu_const_exp, [%dst_gelu_const_exp_pe]) {}
        ddl.data_transfer(%src_gelu_const_C7, [%dst_gelu_const_C7_sfp]) {}
        ddl.data_transfer(%src_gelu_const_0, [%dst_gelu_const_0_sfp]) {}
    
        // main dataflow
        ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
          ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
      
            // %pe_lrf_allocation = ddl.allocate(%inptensor) {memory="pelrf"} // do not used (stream input)
           
            // send input
            // lx-pe - inptensor2
            %src_inp2_lx = ddl.unit(%inptensor2, %allocate_handler_input2_lx) {unit="lxlu", data_connect="l3_lx_input2"}
            %dst_inp2_pe = ddl.unit(%inptensor2) {unit="pe", data_connect="lx_pe_input2"}  // no need to allocate input in pe-register as its done on the fly
            ddl.data_transfer(%src_inp2_lx, [%dst_inp2_pe]) {}
   
            // compute in PE
            ddl.if (%gelu_bwd_op) {
              ddl.opaque(%outtensor, %gelu_pe_const_C1_allocation, %gelu_pe_const_C2_allocation, %gelu_pe_const_C4_allocation, %gelu_pe_const_C5_allocation, %gelu_pe_const_exp_allocation)
                  {unit="pe", op="GELU_BWD_P1", input_output_registers=["c1", "c2", "c4", "c5", "cexp"], 
                  internal_registers=["p0_unroll", "t0_unroll", "t1_unroll", "t2_unroll", "t3_unroll"],
                  max_unroll_factor=4, params={"in0"="lxlu", "out0"="result"},
                  input_data_connects=["gelu_const_C1_pe_lrf", "gelu_const_C2_pe_lrf", "gelu_const_C4_pe_lrf", "gelu_const_C5_pe_lrf", "lx_pe_input2"],
                  output_data_connects=["pe_output"]}
            }
      
            // pe-sfp
            %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_output"} 
            %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_input"}
            ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
   
            // lx-sfp - inptensor1
            %src_inp1_lx = ddl.unit(%inptensor1, %allocate_handler_input1_lx) {unit="lxlu", data_connect="l3_lx_input1"}
            %dst_inp1_sfp = ddl.unit(%inptensor1) {unit="sfp", data_connect="lx_sfp_input1"}  // no need to allocate input in pe-register as its done on the fly
            ddl.data_transfer(%src_inp1_lx, [%dst_inp1_sfp]) {}
      
            ddl.if (%gelu_bwd_op) {
              ddl.opaque(%outtensor, %const_0_sfp_allocation, %gelu_sfp_const_C6_allocation, %gelu_sfp_const_C7_allocation)
                  {unit="sfp", op="GELU_BWD_P2", input_output_registers=["czero", "c6", "c7"], internal_registers=["p0_unroll", "t0_unroll", "t1_unroll", "t2_unroll"],
                  max_unroll_factor=4, params={"in0"="pe", "in1"="lxlu", "out0"="result"}, 
                  input_data_connects=["gelu_const_C6_sfp_lrf", "gelu_const_C7_sfp_lrf", "sfp_input", "lx_sfp_input1"], 
                  output_data_connects=["sfp_output"]}
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
    }
    
    }
