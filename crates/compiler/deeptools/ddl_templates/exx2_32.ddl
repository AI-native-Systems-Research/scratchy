//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

// Sum Mean Max Exx2

module {
    // Dimension
    %outer_dim:5 = ddl.dimension{} : index, index, index, index,index // X, Y, I, J, MB dimension
    %reduce_dim:3 = ddl.dimension{} : index, index, index // OUT dimension

    // Layout
    %slice_layout_stick = ddl.layout (%reduce_dim#0) {is_order_fixed=true}
    %stick_layout_stick = ddl.layout (%reduce_dim#0) {is_order_fixed=true}

    // Nonstick layout
    %slice_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}
    %stick_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}

    %global_layout_input = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {is_order_fixed=false}
    %global_layout_output = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {is_order_fixed=false}


    // DataType 
    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
    
    // Tensor stick
    %input_tensor = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_input, [%type_fp16]) : index
    %output_tensor = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_output, [%type_fp16]) : index

    %output_tensor_x2 = ddl.internal_tensor(%output_tensor, [%type_fp16]) : index
    %output_tensor_low = ddl.internal_tensor(%output_tensor, [%type_fp16]) : index
    %output_tensor_high = ddl.internal_tensor(%output_tensor, [%type_fp16]) : index

    // Op
    %exx2_op = ddl.operation_bind([%type_fp16], [%input_tensor, %output_tensor], [%output_tensor], [%output_tensor_x2, %output_tensor_low, %output_tensor_high]) {opFuncName="exx2", required=false}
    // %exx2_zeromean_op = ddl.operation_bind([%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_x2, %output_tensor_low, %output_tensor_high]) {opFuncName="exx2_zeromean", required=false}

    // Constant 
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}
    %zero_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %negInf_const_reg = ddl.define_constant(%type_fp16){value=[0xFFFE], name="negInf"}
    %scaling_factor_const_reg = ddl.get_external_constant(%type_fp16){name="scaling_factor", num_elements=1} // (1 / N)
    %nfwd0_const = ddl.operand_constant {name="nfwd0"}
    %nfwd2_const = ddl.operand_constant {name="nfwd2"}
    %a_const_reg = ddl.define_constant(%type_fp16) {value=[0xA], name="aconst"}
    %exx2_div = ddl.get_external_constant(%type_fp16){name="exx2scale", num_elements=1}

    // Allocation
    %input_lx_allocation = ddl.get_external_data_transfer_allocation (%input_tensor) {memory="lx", data_connect="l3_lx_input"} 
    %output_lx_allocation = ddl.get_external_data_transfer_allocation (%output_tensor) { memory="lx", data_connect="lxsu_output"}

    // Constraints
    ddl.constraint(%exx2_op) {min_num_valid = 1, max_num_valid = 1}
    ddl.constraint() {min_num_cores = 1}

    // Dataflow
    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage{property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %above_interleave = ddl.datastage {strategy="maximize", allow_epilogue=true}
        %below_interleave = ddl.datastage {strategy="minimize"}
        ddl.datastage_constraint(%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {values=["1"]}

        // Constant:
        %zero_sfp_allocation = ddl.allocate(%zero_const_reg) {memory="sfplrf"} 
        %src_sfp_zero = ddl.unit(%zero_const_reg) {unit="constant", data_connect= "zero_connect"} 
        %dst_sfp_zero = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
        ddl.data_transfer(%src_sfp_zero, [%dst_sfp_zero]) {}

        %aconst_sfp_allocation = ddl.allocate(%a_const_reg) {memory="sfplrf"} 
        %src_sfp_a = ddl.unit(%a_const_reg) {unit="constant", data_connect= "aconst_connect"} 
        %dst_sfp_a = ddl.unit(%a_const_reg, %aconst_sfp_allocation) {unit="sfp", data_connect= "aconst_sfp_lrf"}
        ddl.data_transfer(%src_sfp_a, [%dst_sfp_a]) {}

        %exx2div_sfp_allocation = ddl.allocate(%exx2_div) {memory="sfplrf"} 
        %src_exx2div_const = ddl.unit(%exx2_div) {unit="constant", data_connect= "exx2div_connect"} 
        %dst_exx2div_sfp = ddl.unit(%exx2_div, %exx2div_sfp_allocation) {unit="sfp", data_connect= "exx2div_sfp_lrf"}
        ddl.data_transfer(%src_exx2div_const, [%dst_exx2div_sfp]) {}

        ddl.loop (%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2){} {
            ddl.loop (%b_datastage, %above_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4){} {
                
                %sfp_zero_lrf = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}

                %sfp_lrf_allocation = ddl.allocate(%output_tensor) {memory="sfplrf"}
                // %sfp_reduce_dst00_lrf = ddl.unit(%output_tensor, %sfp_lrf_allocation) {unit="sfp", data_connect="inter_stick_output"}

                %sfp_lrf_x2_allocation = ddl.allocate(%output_tensor_x2) {memory="sfplrf"}
                // %sfp_reduce_dst00_x2_lrf = ddl.unit(%output_tensor_x2, %sfp_lrf_x2_allocation) {unit="sfp", data_connect="inter_stick_output_x2"}

                // // explicitly initialize to 0
                // // LRF x 1 + 0 -> LRF
                ddl.opaque(%output_tensor, %sfp_lrf_allocation, %sfp_lrf_x2_allocation, %zero_sfp_allocation)
                          {unit="sfp", op="EXX2_32_P3",
                          input_output_registers=["x1", "x2", "c0"],
                          internal_registers=[],
                          max_unroll_factor=1, params={},
                          input_data_connects=[],
                          output_data_connects=["inter_stick_output", "inter_stick_output_x2"]}

                // ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                //     ddl.compute([%sfp_zero_lrf, %one_const, %zero_const], [%sfp_reduce_dst00_lrf]) {computetype="FMA16", unit="sfp"}
                // }

                // ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                //     // explicitly initialize to 0
                //     // LRF x 1 + 0 -> LRF              
                //     ddl.compute([%sfp_zero_lrf, %one_const, %zero_const], [%sfp_reduce_dst00_x2_lrf]) {computetype="FMA16", unit="sfp"}
                // }

                // Inner-loop of reduction
                ddl.loop (%b_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {} {
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        // LX->SFP 
                        %src_inp_lxsfp = ddl.unit(%input_tensor, %input_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
                        %dst_inp_lxsfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_lx_input"}
                        // For Exx2, lx->sfplrf first to reuse the data
                        // %sfp_input_lrf_allocation = ddl.allocate(%input_tensor) {memory="sfplrf"}
                        // %dst_inp_lxsfplrf = ddl.unit(%input_tensor, %sfp_input_lrf_allocation) {unit="sfp", data_connect="sfp_lx_input_lrf"}

                        // sfp input -> Rx
                        // ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfplrf]) {}
                        ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

                        // sfp inter-stick reduction
                        // chunk accumulation in PE

                        // %sfp_opaque_lrf = ddl.unit(%output_tensor, %sfp_lrf_allocation) {unit="sfp", data_connect="inter_stick_output"}
                        // %sfp_opaque_x2_lrf = ddl.unit(%output_tensor_x2, %sfp_lrf_x2_allocation) {unit="sfp", data_connect="inter_stick_output_x2"}
                        ddl.opaque(%output_tensor, %sfp_lrf_allocation, %sfp_lrf_x2_allocation)
                            {unit="sfp", op="EXX2_32_P1",
                            input_output_registers=["x1", "x2"],
                            internal_registers=["p0_unroll", "p1_unroll", "p2_unroll"],
                            max_unroll_factor=1, params={"in0"="lxlu"},
                            input_data_connects=["sfp_lx_input", "inter_stick_output", "inter_stick_output_x2"],
                            output_data_connects=["sfp_inter_output", "inter_stick_output2", "inter_stick_output_x22"]}
                            
                    } // end inner reduction
                }


                // %sfp_opaque_lrf = ddl.unit(%output_tensor, %sfp_lrf_allocation) {unit="sfp", data_connect="inter_stick_output2"}
                // %sfp_opaque_x2_lrf = ddl.unit(%output_tensor_x2, %sfp_lrf_x2_allocation) {unit="sfp", data_connect="inter_stick_output_x22"}
                ddl.opaque(%output_tensor, %sfp_lrf_allocation, %sfp_lrf_x2_allocation, %zero_sfp_allocation, %exx2div_sfp_allocation)
                    {unit="sfp", op="EXX2_32_P2",
                    input_output_registers=["x1", "x2", "c0", "c1"],
                    internal_registers=["t1_unroll"],
                    max_unroll_factor=1, params={"out0"="result"},
                    input_data_connects=["inter_stick_output2", "inter_stick_output_x22", "zero_sfp_lrf", "exx2div_sfp_lrf"],
                    output_data_connects=["sfp_output", "sfp_x2_output"]}


                // SFP->LX transfer
                %dst_div = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_output"}
                %dst_x2_div = ddl.unit(%output_tensor_x2) {unit="sfp", data_connect="sfp_x2_output"}

                %dst_out_sfplx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                %dst_out_x2_sfplx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output", stick_replicated_dim_offset_elements=8}

                ddl.data_transfer(%dst_div, [%dst_out_sfplx]) {}
                ddl.data_transfer(%dst_x2_div, [%dst_out_x2_sfplx]) {limit_num_elements_stick_replicated_dim=8}

            } // above reduce dim
        } // D/B
    } // dataflow
} // module
