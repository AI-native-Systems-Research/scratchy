//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

// CSQ-int8wt/qfp8wt/int4wt
module {
    // Dimension
    %outer_dim:4 = ddl.dimension{} : index, index, index, index // X, Y, I, MB dimension
    %stick_dim = ddl.dimension{} : index // OUT dimension
    %output_stick_dim = ddl.dimension{} : index // J dimension

    // Layout
    %slice_layout_in = ddl.layout (%stick_dim) {is_order_fixed=true}
    %slice_layout_out = ddl.layout (%output_stick_dim, %stick_dim) {is_order_fixed=true} 
    %stick_layout = ddl.layout (%stick_dim) {is_order_fixed=true} 
    %global_layout = ddl.layout(%output_stick_dim, %stick_dim, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3 ) {is_order_fixed=false}

    // Datatype
    %type_fp16 = ddl.type{data_type="SEN169_FP16", bit_width=16}
    %type_bf16 = ddl.type{data_type="BFLOAT16", bit_width=16}
    %type_int8 = ddl.type{data_type="SENINT8", bit_width=8}
    %type_fp8 = ddl.type{data_type="SEN143_FP8", bit_width=8}
    %type_inter_int8 = ddl.type{data_type="SENINT8", bit_width=16}
    %type_inter_fp8 = ddl.type{data_type="SEN143_FP8", bit_width=16}
    %type_int4 = ddl.type{data_type="SENINT4", bit_width=4}
    %type_int4_16 = ddl.type{data_type ="SENINT4", bit_width=16}
    %type_int4_8 = ddl.type{data_type ="SENINT4", bit_width=8}

    // Tensor
    %input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias = 
            ddl.tensor(%slice_layout_in, %stick_layout, %global_layout, [%type_fp16]) : index, index, index, index, index

    %inter_int8_tensor = ddl.internal_tensor(%input_tensor, [%type_inter_int8]) : index
    %inter_int4_tensor = ddl.internal_tensor(%input_tensor, [%type_int4_16]) : index
    %output_int8_tensor = ddl.tensor(%slice_layout_out, %stick_layout, %global_layout, [%type_int8]) : index
    %output_fp8_tensor = ddl.tensor(%slice_layout_out, %stick_layout, %global_layout, [%type_fp8]) : index
    %output_int4_tensor = ddl.tensor(%slice_layout_out, %stick_layout, %global_layout, [%type_int4]) : index
    %output_bf16_tensor = ddl.tensor(%slice_layout_in, %stick_layout, %global_layout, [%type_bf16]) : index
    
    // Op
    %csqint8wt_op = ddl.operation_bind([], [%input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int8_tensor], [%inter_int8_tensor]) {opFuncName="csqint8wt", required=false}
    %csqint4wt_op = ddl.operation_bind([], [%input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int4_tensor], [%inter_int4_tensor]) {opFuncName="csqint4wt", required=false}
    %qfp8wt_op = ddl.operation_bind([], [%input_tensor], [%output_fp8_tensor]) {opFuncName="qfp8wt", required=false}
    %dl16tobf16_op = ddl.operation_bind([], [%input_tensor], [%output_bf16_tensor]) {opFuncName="dl16tobf16", required=false}

    // Constraints
    ddl.constraint() {min_num_cores = 1}
    ddl.constraint(%output_int8_tensor, %output_fp8_tensor, %output_int4_tensor) {property = "slice", dim_idx = 1, cmp = "equal", value = 8}
    ddl.constraint(%input_tensor, %output_bf16_tensor) {property = "slice", cmp = "equal"}
    ddl.constraint(%input_tensor, %output_bf16_tensor) {property = "stick", cmp = "equal"}
    ddl.constraint(%csqint8wt_op, %csqint4wt_op, %qfp8wt_op, %dl16tobf16_op) {min_num_valid = 1, max_num_valid = 1}

    // Constants
    %scale_const = ddl.get_external_constant(%type_fp16) {name="scaleact", num_elements=1}
    %offset_const = ddl.get_external_constant(%type_fp16) {name="shiftact", num_elements=1}
    %ext_zero = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}

    // Alias tensors
    %output_tensor = ddl.alias_one_tensor_of(%output_int8_tensor, %output_fp8_tensor, %output_int4_tensor, %output_bf16_tensor)

    // Memory Allocation
    %allocate_handler_input_lx = ddl.get_external_data_transfer_allocation(%input_tensor) {memory="lx", data_connect="l3_lx_input"}
    %allocate_handler_output_lx = ddl.get_external_data_transfer_allocation(%output_tensor) {memory="lx", data_connect="lx_l3_output"}

    // Dataflow
    // %fp_ops = %qfp8_op 
    %csq_ops = ddl.condition_or(%csqint8wt_op, %csqint4wt_op)
    %pe_forward_ops = ddl.condition_or(%qfp8wt_op)

    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage {property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        
        %bottom_datastage = ddl.datastage {strategy = "minimize"}
        // %pack_datastage = ddl.datastage {strategy = set to either 2 or 4 ==> ** need syntax defn **}  
        %pack_datastage = ddl.datastage {strategy = "minimize"}
        // ddl.constraint - check chunk outstick-dim same with core outstick-dim
        // ddl.constraint - check chunk padded-dim same with core padded-dim
        // ddl.constraint - check Cj is a multiple of 4
        // ddl.constraint - ensure pack datastage is 2 or 4

        // Constants transfer
        %scale_pe_allocation = ddl.allocate(%scale_const) {memory="pelrf"} 
        %offset_pe_allocation = ddl.allocate(%offset_const) {memory="pelrf"} 

        %pe_scale = ddl.unit(%scale_const, %scale_pe_allocation) {unit="pe", data_connect="scale_pe_lrf"}
        %pe_offset = ddl.unit(%offset_const, %offset_pe_allocation) {unit="pe", data_connect="offset_pe_lrf"}

        ddl.if(%csq_ops) {
            %src_scale_const = ddl.unit(%scale_const) {unit="constant", data_connect= "scale_const_connect"} 
            ddl.data_transfer(%src_scale_const, [%pe_scale]) {}

            %src_offset_const = ddl.unit(%offset_const) {unit="constant", data_connect= "offset_const_connect"} 
            ddl.data_transfer(%src_offset_const, [%pe_offset]) {}
        }

        %zero_pe_allocation = ddl.allocate(%ext_zero) {memory="pelrf"} 
        %pe_zero_lrf = ddl.unit(%ext_zero, %zero_pe_allocation) {unit="pe", data_connect= "zero_pe_lrf"}
        ddl.if(%qfp8wt_op) {
            %src_pe_zero = ddl.unit(%ext_zero) {unit="constant", data_connect= "zero_connect"} 
            ddl.data_transfer(%src_pe_zero, [%pe_zero_lrf]) {}
        }


        ddl.loop(%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %stick_dim, %output_stick_dim) {} {
            ddl.loop(%b_datastage, %bottom_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %stick_dim) {} {
                                
                // Inputs Loop
                ddl.loop(%b_datastage, %bottom_datastage, %output_stick_dim) {} {
                    %src_lxpe = ddl.unit(%input_tensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
                    ddl.if (%csqint8wt_op) {
                        %dst_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="lx_pe_input"}
                        ddl.data_transfer(%src_lxpe, [%dst_lxpe]) {}                    
                        %pe_src00 = ddl.unit(%input_tensor) {unit="lxlu", data_connect="lx_pe_input"}
                        %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                        ddl.compute([%pe_src00, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                        %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_output"}
                        %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_input"}
                        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                    }
                    ddl.if (%csqint4wt_op) {
                        %dst_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="lx_pe_input"}
                        ddl.data_transfer(%src_lxpe, [%dst_lxpe]) {}                    
                        %pe_src00 = ddl.unit(%input_tensor) {unit="lxlu", data_connect="lx_pe_input"}
                        %pe_dst00 = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                        ddl.compute([%pe_src00, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                        %src_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_output"}
                        %dst_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_input"}
                        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                    }
                    %sfp_forward_ops = ddl.condition_or(%qfp8wt_op, %dl16tobf16_op)
                    ddl.if (%sfp_forward_ops) {
                        %dst_lxsfp = ddl.unit(%input_tensor) {unit="sfp", vias=["pe"], data_connect="pe_sfp_input"}
                        ddl.data_transfer(%src_lxpe, [%dst_lxsfp]) {}
                    }   
                }

                // currently no padding needed

                // output loop:
                ddl.loop(%b_datastage, %pack_datastage, %output_stick_dim) {} {
                    %sfp_dst00 = ddl.unit(%output_tensor) {unit="lxsu", data_connect="sfp_lx_output"} 
                    ddl.if (%csqint8wt_op) {
                    // J / 2 loop:
                        %sfp_src00 = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                    ddl.if (%qfp8wt_op) {
                    // J / 2 loop:
                        %sfp_src00 = ddl.unit(%input_tensor) {unit="pe", data_connect="pe_sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                    ddl.if (%csqint4wt_op) {
                    // J / 4 loop:
                        %sfp_src00 = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                    ddl.if (%dl16tobf16_op) {
                        // %sfp_src = ddl.unit(%input_tensor) {unit="pe", data_connect="pe_sfp_input"}
                        // ddl.compute([%sfp_src], [%sfp_dst00]) {computetype="PACKMERGE", unit="sfp"}
                        ddl.opaque(%output_tensor)
                            {unit="sfp", op="DL16TOBF16", input_output_registers=[], internal_registers=[],
                            max_unroll_factor=1, params={"in0"="pe", "out0"="result"},
                            input_data_connects=["pe_sfp_input"], output_data_connects=["sfp_lx_output"]}
                    }
                }
                %src_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_lx_output"}
                %dst_sfplx = ddl.unit(%output_tensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_l3_output"}
                ddl.data_transfer(%src_sfplx, [%dst_sfplx]) {}                
            }
        }
    }
    
    ddl.transformations {
        ddl.disable_transfer_promotion()
    }
}
