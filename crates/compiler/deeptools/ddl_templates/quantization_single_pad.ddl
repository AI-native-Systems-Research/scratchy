//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    // Dimension
    %outer_dim:4 = ddl.dimension{} : index, index, index, index // X, Y, I, J or MB dimension
    %stick_dim = ddl.dimension{} : index // OUT dimension
    %output_stick_dim = ddl.dimension{} : index // MB or J, not included in outer_dim
    %z_front = ddl.dimension{dim_property="pad_front"} : index
    %z_back = ddl.dimension{dim_property="pad_back"} : index
    %z_valid = ddl.dimension{dim_property="pad_valid"} : index
    %padded_dim = ddl.padded_dimension(primary=%output_stick_dim, padding=[%z_front, %z_back, %z_valid], window=[]) // C dimension J is in output_stick_dim, else nothing
    
    // Layout
    %slice_layout_in = ddl.layout (%stick_dim) {is_order_fixed=true}
    %slice_layout_out_padded = ddl.layout (%stick_dim, %padded_dim) {is_order_fixed=true}
    %slice_layout_out = ddl.layout (%stick_dim, %output_stick_dim) {is_order_fixed=true}
    %stick_layout = ddl.layout (%stick_dim) {is_order_fixed=true}
    %global_layout = ddl.layout(%output_stick_dim, %stick_dim, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3 ) {is_order_fixed=false}
    %global_layout_padded = ddl.layout(%padded_dim, %stick_dim, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3 ) {is_order_fixed=false}

    // Data Type
    %type_fp16 = ddl.type{data_type="SEN169_FP16", bit_width=16}
    %type_int8 = ddl.type{data_type="SENINT8",  bit_width=8}
    %type_fp8 = ddl.type{data_type="SEN143_FP8", bit_width=8}
    %type_inter_int8 = ddl.type{data_type ="SENINT8" , bit_width=16}

    // Tensor
    %input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias = 
            ddl.tensor(%slice_layout_in, %stick_layout, %global_layout, [%type_fp16]) : index, index, index, index, index

    %inter_int8_tensor = ddl.internal_tensor(%input_tensor, [%type_inter_int8]) : index
    %output_int8_tensor_padded = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_int8]) : index
    %output_int8_tensor = ddl.tensor(%slice_layout_out, %stick_layout, %global_layout, [%type_int8]) : index
    %output_fp8_tensor_padded = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_fp8]) : index
    %output_fp8_tensor = ddl.tensor(%slice_layout_out, %stick_layout, %global_layout, [%type_fp8]) : index

    // Op
    %csqint8_op = ddl.operation_bind([], [%input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int8_tensor_padded], [%inter_int8_tensor]) {opFuncName="csqint8", required=false}
    %csqint8mb_op = ddl.operation_bind([], [%input_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int8_tensor], [%inter_int8_tensor]) {opFuncName="csqint8mb", required=false}
    %qfp8_op = ddl.operation_bind([], [%input_tensor], [%output_fp8_tensor_padded]) {opFuncName="qfp8", required=false}
    %qfp8mb_op = ddl.operation_bind([], [%input_tensor], [%output_fp8_tensor]) {opFuncName="qfp8mb", required=false}

    // Constraints
    ddl.constraint(){min_num_cores = 1}
    ddl.constraint(%output_int8_tensor, %output_int8_tensor_padded, %output_fp8_tensor, %output_fp8_tensor_padded) {property = "slice", dim_idx = 0, cmp = "equal", value = 8}
    ddl.constraint(%csqint8_op, %csqint8mb_op, %qfp8_op, %qfp8mb_op) {min_num_valid = 1, max_num_valid = 1}

    // Constants
    %scale_const = ddl.get_external_constant(%type_fp16) {name="scaleact", num_elements=1}
    %offset_const = ddl.get_external_constant(%type_fp16) {name="shiftact", num_elements=1}
    %ext_zero = ddl.define_constant(%type_fp16) {value=[0], name="zero"}

    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}

    // Alias tensors
    // cannot define this before Op/Constraints for definition purpose.
    // can use in dataflow 
    %output_tensor = ddl.alias_one_tensor_of(%output_int8_tensor, %output_int8_tensor_padded, %output_fp8_tensor, %output_fp8_tensor_padded)
    %inter_tensor = ddl.alias_one_tensor_of(%input_tensor, %inter_int8_tensor)

    // Memory Allocation
    %allocate_handler_input_lx = ddl.get_external_data_transfer_allocation(%input_tensor) {memory="lx", data_connect="l3_lx_input"}
    %allocate_handler_output_lx = ddl.get_external_data_transfer_allocation(%output_tensor) {memory="lx", data_connect="lx_l3_output"} 

    // Dataflow
    ddl.dataflow { 
        %d_datastage = ddl.get_external_datastage {property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}

        %bottom_datastage = ddl.datastage {strategy = "minimize"}
        %pack_datastage = ddl.datastage {strategy = "minimize"}  

        %csq_ops = ddl.condition_or(%csqint8_op, %csqint8mb_op)

        // PE Constants
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
        %qfp_ops = ddl.condition_or(%qfp8_op, %qfp8mb_op)
        ddl.if(%qfp_ops) {
            %src_pe_zero = ddl.unit(%ext_zero) {unit="constant", data_connect= "zero_connect"} 
            ddl.data_transfer(%src_pe_zero, [%pe_zero_lrf]) {}
        }

        // TODO: ddl.constraint - check chunk outstick-dim same with core outstick-dim
        // TODO: ddl.constraint - check chunk padded-dim same with core padded-dim
        // ddl.constraint - ensure pack datastage is 2 or 4

        ddl.loop(%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %stick_dim) {} {
            ddl.loop(%b_datastage, %bottom_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %stick_dim) {} {
                // Pad Front
                // CSQ-int8MB will not have pad
                ddl.if (%csqint8_op) {
                    ddl.parametric_loop (%z_front, %input_tensor) {} {
                        %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_output"}
                        ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                        %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_output"}
                        %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                    }
                }
                // QFP8: pad as sending a FP16-0s to SFP from PE
                ddl.if (%qfp8_op) {
                    ddl.parametric_loop (%z_front, %input_tensor) {} {
                        %dst_out_pesfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_input"}
                        ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                    }
                } 
                // Inputs Loop
                ddl.loop(%b_datastage, %bottom_datastage, %output_stick_dim) {} {
                    // LX->PE transfer
                    %src_lxpe = ddl.unit(%input_tensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
                    ddl.if (%csq_ops) {
                        %dst_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="lx_pe_input"}
                        ddl.data_transfer(%src_lxpe, [%dst_lxpe]) {}
                        // PE compute - quantization
                        %pe_src00 = ddl.unit(%input_tensor) {unit="lxlu", data_connect="lx_pe_input"}
                        %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_output"}
                        ddl.compute([%pe_src00, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                        %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_output"}
                        %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                    }
                    ddl.if (%qfp_ops) {
                        // If FP8: send to SFP forwarding through PE
                        %dst_lxsfp = ddl.unit(%input_tensor) {unit="sfp", vias=["pe"], data_connect="sfp_input"}
                        ddl.data_transfer(%src_lxpe, [%dst_lxsfp]) {}
                    }
                } // INPUT loop
                // Pad Back
                // CSQ-int8MB will not have pad
                ddl.if (%csqint8_op) {
                    ddl.parametric_loop (%z_back, %input_tensor) {} {
                        %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_output"}
                        ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                        %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_output"}
                        %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                        ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                    }
                }
                // QFP8: pad as sending a FP16-0s to SFP from PE
                ddl.if (%qfp8_op) {
                    ddl.parametric_loop (%z_back, %input_tensor) {} {
                        %dst_out_pesfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_input"}
                        ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                    }
                } 

                // Outputs Loop
                %sfp_dst00 = ddl.unit(%output_tensor) {unit="lxsu", data_connect="sfp_lx_output"} 
                ddl.if (%csqint8mb_op) {
                    // mb / 2 loop:
                    ddl.loop(%b_datastage, %pack_datastage, %output_stick_dim) {} {
                        %sfp_src00 = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                }
                ddl.if (%qfp8mb_op) {
                    // mb / 2 loop:
                    ddl.loop(%b_datastage, %pack_datastage, %output_stick_dim) {} {
                        %sfp_src00 = ddl.unit(%input_tensor) {unit="pe", data_connect="sfp_input"}
                        %sfp_src02 = ddl.unit(%input_tensor) {unit="pe", data_connect="sfp_input"}
                        ddl.compute([%sfp_src00, %sfp_src02], [%sfp_dst00]) {computetype="PACKMERGE", unit="sfp", repetition=8, indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
                    }
                }
                ddl.if (%csqint8_op) {
                    ddl.parametric_loop (%padded_dim, %output_tensor) {} {
                        %sfp_src00 = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                }
                ddl.if (%qfp8_op) {
                    ddl.parametric_loop (%padded_dim, %output_tensor) {} {
                        %sfp_src00 = ddl.unit(%input_tensor) {unit="pe", data_connect="sfp_input"}
                        ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                    }
                }
                %src_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_lx_output"}
                %dst_sfplx = ddl.unit(%output_tensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_l3_output"}
                ddl.data_transfer(%src_sfplx, [%dst_sfplx]) {}                
            } // B/below
        } // D/B
    } // dataflow
    
    ddl.transformations {
        ddl.disable_transfer_promotion()
    }
} // module
