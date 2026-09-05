//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    // Dimension
    %outer_dim:4 = ddl.dimension{} : index, index, index, index // X, Y, I, MB dimension
    %j_dim = ddl.dimension{} : index // J dimension - may have full z-pad 
    %stick_dim = ddl.dimension{} : index // OUT dimension
    %z_front = ddl.dimension{dim_property="pad_front"} : index // J pad front
    %z_back = ddl.dimension{dim_property="pad_back"} : index   // J pad back
    %z_valid = ddl.dimension{dim_property="pad_valid"} : index
    %out_front = ddl.dimension{dim_property="pad_front"} : index
    %out_back = ddl.dimension{dim_property="pad_back"} : index
    %out_valid = ddl.dimension{dim_property="pad_valid"} : index
    %padded_dim_j = ddl.padded_dimension(primary=%j_dim, padding=[%z_front, %z_back, %z_valid], window=[]) // C dimension J is in stick_dim, else nothing
    %padded_dim_out = ddl.padded_dimension(primary=%stick_dim, padding=[%out_front, %out_back, %out_valid], window=[]) // OUT dimension padding
    
    
    
    // Layout
    %slice_layout_in = ddl.layout (%stick_dim) {is_order_fixed=true}
    %slice_layout_out = ddl.layout (%j_dim) {is_order_fixed=true}
    %slice_layout_out_padded = ddl.layout (%padded_dim_out) {is_order_fixed=true}
    %slice_layout_out_padded_int4 = ddl.layout (%padded_dim_out, %padded_dim_j) {is_order_fixed=true}
    %stick_layout = ddl.layout (%stick_dim) {is_order_fixed=true}
    %global_layout = ddl.layout(%stick_dim, %j_dim, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3 ) {is_order_fixed=false}
    %global_layout_padded = ddl.layout(%padded_dim_out, %padded_dim_j, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3 ) {is_order_fixed=false}


    
    // Data Type
    %type_fp16 = ddl.type{data_type="SEN169_FP16", bit_width=16}
    %type_fp32 = ddl.type{data_type="IEEE_FP32"}
    %type_int8 = ddl.type{data_type="SENINT8",  bit_width=8}
    %type_fp8 = ddl.type{data_type="SEN143_FP8", bit_width=8}
    %type_inter_int8 = ddl.type{data_type ="SENINT8" , bit_width=16}
    %type_int4 = ddl.type{data_type="SENINT4",  bit_width=4}
    %type_int4_16 = ddl.type{data_type ="SENINT4" , bit_width=16}
    %type_int4_8 = ddl.type{data_type ="SENINT4" , bit_width=8}

    // Tensor
    %inp_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias = 
            ddl.tensor(%slice_layout_in, %stick_layout, %global_layout, [%type_fp16, %type_fp32, %type_fp8]) : index, index, index, index, index

    %inter_int8_tensor = ddl.internal_tensor(%inp_tensor, [%type_inter_int8]) : index
    %output_int8_tensor_padded = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_int8]) : index
    %output_fp8_tensor = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_fp8]) : index
    %inter_int4_tensor = ddl.internal_tensor(%inp_tensor, [%type_int4_16]) : index
    %output_int4_tensor_padded = ddl.tensor(%slice_layout_out_padded_int4, %stick_layout, %global_layout_padded, [%type_int4]) : index
    %output_fp16_tensor = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_fp16]) : index
    %output_fp32_tensor = ddl.tensor(%slice_layout_out_padded, %stick_layout, %global_layout_padded, [%type_fp32]) : index

    // Op
    %csqint8ch_op = ddl.operation_bind([], [%inp_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int8_tensor_padded], [%inter_int8_tensor]) {opFuncName="csqint8ch", required=false}
    %qfp8ch_op = ddl.operation_bind([], [%inp_tensor], [%output_fp8_tensor]) {opFuncName="qfp8ch", required=false}
    %csqint4_op = ddl.operation_bind([], [%inp_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int4_tensor_padded], [%inter_int4_tensor]) {opFuncName="csqint4", required=false}
    // %csqint4mb_op = ddl.operation_bind([], [%inp_tensor, %dummy_tensor_alamax, %dummy_tensor_alamin, %dummy_tensor_scale, %dummy_tensor_bias], [%output_int8_tensor], [%inter_int4_tensor]) {opFuncName="csqint4mb", required=false}
    %dl16tofp32_op = ddl.operation_bind([], [%inp_tensor], [%output_fp32_tensor]) {opFuncName="dl16tofp32", required=false}
    %fp32todl16_op = ddl.operation_bind([], [%inp_tensor], [%output_fp16_tensor]) {opFuncName="fp32todl16", required=false}
    %fp8todl16_op = ddl.operation_bind([], [%inp_tensor], [%output_fp16_tensor]) {opFuncName="fp8todl16", required=false} 
    
    // Constraints
    ddl.constraint(){min_num_cores = 1}
    ddl.constraint(%csqint8ch_op, %qfp8ch_op, %csqint4_op, %dl16tofp32_op, %fp32todl16_op, %fp8todl16_op) {min_num_valid = 1, max_num_valid = 1}  //, %csqint4mb_op)
    ddl.constraint(%out_front) {cmp = "equal", value = 0}
    ddl.constraint(%out_back) {cmp = "less", value = 65}

    // Constants
    %scale_const = ddl.get_external_constant(%type_fp16) {name="scaleact", num_elements=1}
    %offset_const = ddl.get_external_constant(%type_fp16) {name="shiftact", num_elements=1}
    %ext_zero = ddl.define_constant(%type_fp16) {value=[0], name="zero"}

    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}

    // Alias tensors
    // cannot define this before Op/Constraints for definition purpose.
    // can use in dataflow 
    %input_tensor = ddl.alias_one_tensor_of(%inp_tensor) 
    %output_tensor = ddl.alias_one_tensor_of(%output_int8_tensor_padded, %output_fp8_tensor, %output_int4_tensor_padded, %output_fp32_tensor, %output_fp16_tensor)

    // Memory Allocation
    %allocate_handler_input_lx = ddl.get_external_data_transfer_allocation(%input_tensor) {memory="lx", data_connect="l3_lx_input"}
    %allocate_handler_output_lx = ddl.get_external_data_transfer_allocation(%output_tensor) {memory="lx", data_connect="lx_l3_output"} 

    // Dataflow
    ddl.dataflow { 
        %d_datastage = ddl.get_external_datastage {property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}

        %bottom_datastage = ddl.datastage {strategy = "minimize"}
        %stick_compose_datastage = ddl.datastage {strategy = "maximize", allow_epilogue=true}
        ddl.datastage_constraint(%stick_compose_datastage, %bottom_datastage, %stick_dim) {max="2"}
        ddl.datastage_constraint(%stick_compose_datastage, %bottom_datastage, %j_dim) {max="2"}

        %csq_ops = ddl.condition_or(%csqint8ch_op, %csqint4_op)  //, %csqint4mb_op)
        %chdowncast_ops = ddl.condition_or(%csqint8ch_op, %qfp8ch_op, %fp32todl16_op)
        %chupcast_ops = ddl.condition_or(%dl16tofp32_op, %fp8todl16_op)
        %peforwardpadding_ops = ddl.condition_or(%fp32todl16_op, %qfp8ch_op)
        %peforward_ops = ddl.condition_or(%dl16tofp32_op, %fp32todl16_op, %qfp8ch_op, %fp8todl16_op)


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
        ddl.if(%peforwardpadding_ops) {
            %src_pe_zero = ddl.unit(%ext_zero) {unit="constant", data_connect= "zero_connect"} 
            ddl.data_transfer(%src_pe_zero, [%pe_zero_lrf]) {}
        }

        // ddl.constraint - ensure pack datastage is 2 or 4

        ddl.loop(%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %j_dim, %stick_dim) {} {
            ddl.loop(%b_datastage, %bottom_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3) {} {

                // Inputs Loop
                ddl.if (%csqint4_op) {
                    ddl.loop(%b_datastage, %stick_compose_datastage, %j_dim, %stick_dim) {label="across_sticks"} {
                        %cond_first_j = ddl.condition(%j_dim){loop_label="across_sticks", condition="eq", value_expr="first"}
                        ddl.if(%cond_first_j) {
                            ddl.parametric_loop (%z_front, %input_tensor) {} {
                                // send full OUT (128 dimension)
                                %pe_dst00 = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                        }
                        ddl.loop(%stick_compose_datastage, %bottom_datastage, %j_dim) {} {
                            ddl.loop(%stick_compose_datastage, %bottom_datastage, %stick_dim) {} {
                                // LX->PE transfer
                                %src_lxpe = ddl.unit(%input_tensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
                                %dst_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="lx_pe_input"}
                                ddl.data_transfer(%src_lxpe, [%dst_lxpe]) {}
                                // PE compute - quantization
                                %pe_src00 = ddl.unit(%input_tensor) {unit="lxlu", data_connect="lx_pe_input"}
                                %pe_dst00 = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%pe_src00, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                            %cond_last_channel = ddl.condition(%stick_dim){loop_label="across_sticks", condition="eq", value_expr="last"}
                            ddl.if(%cond_last_channel) {
                                ddl.parametric_loop (%out_back, %input_tensor) {} {
                                    %pe_dst00 = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                    ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                    %src_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                    %dst_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="sfp_input"}
                                    ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                                }
                            }
                        }
                        %cond_last_j = ddl.condition(%j_dim){loop_label="across_sticks", condition="eq", value_expr="last"}
                        ddl.if(%cond_last_j) {
                            ddl.parametric_loop (%z_back, %input_tensor) {} {
                                // send full OUT (128 dimension)
                                %pe_dst00 = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int4_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                } else {
                    // Pad Front
                    ddl.if (%csqint8ch_op) {
                        ddl.parametric_loop (%z_front, %input_tensor) {} {
                            ddl.parametric_loop (%padded_dim_out, %output_tensor) {} {
                                // send full OUT (128 dimension) stick, so 2 zero input sticks
                                %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                    // QFP8: pad as sending a FP16-0s to SFP from PE
                    ddl.if (%qfp8ch_op) {
                        ddl.parametric_loop (%z_front, %input_tensor) {} {
                            ddl.parametric_loop (%padded_dim_out, %output_tensor) {} {
                                %dst_out_pesfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                    ddl.loop(%b_datastage, %bottom_datastage, %j_dim) {} {
                        // Pad Front - OUT 
                        // should not exist
                        ddl.loop(%b_datastage, %bottom_datastage, %stick_dim) {} {
                            // LX->PE transfer
                            %src_lxpe = ddl.unit(%input_tensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
                            ddl.if (%csqint8ch_op) {
                                %dst_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="lx_pe_input"}
                                ddl.data_transfer(%src_lxpe, [%dst_lxpe]) {}
                                // PE compute - quantization
                                %pe_src00 = ddl.unit(%input_tensor) {unit="lxlu", data_connect="lx_pe_input"}
                                %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%pe_src00, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                            ddl.if (%peforward_ops) {
                                //LX -> PE -> SFP transfer:   send to SFP forwarding through PE
                                %dst_lxsfp = ddl.unit(%input_tensor) {unit="sfp", vias=["pe"], data_connect="sfp_input"}
                                ddl.data_transfer(%src_lxpe, [%dst_lxsfp]) {}
                            }
                        }
                        // Pad Back - IN/OUT dimension
                        ddl.if (%csqint8ch_op) {
                            ddl.parametric_loop (%out_back, %input_tensor) {} {
                                %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                        }
                        // Pad Back - IN/OUT dimension
                        // QFP8: pad as sending a FP16-0s to SFP from PE
                        ddl.if (%peforwardpadding_ops) {
                            ddl.parametric_loop (%out_back, %input_tensor) {} {
                                %dst_out_pesfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                    // Pad Back
                    ddl.if (%csqint8ch_op) {
                        ddl.parametric_loop (%z_back, %input_tensor) {} {
                            ddl.parametric_loop (%padded_dim_out, %output_tensor) {} {
                                // send full OUT (128 dimension) stick
                                %pe_dst00 = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="pe_sfp_output"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                ddl.compute([%zero_const, %pe_scale, %pe_offset], [%pe_dst00]) {computetype="FMA16", unit="pe"}
                                %src_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="pe_sfp_output"}
                                %dst_out_pesfp = ddl.unit(%inter_int8_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                    // pad as sending a FP16-0s to SFP from PE
                    ddl.if (%peforwardpadding_ops) {
                        ddl.parametric_loop (%z_back, %input_tensor) {} {
                            ddl.parametric_loop (%padded_dim_out, %output_tensor) {} {
                                // send full OUT (128 dimension) stick
                                %dst_out_pesfp = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_input"}
                                ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                                ddl.data_transfer(%pe_zero_lrf, [%dst_out_pesfp]) {}
                            }
                        }
                    }
                } // INPUT loop

                // Outputs Loop
                ddl.parametric_loop (%padded_dim_j, %output_tensor) {} {
                    %sfp_dst00 = ddl.unit(%output_tensor) {unit="lxsu", data_connect="sfp_lx_output"} 
                    ddl.parametric_loop (%padded_dim_out, %output_tensor) {} {
                        ddl.if (%csqint8ch_op) {
                            %sfp_src00 = ddl.unit(%inter_int8_tensor) {unit="pe", data_connect="sfp_input"}
                            ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                        }
                        ddl.if (%csqint4_op) {
                            %sfp_src00 = ddl.unit(%inter_int4_tensor) {unit="pe", data_connect="sfp_input"}
                            ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                        }
                        ddl.if (%qfp8ch_op) {
                            %sfp_src00 = ddl.unit(%input_tensor) {unit="pe", data_connect="sfp_input"}
                            ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}
                        }
                        ddl.if(%fp32todl16_op) {
                            ddl.opaque(%output_tensor)
                                {unit="sfp", op="FP32TODL16", input_output_registers=[], internal_registers=[],
                                 max_unroll_factor=1, params={"in0"="pe", "out0"="result"},
                                 input_data_connects=["sfp_input"], output_data_connects=["sfp_lx_output"]}
                        }
                        ddl.if(%chdowncast_ops) {
                           %src_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_lx_output"}
                           %dst_sfplx = ddl.unit(%output_tensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_l3_output"}
                           ddl.data_transfer(%src_sfplx, [%dst_sfplx]) {}  
                        }
                    }
                    ddl.parametric_loop (%padded_dim_out, %input_tensor) {} {
                        ddl.if (%dl16tofp32_op) {
                            ddl.opaque(%input_tensor)
                                {unit="sfp", op="DL16TOFP32", input_output_registers=[], internal_registers=[],
                                max_unroll_factor=1, params={"in0"="pe", "out0"="result"},
                                input_data_connects=["sfp_input"], output_data_connects=["sfp_lx_output"]}
                        }
                        ddl.if (%fp8todl16_op) {
                            // Casting: fp8 one stick (top, bot) -> (fp16 stick0) (fp16 stick1)
                            %pe_src = ddl.unit(%input_tensor) {unit="pe", data_connect="sfp_input"}
                            %sfp_lrf_allocation00 = ddl.allocate(%input_tensor) {memory="sfplrf"}
                            %sfp_src00 = ddl.unit(%input_tensor, %sfp_lrf_allocation00) {unit="sfp", data_connect="sfp_input_lrf0"}
                            ddl.data_transfer(%pe_src, [%sfp_src00]) {}
                            ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="SHUFFLE", unit="sfp", repetition=8, indices=[0,1,2,3,4,5,6,7]}
                            ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="SHUFFLE", unit="sfp", repetition=8, indices=[8,9,10,11,12,13,14,15]}
                        }
                        ddl.if(%chupcast_ops) {
                            %src_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_lx_output"}
                            %dst_sfplx = ddl.unit(%output_tensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_l3_output"}
                            ddl.data_transfer(%src_sfplx, [%dst_sfplx]) {}
                        }
                    }
                }
                ddl.if(%csqint4_op) {
                    %src_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_lx_output"}
                    %dst_sfplx = ddl.unit(%output_tensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_l3_output"}
                    ddl.data_transfer(%src_sfplx, [%dst_sfplx]) {} 
                }
            } // B/below
        } // D/B
    } // dataflow
    
    ddl.transformations {
        ddl.disable_transfer_promotion()
    }
} // module
