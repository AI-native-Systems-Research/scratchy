//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

// Sum Mean Max AbsMax Min Exx2

module {
    // Dimension
    %outer_dim:5 = ddl.dimension{} : index, index, index, index, index // X, Y, I, J, MB dimension
    %reduce_dim:4 = ddl.dimension{} : index, index, index, index // OUT dimension

    // Layout
    %slice_layout_stick = ddl.layout (%reduce_dim#0) {is_order_fixed=true}
    %stick_layout_stick = ddl.layout (%reduce_dim#0) {is_order_fixed=true}

    // Nonstick layout
    %slice_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}
    %stick_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}

    %global_layout_input = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3) {is_order_fixed=false}
    %global_layout_output = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {is_order_fixed=false}

    // DataType 
    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
    // Tensor stick
    %input_tensor_stick = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_input, [%type_fp16]) : index
    %output_tensor_stick = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_output, [%type_fp16]) : index

    // Tensor non stick
    %input_tensor_nonstick = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_input, [%type_fp16]) : index
    %output_tensor_nonstick = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_output, [%type_fp16]) : index

    %output_tensor_x2 = ddl.internal_tensor(%output_tensor_stick, [%type_fp16]) : index
    %output_tensor_reduce_stick = ddl.internal_tensor(%output_tensor_stick, [%type_fp16]) : index
    %output_tensor_reduce_nonstick = ddl.internal_tensor(%output_tensor_nonstick, [%type_fp16]) : index
    %output_tensor_x2_reduce = ddl.internal_tensor(%output_tensor_stick, [%type_fp16]) : index

    // Op
    %sum_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_reduce_stick]) {opFuncName="sum", required=false}
    %sum_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="sumnonstick", required=false}

    %mean_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_reduce_stick]) {opFuncName="mean", required=false}
    %mean_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="meannonstick", required=false}

    %max_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_reduce_stick]) {opFuncName="max", required=false}
    %max_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="maxnonstick", required=false}

    %min_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_reduce_stick]) {opFuncName="min", required=false}
    %min_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="minnonstick", required=false}

    %exx2_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_x2, %output_tensor_reduce_stick, %output_tensor_x2_reduce]) {opFuncName="exx2", required=false}
    %exx2_zeromean_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_x2, %output_tensor_reduce_stick, %output_tensor_x2_reduce]) {opFuncName="exx2_zeromean", required=false}

    %absmax_op = ddl.operation_bind([%type_fp16], [%input_tensor_stick, %output_tensor_stick], [%output_tensor_stick], [%output_tensor_reduce_stick]) {opFuncName="absmax", required=false}
    %absmax_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="absmaxnonstick", required=false}

    %prod_nonstick_op = ddl.operation_bind([%type_fp16], [%input_tensor_nonstick, %output_tensor_nonstick], [%output_tensor_nonstick], [%output_tensor_reduce_nonstick]) {opFuncName="prodnonstick", required=false}

    // Tensor
    %input_tensor = ddl.alias_one_tensor_of(%input_tensor_stick, %input_tensor_nonstick)
    %output_tensor = ddl.alias_one_tensor_of(%output_tensor_stick, %output_tensor_nonstick)
    %output_tensor_reduce = ddl.alias_one_tensor_of(%output_tensor_reduce_stick, %output_tensor_reduce_nonstick)

    // p-sum
    %psum_op = ddl.operation_bind([%type_fp16], [%output_tensor], [%output_tensor]) {opFuncName="genericpartialreduction", required=false}

    // Constant 
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}
    %zero_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %one_const_reg = ddl.define_constant(%type_fp16) {value=[0x3E00], name="one"}
    %negInf_const_reg = ddl.define_constant(%type_fp16){value=[0xFFFE], name="negInf"}
    %posInf_const_reg = ddl.define_constant(%type_fp16){value=[0x7FFE], name="posInf"}

    %scaling_factor_const_reg = ddl.get_external_constant(%type_fp16){name="scaling_factor", num_elements=1} // (1 / N)
    %nfwd0_const = ddl.operand_constant {name="nfwd0"}
    %nfwd2_const = ddl.operand_constant {name="nfwd2"}
    %exx2_div = ddl.get_external_constant(%type_fp16){name="exx2scale", num_elements=1}

    // Allocation
    %input_lx_allocation = ddl.get_external_data_transfer_allocation (%input_tensor) {memory="lx", data_connect="l3_lx_input"} 
    %output_lx_allocation = ddl.get_external_data_transfer_allocation (%output_tensor) { memory="lx", data_connect="lxsu_output"}

    // Constraints
    ddl.constraint(%sum_op, %sum_nonstick_op, %mean_op, %mean_nonstick_op, %max_op, %max_nonstick_op, %min_op, %min_nonstick_op, %exx2_op, %exx2_zeromean_op, %absmax_op, %absmax_nonstick_op, %prod_nonstick_op) {min_num_valid = 1, max_num_valid = 1}
    ddl.constraint() {min_num_cores = 1}

    // Dataflow
    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage{property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %subchunk_datastage = ddl.datastage {strategy="maximize"}
        %above_interleave = ddl.datastage {strategy="maximize", allow_epilogue=true}
        %below_interleave = ddl.datastage {strategy="minimize"}
        ddl.datastage_constraint(%subchunk_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3) {max="64"}
        ddl.datastage_constraint(%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {values=["1", "2", "4"]}

        %max_or_maxnonstick_op = ddl.condition_or(%max_op, %max_nonstick_op)
        %min_or_minnonstick_op = ddl.condition_or(%min_op, %min_nonstick_op)
        %absmax_or_absmaxnonstick_op = ddl.condition_or(%absmax_op, %absmax_nonstick_op)

        // Constant:
        %zero_pe_allocation = ddl.allocate(%zero_const_reg) {memory="pelrf"} 
        %src_pe_zero = ddl.unit(%zero_const_reg) {unit="constant", data_connect= "zero_connect"} 
        %pe_zero_lrf = ddl.unit(%zero_const_reg, %zero_pe_allocation) {unit="pe", data_connect= "zero_pe_lrf"}
        ddl.data_transfer(%src_pe_zero, [%pe_zero_lrf]) {}

        %zero_sfp_allocation = ddl.allocate(%zero_const_reg) {memory="sfplrf"} 
        %src_sfp_zero = ddl.unit(%zero_const_reg) {unit="constant", data_connect= "zero_connect"} 
        %dst_sfp_zero = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
        ddl.if (%exx2_zeromean_op) {
            // for zero-mean
            ddl.data_transfer(%src_sfp_zero, [%dst_sfp_zero]) {}
        }
        
        %negInf_pe_allocation = ddl.allocate(%negInf_const_reg) {memory="pelrf"} 
        %src_pe_negInf = ddl.unit(%negInf_const_reg) {unit="constant", data_connect= "negInf_connect"} 
        %pe_negInf_lrf = ddl.unit(%negInf_const_reg, %negInf_pe_allocation) {unit="pe", data_connect= "negInf_pe_lrf"}
        ddl.if (%max_or_maxnonstick_op) {
            ddl.data_transfer(%src_pe_negInf, [%pe_negInf_lrf]) {}
        }

        %posInf_pe_allocation = ddl.allocate(%posInf_const_reg) {memory="pelrf"}
        %src_pe_posInf = ddl.unit(%posInf_const_reg) {unit="constant", data_connect="posInf_connect"}
        %pe_posInf_lrf = ddl.unit(%posInf_const_reg, %posInf_pe_allocation) {unit="pe", data_connect="posInf_pe_lrf"}
        ddl.if (%min_or_minnonstick_op) {
            ddl.data_transfer(%src_pe_posInf, [%pe_posInf_lrf]) {}
        }

        // scaling factor constant for mean
        %scaling_factor_pe_allocation = ddl.allocate(%scaling_factor_const_reg) {memory="pelrf"}
        %src_pe_scaling_factor = ddl.unit(%scaling_factor_const_reg) {unit="constant", data_connect= "scaling_factor_connect"}
        %pe_scaling_factor_lrf = ddl.unit(%scaling_factor_const_reg, %scaling_factor_pe_allocation) {unit="pe", data_connect= "scaling_factor_pe_lrf"}
        %is_mean_or_meannonstick = ddl.condition_or(%mean_op, %mean_nonstick_op)
        %is_sum_or_mean_op = ddl.condition_or(%sum_op, %mean_op)

        ddl.if (%is_mean_or_meannonstick) {
            ddl.data_transfer(%src_pe_scaling_factor, [%pe_scaling_factor_lrf]) {}
        }

        // one constant for prod initialization
        %one_pe_allocation = ddl.allocate(%one_const_reg) {memory="pelrf"}
        %src_pe_one = ddl.unit(%one_const_reg) {unit="constant", data_connect="one_connect"}
        %pe_one_lrf = ddl.unit(%one_const_reg, %one_pe_allocation) {unit="pe", data_connect="one_pe_lrf"}

        ddl.if (%prod_nonstick_op) {
            ddl.data_transfer(%src_pe_one, [%pe_one_lrf]) {}
        }

        %exx2div_sfp_allocation = ddl.allocate(%exx2_div) {memory="sfplrf"} 
        %src_exx2div_const = ddl.unit(%exx2_div) {unit="constant", data_connect= "exx2div_connect"} 
        %dst_exx2div_sfp = ddl.unit(%exx2_div, %exx2div_sfp_allocation) {unit="sfp", data_connect= "exx2div_sfp_lrf"}
        %is_exx2_or_exx2_zeromean_op = ddl.condition_or(%exx2_op, %exx2_zeromean_op)
        %not_exx2_or_exx2_zeromean_op = ddl.condition_not(%is_exx2_or_exx2_zeromean_op)
        ddl.if (%is_exx2_or_exx2_zeromean_op) {
            ddl.data_transfer(%src_exx2div_const, [%dst_exx2div_sfp]) {}
        }

        ddl.loop (%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3){label="chunk_loop"} {
            %cond_first_loopreduce_dim0 = ddl.condition(%reduce_dim#0){loop_label="chunk_loop", condition="eq", value_expr="first"}
            %cond_first_loopreduce_dim1 = ddl.condition(%reduce_dim#1){loop_label="chunk_loop", condition="eq", value_expr="first"}
            %cond_first_loopreduce_dim2 = ddl.condition(%reduce_dim#2){loop_label="chunk_loop", condition="eq", value_expr="first"}
            %cond_first_loopreduce_dim3 = ddl.condition(%reduce_dim#3){loop_label="chunk_loop", condition="eq", value_expr="first"}
            %cond_first_loopreduce_dims = ddl.condition_and(%cond_first_loopreduce_dim0, %cond_first_loopreduce_dim1, %cond_first_loopreduce_dim2, %cond_first_loopreduce_dim3)
            %cond_not_first_loopreduce_dims = ddl.condition_not(%cond_first_loopreduce_dims)
            %cond_last_loopreduce_dim0 = ddl.condition(%reduce_dim#0){loop_label="chunk_loop", condition="eq", value_expr="last"}
            %cond_last_loopreduce_dim1 = ddl.condition(%reduce_dim#1){loop_label="chunk_loop", condition="eq", value_expr="last"}
            %cond_last_loopreduce_dim2 = ddl.condition(%reduce_dim#2){loop_label="chunk_loop", condition="eq", value_expr="last"}
            %cond_last_loopreduce_dim3 = ddl.condition(%reduce_dim#3){loop_label="chunk_loop", condition="eq", value_expr="last"}
            %cond_last_loopreduce_dims = ddl.condition_and(%cond_last_loopreduce_dim0, %cond_last_loopreduce_dim1, %cond_last_loopreduce_dim2, %cond_last_loopreduce_dim3)
            %psum_start, %psum_end, %next_core, %prev_core = ddl.core_to_core_communication(%reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3)

            ddl.if(%psum_end) {
                ddl.if(%cond_not_first_loopreduce_dims) {
                    ddl.sync {units=["lxsu"], is_receive=false, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
                    ddl.sync {units=["lxlu"], is_receive=true, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
                }
            }

            ddl.loop (%b_datastage, %above_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4){} {
                %pe_lrf_allocation_subchunk = ddl.allocate(%output_tensor_reduce) {memory="pelrf"}
                %sfp_lrf_allocation_reduce = ddl.allocate(%output_tensor_reduce) {memory="sfplrf"}
                %pe_subchunk_lrf = ddl.unit(%output_tensor_reduce, %pe_lrf_allocation_subchunk) {unit="pe", data_connect="lxpe_output"}
                %sfp_out_lrf = ddl.unit(%output_tensor_reduce, %sfp_lrf_allocation_reduce) {unit="sfp", data_connect="lxsfp_output"}
                %sfp_lrf_allocation_x2_reduce = ddl.allocate(%output_tensor_x2_reduce) {memory="sfplrf"}
                %sfp_out_lrf_x2 = ddl.unit(%output_tensor_x2_reduce, %sfp_lrf_allocation_x2_reduce) {unit="sfp", data_connect="lxsfp_output_x2"}
                %is_sum_min_max_mean_absmax_prod_nonstick_op = ddl.condition_or(%sum_nonstick_op, %max_nonstick_op, %min_nonstick_op, %mean_nonstick_op, %absmax_nonstick_op, %prod_nonstick_op)
                ddl.if(%psum_end) {
                    ddl.if(%cond_not_first_loopreduce_dims) {
                        %src_out_lxpesfp = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxlu", data_connect="lxsu_output"}
                        ddl.data_transfer(%src_out_lxpesfp, [%sfp_out_lrf]) {}
                        ddl.if (%exx2_op) {
                            %src_out_lxpesfp_x2 = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxlu", data_connect="lxsu_output", stick_replicated_dim_offset_elements=8}
                            ddl.data_transfer(%src_out_lxpesfp_x2, [%sfp_out_lrf_x2])
                        }
                    }
                }

                %pe_lrf_allocation = ddl.allocate(%output_tensor) {memory="pelrf"}
                %pe_reduce_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                
                // for Exx2
                %pe_lrf_x2_allocation = ddl.allocate(%output_tensor_x2) {memory="pelrf"}
                %pe_reduce_x2_lrf = ddl.unit(%output_tensor_x2, %pe_lrf_x2_allocation) {unit="pe", data_connect="inter_stick_output_x2"}

                // explicitly initialize to 0
                // LRF x 1 + 0 -> LRF
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.if (%max_or_maxnonstick_op) {
                        ddl.compute([%pe_negInf_lrf, %one_const, %zero_const], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                    } else {
                        ddl.if (%min_or_minnonstick_op) {
                            ddl.compute([%pe_posInf_lrf, %one_const, %zero_const], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                        } else {
                            ddl.if (%prod_nonstick_op) {
                                ddl.compute([%pe_one_lrf, %one_const, %zero_const], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                            } else {
                                ddl.compute([%pe_zero_lrf, %one_const, %zero_const], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                            }
                        }
                    }
                }
                ddl.if (%exx2_op) {
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        // explicitly initialize to 0
                        // LRF x 1 + 0 -> LRF              
                        ddl.compute([%pe_zero_lrf, %one_const, %zero_const], [%pe_reduce_x2_lrf]) {computetype="FMA16", unit="pe"}
                    }
                }
                // Inner-loop of reduction
                ddl.loop (%b_datastage, %subchunk_datastage, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3) {} {
                    // explicitly initialize to 0
                    // LRF x 1 + 0 -> LRF
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.if (%max_or_maxnonstick_op) {
                            ddl.compute([%pe_negInf_lrf, %one_const, %zero_const], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                        } else {
                            ddl.if (%min_or_minnonstick_op) {
                                ddl.compute([%pe_posInf_lrf, %one_const, %zero_const], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                            } else {
                                ddl.if (%prod_nonstick_op) {
                                    ddl.compute([%pe_one_lrf, %one_const, %zero_const], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                                } else {
                                    ddl.if (%not_exx2_or_exx2_zeromean_op) {
                                        ddl.compute([%pe_zero_lrf, %one_const, %zero_const], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                                    }
                                }
                            }
                        }
                    }
                    // ddl.if (%exx2_op) {
                    //     ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    //         // explicitly initialize to 0
                    //         // LRF x 1 + 0 -> LRF              
                    //         ddl.compute([%pe_zero_lrf, %one_const, %zero_const], [%pe_reduce_x2_lrf]) {computetype="FMA16", unit="pe"}
                    //     }
                    // }

                    ddl.loop (%subchunk_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3) {} {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            // LX->PE 
                            %src_inp_lxpe = ddl.unit(%input_tensor, %input_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
                            %dst_inp_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="pe_lx_input"}
                            // For Exx2, lx->pelrf first to reuse the data
                            %pe_input_lrf_allocation = ddl.allocate(%input_tensor) {memory="pelrf"}
                            %dst_inp_lxpelrf = ddl.unit(%input_tensor, %pe_input_lrf_allocation) {unit="pe", data_connect="pe_lx_input_lrf"}
                            
                            %pe_lxlu_input = ddl.unit(%input_tensor) {unit="lxlu", data_connect="pe_lx_input"}

                            // PE Inter-stick subchunk reduction
                            // chunk accumulation in PE

                            %sum_or_sumnonstick_op = ddl.condition_or(%sum_op, %sum_nonstick_op)
                            ddl.if (%sum_or_sumnonstick_op) {
                                // chunk accumulation in PE: 0 + lxlu => lrf 
                                // chunk accumulation in PE: lrf + lxlu => lrf
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %one_const, %pe_subchunk_lrf], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                            }

                            ddl.if(%is_mean_or_meannonstick) {
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %pe_scaling_factor_lrf, %pe_subchunk_lrf], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                            }
                            ddl.if (%max_or_maxnonstick_op) {
                                // chunk max in PE: max(lxlu, lxlu) => lrf 
                                // chunk max in PE: max(lrf, lxlu) => lrf
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %pe_subchunk_lrf], [%pe_subchunk_lrf]) {computetype="FMAX", unit="pe"}
                            }
                            ddl.if (%absmax_or_absmaxnonstick_op) {
                                // chunk absmax in PE: absmax(lrf, lxlu) => lrf
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %pe_subchunk_lrf], [%pe_subchunk_lrf]) {computetype="FABSMAX", unit="pe"}
                            }
                            ddl.if (%min_or_minnonstick_op) {
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %pe_subchunk_lrf], [%pe_subchunk_lrf]) {computetype="FMIN", unit="pe"}
                            }
                            ddl.if (%prod_nonstick_op) {
                                // chunk prod in PE: prod(lrf, lxlu) => lrf
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                                ddl.compute([%pe_lxlu_input, %pe_subchunk_lrf, %zero_const], [%pe_subchunk_lrf]) {computetype="FMA16", unit="pe"}
                            }
                            ddl.if (%exx2_zeromean_op) {
                                // pe input -> Rx
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpelrf]) {}
                                // chunk accumulation in PE: lrf + Rx x Rx => lrf
                                ddl.compute([%dst_inp_lxpelrf, %dst_inp_lxpelrf, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                            }
                            ddl.if (%exx2_op) {
                                // pe input -> Rx
                                ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpelrf]) {}
                                // chunk accumulation in PE: lrf + Rx => lrf
                                // chunk accumulation in PE: lrf + Rx x Rx => lrf
                                ddl.compute([%dst_inp_lxpelrf, %dst_inp_lxpelrf, %pe_reduce_x2_lrf], [%pe_reduce_x2_lrf]) {computetype="FMA16", unit="pe"}
                                ddl.compute([%dst_inp_lxpelrf, %one_const, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                            }
                        }
                    } // end subchunk reduction

                    // PE across-subchunks reduction
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.if (%absmax_or_absmaxnonstick_op) {
                            ddl.compute([%pe_subchunk_lrf, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FABSMAX", unit="pe"}
                        } else {
                            ddl.if (%max_or_maxnonstick_op) {
                                ddl.compute([%pe_subchunk_lrf, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FMAX", unit="pe"}
                            } else {
                                ddl.if (%min_or_minnonstick_op) {
                                    ddl.compute([%pe_subchunk_lrf, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FMIN", unit="pe"}
                                } else {
                                    ddl.if (%prod_nonstick_op) {
                                        ddl.compute([%pe_subchunk_lrf, %pe_reduce_lrf, %zero_const], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                                    } else {
                                        ddl.if (%not_exx2_or_exx2_zeromean_op) {
                                            ddl.compute([%pe_subchunk_lrf, %one_const, %pe_reduce_lrf], [%pe_reduce_lrf]) {computetype="FMA16", unit="pe"}
                                        }
                                    }
                                    // ddl.if (%exx2_op) {
                                    //     ddl.compute([%sfp_lrf_x2_dst00, %one_const, %sfp_psum_src02], [%sfp_lrf_x2_dst00]) {computetype="FMA16", unit="pe"}
                                    // }
                                }
                            }
                        }
                    }
                } // end across-sticks reduction

                %is_not_sum_min_max_mean_absmax_prod_nonstick_op = ddl.condition_not(%is_sum_min_max_mean_absmax_prod_nonstick_op)
                %pe_nfwd_dst3_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="pe_reduce_output"}
                %pe_nfwd_dst3_x2_lrf = ddl.unit(%output_tensor_x2, %pe_lrf_x2_allocation) {unit="pe", data_connect="pe_reduce_x2_output"}
                ddl.if(%is_not_sum_min_max_mean_absmax_prod_nonstick_op) {
                    // PE nfwd reduction
                    %pe_nfwd_src_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                    %pe_nfwd_dst1_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                    %pe_nfwd_dst2_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}

                    %pe_nfwd_src_x2_lrf = ddl.unit(%output_tensor_x2, %pe_lrf_x2_allocation) {unit="pe", data_connect="inter_stick_x2_output"}
                    %pe_nfwd_dst1_x2_lrf = ddl.unit(%output_tensor_x2, %pe_lrf_x2_allocation) {unit="pe", data_connect="inter_stick_x2_output"}
                    %pe_nfwd_dst2_x2_lrf = ddl.unit(%output_tensor_x2, %pe_lrf_x2_allocation) {unit="pe", data_connect="inter_stick_x2_output"}
                    // explicitly perform transfer.

                    ddl.if (%is_sum_or_mean_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %one_const, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FMA16", unit="pe"}
                        }
                    }
                    ddl.if (%absmax_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FABSMAX", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FABSMAX", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FABSMAX", unit="pe"}
                        }
                    }
                    ddl.if (%max_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FMAX", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FMAX", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FMAX", unit="pe"}
                        }
                    }
                    ddl.if (%min_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FMIN", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FMIN", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FMIN", unit="pe"}
                        }
                    }
                    ddl.if (%exx2_zeromean_op) {
                        // TODO: possible combine with Sum
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %one_const, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FMA16", unit="pe"}
                        }
                    }
                    ddl.if (%exx2_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %pe_nfwd_src_x2_lrf], [%pe_nfwd_dst1_x2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_lrf, %one_const, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%pe_nfwd_dst1_x2_lrf, %one_const, %nfwd2_const], [%pe_nfwd_dst2_x2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FMA16", unit="pe"}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%nfwd0_const, %one_const, %nfwd2_const], [%pe_nfwd_dst3_x2_lrf]) {computetype="FMA16", unit="pe"}
                        }
                    }
                }

                // PE->SFP
                %output_sfp_lrf_allocation = ddl.allocate(%output_tensor) {memory="sfplrf"}
                %output_sfp_x2_lrf_allocation = ddl.allocate(%output_tensor_x2) {memory="sfplrf"}
                %dst_out_pesfp = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_splat_input"}
                %src_out_pesfp = ddl.unit(%output_tensor) {unit="pe", data_connect="sfp_splat_input"}
                %sfp_zero_lrf = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}

                %sfp_lrf_src00 = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_input_output"}
                %sfp_lrf_dst00 = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_input_output"}
                %sfp_lrf_x2_src00 = ddl.unit(%output_tensor_x2, %output_sfp_x2_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_x2_input"}
                %sfp_lrf_x2_dst00 = ddl.unit(%output_tensor_x2, %output_sfp_x2_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_x2_output"}
                ddl.if(%is_sum_min_max_mean_absmax_prod_nonstick_op) {
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.data_transfer(%pe_reduce_lrf, [%dst_out_pesfp]) {}
                    }
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.compute([%src_out_pesfp, %one_const, %zero_const], [%sfp_lrf_dst00]) {computetype="FMA16", unit="sfp"}
                    }
                } else {
                    %dst_out_x2_pesfp = ddl.unit(%output_tensor_x2) {unit="sfp", data_connect="sfp_splat_x2_input"}
                    %src_out_x2_pesfp = ddl.unit(%output_tensor_x2) {unit="pe", data_connect="sfp_splat_x2_input"}

                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.data_transfer(%pe_nfwd_dst3_lrf, [%dst_out_pesfp]) {}
                    }
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.if (%exx2_op) {
                            // Second transfer for x2
                            ddl.data_transfer(%pe_nfwd_dst3_x2_lrf, [%dst_out_x2_pesfp]) {}
                        }
                    }
                    %dst_out_sfpsplat = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_input_output"}
                    %dst_out_x2_sfpsplat = ddl.unit(%output_tensor_x2, %output_sfp_x2_lrf_allocation) {unit="sfp", data_connect="sfp_lrf_x2_input"}

                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.compute([%src_out_pesfp], [%dst_out_sfpsplat]) {computetype="SPLAT", unit="sfp"}
                    }
                    ddl.if (%exx2_op) {
                        // one more splat
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute([%src_out_x2_pesfp], [%dst_out_x2_sfpsplat]) {computetype="SPLAT", unit="sfp"}
                        }
                    }

                    // SFP Reduce
                    ddl.if (%is_sum_or_mean_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=1}
                        }
                    }
                    ddl.if (%max_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=8}
                        }
                    }
                    ddl.if (%absmax_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=10}
                        }
                    }
                    ddl.if (%min_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=12}
                        }
                    }
                    ddl.if (%exx2_zeromean_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=1}
                        }
                    }
                    ddl.if (%exx2_op) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=1}
                        }
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.compute ([%sfp_lrf_x2_src00], [%sfp_lrf_x2_dst00]) {computetype="REDUCE", unit="sfp", mode=1}
                        }
                    }
                }

                ddl.if (%psum_op) {
                    // Do p-sum for stick
                    %sfp_psum_src02 = ddl.unit(%output_tensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
                    %sfp_psum_dst00_sfpring = ddl.unit(%output_tensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
                    %sfp_psum_dst00_lxsu = ddl.unit(%output_tensor) {unit="lxsu", data_connect="lxsu_output"}
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.if (%psum_start) {
                            // psum: sfplrf + 0 -> sfpring
                            ddl.compute([%sfp_lrf_src00, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                            ddl.if (%exx2_op) {
                                ddl.compute([%sfp_lrf_x2_dst00, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                            }
                        } else {
                            ddl.if (%psum_end) {
                                // psum: op(sfplrf, sfpring) -> lxsu
                                ddl.if (%absmax_or_absmaxnonstick_op) {
                                    ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_lrf_dst00]) {computetype="FABSMAX", unit="sfp"}
                                } else {
                                    ddl.if (%max_or_maxnonstick_op) {
                                        ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_lrf_dst00]) {computetype="FMAX", unit="sfp"}
                                    } else {
                                        ddl.if (%min_or_minnonstick_op) {
                                            ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_lrf_dst00]) {computetype="FMIN", unit="sfp"}
                                        } else {
                                            ddl.if (%prod_nonstick_op) {
                                                ddl.compute([%sfp_lrf_src00, %sfp_psum_src02, %zero_const], [%sfp_lrf_dst00]) {computetype="FMA16", unit="sfp"}
                                            } else {
                                                ddl.compute([%sfp_lrf_src00, %one_const, %sfp_psum_src02], [%sfp_lrf_dst00]) {computetype="FMA16", unit="sfp"}
                                                ddl.if (%exx2_op) {
                                                    ddl.compute([%sfp_lrf_x2_dst00, %one_const, %sfp_psum_src02], [%sfp_lrf_x2_dst00]) {computetype="FMA16", unit="sfp"}
                                                }
                                            }
                                        }
                                    }
                                }
                            } else { // middle cores
                                // psum: op(sfplrf, sfpring) -> sfpring
                                ddl.if (%absmax_or_absmaxnonstick_op) {
                                    ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FABSMAX", unit="sfp"}
                                } else {
                                    ddl.if (%max_or_maxnonstick_op) {
                                        ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMAX", unit="sfp"}
                                    } else {
                                        ddl.if (%min_or_minnonstick_op) {
                                            ddl.compute([%sfp_lrf_src00, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMIN", unit="sfp"}
                                        } else {
                                            ddl.if (%prod_nonstick_op) {
                                                ddl.compute([%sfp_lrf_src00, %sfp_psum_src02, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                                            } else {
                                                ddl.compute([%sfp_lrf_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                                                ddl.if (%exx2_op) {
                                                    ddl.compute([%sfp_lrf_x2_dst00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                ddl.if (%psum_end) {
                    ddl.if(%cond_not_first_loopreduce_dims) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.if (%absmax_or_absmaxnonstick_op) {
                                // chunk absmax in SFP: absmax(lrf, lxlu) => lrf
                                ddl.compute([%sfp_out_lrf, %sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="FABSMAX", unit="sfp"}
                            } else {
                                ddl.if (%max_or_maxnonstick_op) {
                                    // chunk max in SFP: max(lrf, lxlu) => lrf
                                    ddl.compute([%sfp_out_lrf, %sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="FMAX", unit="sfp"}
                                } else {
                                    ddl.if (%min_or_minnonstick_op) {
                                        // chunk min in SFP: min(lrf, lxlu) => lrf
                                        ddl.compute([%sfp_out_lrf, %sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="FMIN", unit="sfp"}
                                    } else {
                                        ddl.if (%prod_nonstick_op) {
                                            // chunk prod in SFP: lrf * lxlu => lrf
                                            ddl.compute([%sfp_out_lrf, %sfp_lrf_src00, %zero_const], [%sfp_lrf_dst00]) {computetype="FMA16", unit="sfp"}
                                        } else {  // sum, mean, exx2, exx2 zeromean
                                            // chunk accumulation in SFP: lrf + lxlu => lrf
                                            ddl.compute([%sfp_out_lrf, %one_const, %sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="FMA16", unit="sfp"}
                                        }
                                    }
                                }
                            }
                        }
                        ddl.if (%exx2_op) {
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                // chunk accumulation in SFP: lrf + Rx => lrf
                                ddl.compute([%sfp_out_lrf_x2, %one_const, %sfp_lrf_x2_src00], [%sfp_lrf_x2_dst00]) {computetype="FMA16", unit="sfp"}
                            }
                        }
                    }

                    // Exx2: one more compute:
                    %dst_div = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_div"}
                    %dst_x2_div = ddl.unit(%output_tensor_x2) {unit="sfp", data_connect="sfp_x2_div"}

                    %dst_div_send = ddl.unit(%output_tensor) {unit="lxsu", data_connect="sfp_div"}
                    %dst_x2_div_send = ddl.unit(%output_tensor_x2) {unit="lxsu", data_connect="sfp_x2_div"}

                    // SFP->LX transfer
                    %dst_out_sfplx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                    %dst_out_x2_sfplx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output", stick_replicated_dim_offset_elements=8}

                    %is_exx2_exx2_zeromean_op = ddl.condition_or(%exx2_op, %exx2_zeromean_op)
                    ddl.if (%is_exx2_exx2_zeromean_op) {
                        ddl.if(%cond_last_loopreduce_dims) {
                            // exx2 last transfer is a compute
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.if (%exx2_op) {
                                    ddl.compute([%sfp_lrf_dst00, %dst_exx2div_sfp], [%dst_div_send]) {computetype="FMUL", unit="sfp"}
                                } else {
                                    ddl.compute([%sfp_zero_lrf, %one_const, %zero_const], [%dst_div_send]) {computetype="FMA16", unit="sfp"}
                                }
                            }
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.if (%exx2_op) {
                                    ddl.compute([%sfp_lrf_x2_dst00, %dst_exx2div_sfp], [%dst_x2_div_send]) {computetype="FMUL", unit="sfp"}
                                } else {
                                    ddl.compute([%sfp_lrf_dst00, %dst_exx2div_sfp], [%dst_x2_div_send]) {computetype="FMUL", unit="sfp"}
                                }
                            }
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.data_transfer(%dst_div, [%dst_out_sfplx]) {}
                            }
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.data_transfer(%dst_x2_div, [%dst_out_x2_sfplx]) {limit_num_elements_stick_replicated_dim=8}
                            }
                        } else {
                            ddl.if (%exx2_op) {
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.compute([%sfp_lrf_dst00, %one_const, %zero_const], [%dst_div_send]) {computetype="FMA16", unit="sfp"}
                                }
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.compute([%sfp_lrf_x2_dst00, %one_const, %zero_const], [%dst_x2_div_send]) {computetype="FMA16", unit="sfp"}
                                }
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.data_transfer(%dst_div, [%dst_out_sfplx]) {}
                                }
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.data_transfer(%dst_x2_div, [%dst_out_x2_sfplx]) {limit_num_elements_stick_replicated_dim=8}
                                }
                            } else {
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.compute([%sfp_lrf_dst00, %one_const, %zero_const], [%dst_div_send]) {computetype="FMA16", unit="sfp"}
                                }
                                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                    ddl.data_transfer(%dst_div, [%dst_out_sfplx]) {}
                                }
                            }
                        }
                    } else {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.data_transfer(%sfp_lrf_dst00, [%dst_out_sfplx]) {}
                        }
                    }
                }
            } // above reduce dim
        } // D/B
    } // dataflow
} // module
