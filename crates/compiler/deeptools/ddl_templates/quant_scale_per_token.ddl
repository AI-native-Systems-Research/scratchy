//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

// quantscalepertoken

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
    %input_tensor = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_input, [%type_fp16]) : index
    %output_tensor = ddl.tensor(%slice_layout_stick, %stick_layout_stick, %global_layout_output, [%type_fp16]) : index


    %output_tensor_intermediate = ddl.internal_tensor(%output_tensor, [%type_fp16]) : index

    %output_tensor_reduce = ddl.internal_tensor(%output_tensor, [%type_fp16]) : index


    // Op
    %psum_op = ddl.operation_bind([%type_fp16], [%output_tensor], [%output_tensor]) {opFuncName="genericpartialreduction", required=false}
    %quantscalepertoken_op = ddl.operation_bind([%type_fp16], [%input_tensor] ,[%output_tensor], [%output_tensor_reduce, %output_tensor_intermediate]) {opFuncName="quantscalepertoken", required=false}
    %quantscalepertokenfp8_op = ddl.operation_bind([%type_fp16], [%input_tensor] ,[%output_tensor], [%output_tensor_reduce]) {opFuncName="quantscalepertokenfp8", required=false}

    // Constant 
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}
    %zero_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %nfwd0_const = ddl.operand_constant {name="nfwd0"}
    %nfwd2_const = ddl.operand_constant {name="nfwd2"}
    %minus1 = ddl.define_constant(%type_fp16) {value=[0xBE00], name = "minus1"}
    %tff = ddl.define_constant(%type_fp16) {value = [0x4DFC], name = "tff"} // 255?

    %mul_const = ddl.get_external_constant(%type_fp16) {name = "mulConst", num_elements=1}
    %clip_min_const = ddl.get_external_constant(%type_fp16){name="clipMin", num_elements=1}
    %clip_max_const = ddl.get_external_constant(%type_fp16){name="clipMax", num_elements=1}

    %ffff_const = ddl.define_constant(%type_fp16) {value=[0xFFFF], name="ffff"}
    %plus1_const = ddl.define_constant(%type_fp16){value=[0x3E00], name="plus1"}

    // Allocation
    %input_lx_allocation = ddl.get_external_data_transfer_allocation (%input_tensor) {memory="lx", data_connect="l3_lx_input"} 
    %output_lx_allocation = ddl.get_external_data_transfer_allocation (%output_tensor) { memory="lx", data_connect="lxsu_output"}

    // Constraints
    ddl.constraint(%quantscalepertoken_op, %quantscalepertokenfp8_op) {min_num_valid = 1, max_num_valid = 1}
    ddl.constraint() {min_num_cores = 1}

    // Dataflow
    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage{property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %above_interleave = ddl.datastage {strategy="maximize", allow_epilogue=true}
        %below_interleave = ddl.datastage {strategy="minimize"}
        ddl.datastage_constraint(%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {values=["1", "2", "4"]}

        // Constant:
        %zero_pe_allocation = ddl.allocate(%zero_const_reg) {memory="pelrf"} 
        %src_pe_zero = ddl.unit(%zero_const_reg) {unit="constant", data_connect= "zero_connect"} 
        %pe_zero_lrf = ddl.unit(%zero_const_reg, %zero_pe_allocation) {unit="pe", data_connect= "zero_pe_lrf"}
        ddl.data_transfer(%src_pe_zero, [%pe_zero_lrf]) {}

        %mul_const_sfp_allocation = ddl.allocate(%mul_const) {memory="sfplrf"}
        %clip_min_sfp_allocation = ddl.allocate(%clip_min_const) {memory="sfplrf"}
        %clip_max_sfp_allocation = ddl.allocate(%clip_max_const) {memory="sfplrf"}

        ddl.if(%quantscalepertokenfp8_op) {
            %src_mul_const = ddl.unit(%mul_const) {unit="constant", data_connect= "mul_const_connect"} 
            %src_clip_min = ddl.unit(%clip_min_const) {unit="constant", data_connect= "clip_min_const_connect"} 
            %src_clip_max = ddl.unit(%clip_max_const) {unit="constant", data_connect= "clip_max_const_connect"} 
            %dst_mul_const_sfp = ddl.unit(%mul_const, %mul_const_sfp_allocation) {unit="sfp", data_connect= "mul_const_sfp_lrf"}
            ddl.data_transfer(%src_mul_const, [%dst_mul_const_sfp]) {}
            %dst_clip_min_sfp = ddl.unit(%clip_min_const, %clip_min_sfp_allocation) {unit="sfp", data_connect= "clip_min_sfp_lrf"}
            ddl.data_transfer(%src_clip_min, [%dst_clip_min_sfp]) {}
            %dst_clip_max_sfp = ddl.unit(%clip_max_const, %clip_max_sfp_allocation) {unit="sfp", data_connect= "clip_max_sfp_lrf"}
            ddl.data_transfer(%src_clip_max, [%dst_clip_max_sfp]) {}
        }


        %tff_sfp_allocation = ddl.allocate(%tff) {memory="sfplrf"} 
        %zero_sfp_allocation = ddl.allocate(%zero_const_reg) {memory="sfplrf"} 
        %ffff_sfp_allocation = ddl.allocate(%ffff_const) {memory="sfplrf"} 
        %plus1_sfp_allocation = ddl.allocate(%plus1_const) {memory="sfplrf"}
        %minus1_sfp_allocation = ddl.allocate(%minus1) {memory="sfplrf"}

        ddl.if(%quantscalepertoken_op){
            %src_tff = ddl.unit(%tff) {unit="constant", data_connect= "tff_const_connect"} 
            %dst_tff_sfp = ddl.unit(%tff, %tff_sfp_allocation) {unit="sfp", data_connect= "tff_sfp_lrf"}
            ddl.data_transfer(%src_tff, [%dst_tff_sfp]) {}

            %src_ffff_const = ddl.unit(%ffff_const) {unit="constant", data_connect= "ffff_const_connect"} 
            %dst_ffff_sfp = ddl.unit(%ffff_const, %ffff_sfp_allocation) {unit="sfp", data_connect= "ffff_sfp_lrf"}
            ddl.data_transfer(%src_ffff_const, [%dst_ffff_sfp]) {}

            %src_plus1_const = ddl.unit(%plus1_const) {unit="constant", data_connect= "plus1_const_connect"} 
            %dst_plus1_sfp = ddl.unit(%plus1_const, %plus1_sfp_allocation) {unit="sfp", data_connect= "plus1_sfp_lrf"}
            ddl.data_transfer(%src_plus1_const, [%dst_plus1_sfp]) {}

            %src_pe_minus1 = ddl.unit(%minus1) {unit="constant", data_connect= "minus1_connect"} 
            %dst_sfp_minus1 = ddl.unit(%minus1, %minus1_sfp_allocation) {unit="sfp", data_connect= "minus1_sfp_lrf"}
            ddl.data_transfer(%src_pe_minus1, [%dst_sfp_minus1]) {}

            %src_zero_const_opaque = ddl.unit(%zero_const_reg) {unit="constant", data_connect= "zero_const_opaque_connect"} 
            %dst_zero_sfp = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect= "zero_sfp_lrf"}
            ddl.data_transfer(%src_zero_const_opaque, [%dst_zero_sfp]) {}
        }

     
        ddl.loop (%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3){label="chunk_loop"} {
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
                %sfp_lrf_allocation_reduce = ddl.allocate(%output_tensor_reduce) {memory="sfplrf"}
                %sfp_out_lrf = ddl.unit(%output_tensor_reduce, %sfp_lrf_allocation_reduce) {unit="sfp", data_connect="lxsfp_output"}
                ddl.if(%psum_end) {
                    ddl.if(%cond_not_first_loopreduce_dims) {
                        %src_out_lxpesfp = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxlu", data_connect="lxsu_output"}
                        ddl.data_transfer(%src_out_lxpesfp, [%sfp_out_lrf]) {}
                    }
                }

                %pe_lrf_allocation = ddl.allocate(%output_tensor) {memory="pelrf"}
                %pe_reduce_src02_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                %pe_reduce_dst00_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}

                // explicitly initialize to 0
                // LRF x 1 + 0 -> LRF
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute([%pe_zero_lrf, %one_const, %zero_const], [%pe_reduce_dst00_lrf]) {computetype="FMA16", unit="pe"}
                }
           
                // Inner-loop of reduction
                ddl.loop (%b_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2, %reduce_dim#3) {} {
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        // LX->PE 
                        %src_inp_lxpe = ddl.unit(%input_tensor, %input_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
                        %dst_inp_lxpe = ddl.unit(%input_tensor) {unit="pe", data_connect="pe_lx_input"}

                        %pe_lxlu_input = ddl.unit(%input_tensor) {unit="lxlu", data_connect="pe_lx_input"}

                        // chunk absmax in PE: absmax(lrf, lxlu) => lrf
                        ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}
                        ddl.compute([%pe_lxlu_input, %pe_reduce_src02_lrf], [%pe_reduce_dst00_lrf]) {computetype="FABSMAX", unit="pe"}
   
                    } // end inner reduction
                }
                // PE nfwd reduction
                %pe_nfwd_src_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                %pe_nfwd_dst1_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                %pe_nfwd_dst2_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="inter_stick_output"}
                %pe_nfwd_dst3_lrf = ddl.unit(%output_tensor, %pe_lrf_allocation) {unit="pe", data_connect="pe_reduce_output"}

                // explicitly perform transfer.

                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute([%nfwd0_const, %pe_nfwd_src_lrf], [%pe_nfwd_dst1_lrf]) {computetype="FABSMAX", unit="pe"}
                }
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute([%pe_nfwd_dst1_lrf, %nfwd2_const], [%pe_nfwd_dst2_lrf]) {computetype="FABSMAX", unit="pe"}
                }
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute([%nfwd0_const, %nfwd2_const], [%pe_nfwd_dst3_lrf]) {computetype="FABSMAX", unit="pe"}
                }

                // PE->SFP
                %output_sfp_lrf_allocation = ddl.allocate(%output_tensor) {memory="sfplrf"}

                %dst_out_pesfp = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_splat_input"}
                %src_out_pesfp = ddl.unit(%output_tensor) {unit="pe", data_connect="sfp_splat_input"}

                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.data_transfer(%pe_nfwd_dst3_lrf, [%dst_out_pesfp]) {}
                }
                %dst_out_sfpsplat = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_reduce_input"}
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute([%src_out_pesfp], [%dst_out_sfpsplat]) {computetype="SPLAT", unit="sfp"}
                }

                // SFP Reduce
                %sfp_lrf_src00 = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_reduce_input"}
                %sfp_lrf_dst00 = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_reduce_output"}

                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                    ddl.compute ([%sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="REDUCE", unit="sfp", mode=10}
                }
                ddl.if (%psum_op) {
                    // Do p_sum
                    %sfp_psum_src00_sfpring = ddl.unit(%output_tensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
                    %sfp_psum_dst00_sfpring = ddl.unit(%output_tensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
                    ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                        ddl.if (%psum_start) {
                            // psum: sfplrf + 0 -> sfpring
                            ddl.compute([%sfp_lrf_src00, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                        } else {
                            ddl.if (%psum_end) {
                                // psum: op(sfplrf, sfpring) -> lxsu
                                ddl.compute([%sfp_lrf_src00, %sfp_psum_src00_sfpring], [%sfp_lrf_dst00]) {computetype="FABSMAX", unit="sfp"}
                            } else { // middle cores
                                // psum: op(sfplrf, sfpring) -> sfpring
                                ddl.compute([%sfp_lrf_src00, %sfp_psum_src00_sfpring], [%sfp_psum_dst00_sfpring]) {computetype="FABSMAX", unit="sfp"}
                            }
                        }
                    }
                }

                ddl.if (%psum_end) {
                    ddl.if(%cond_not_first_loopreduce_dims) {
                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            // chunk absmax in SFP: absmax(lrf, lxlu) => lrf
                            ddl.compute([%sfp_out_lrf, %sfp_lrf_src00], [%sfp_lrf_dst00]) {computetype="FABSMAX", unit="sfp"}
                        }
                    }

                    %dst_out_sfplx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                    ddl.if(%cond_last_loopreduce_dims) {
                        // 2*Max
                        %dst_out_max2 = ddl.unit(%output_tensor, %output_sfp_lrf_allocation) {unit="sfp", data_connect="sfp_max2_input"}

                        ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                            ddl.if(%quantscalepertoken_op)  {
                                ddl.compute([%sfp_lrf_dst00, %one_const, %sfp_lrf_dst00], [%dst_out_max2]) {computetype="FMA16", unit="sfp"}
                            } else {
                                %dst_mul_const_sfp = ddl.unit(%mul_const, %mul_const_sfp_allocation) {unit="sfp", data_connect= "mul_const_sfp_lrf"}
                                ddl.compute([%sfp_lrf_dst00, %dst_mul_const_sfp, %zero_const], [%dst_out_max2]) {computetype="FMA16", unit="sfp"}
                            }
                        }

                        //Real Div ops
                        ddl.if(%quantscalepertoken_op) {
                            ddl.opaque(%output_tensor_intermediate, %zero_sfp_allocation, %ffff_sfp_allocation, %minus1_sfp_allocation, %plus1_sfp_allocation, %tff_sfp_allocation, %output_sfp_lrf_allocation)
                            {unit="sfp", op="REALDIV", input_output_registers=["c1", "c2", "c3", "c4", "in1_unroll", "in0_unroll"], internal_registers=["p0_unroll", "t0_unroll", "t2_unroll", "t4_unroll", "t6_unroll"],
                            max_unroll_factor=2, params={"out0"="result"},
                            input_data_connects=["zero_sfp_lrf", "ffff_sfp_lrf", "minus1_sfp_lrf", "plus1_sfp_lrf", "tff_sfp_lrf", "sfp_max2_input"],
                            output_data_connects=["sfp_output"]}
                        } else {
                            %src_clip_max_sfp = ddl.unit(%clip_max_const, %clip_max_sfp_allocation) {unit="sfp", data_connect= "clip_max_sfp_lrf"}
                            %src_clip_min_sfp = ddl.unit(%clip_min_const, %clip_min_sfp_allocation) {unit="sfp", data_connect= "clip_min_sfp_lrf"}

                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.compute([%dst_out_max2, %src_clip_min_sfp], [%dst_out_max2]) {computetype="FMAX", unit="sfp"}
                            }
                            %dst_output = ddl.unit(%output_tensor) {unit="lxsu", data_connect="sfp_output"}
                            ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4) {} {
                                ddl.compute([%dst_out_max2, %src_clip_max_sfp], [%dst_output]) {computetype="FMIN", unit="sfp"}
                            }
                        }
                        // sfp-lx
                        %src_out_sfplx = ddl.unit(%output_tensor) {unit="sfp", data_connect="sfp_output"}
                        ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
                    } else {
                        // sfp-lx
                        ddl.data_transfer(%sfp_lrf_dst00, [%dst_out_sfplx]) {}
                    }
                }
            } // above reduce dim
        } // D/B
    } // dataflow
} // module
