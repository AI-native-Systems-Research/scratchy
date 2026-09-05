//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    // Dimension
    %outer_dim:5 = ddl.dimension{} : index, index, index, index,index // X, Y, I, J, MB dimension
    %reduce_dim:3 = ddl.dimension{} : index, index, index // OUT dimension
    %k = ddl.dimension{} : index // K dimension 

    // Layout
    %slice_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}
    %stick_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}

    %global_layout_input = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {is_order_fixed=false}
    %global_layout_output = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %k) {is_order_fixed=false}
    
    // DataType 
    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
    %type_fp32 = ddl.type {data_type="IEEE_FP32"}
    %type_bool = ddl.type {data_type="BOOL", bit_width=8}
    
    // Tensor
    %inp1_tensor = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_input, [%type_fp16, %type_fp32]) : index
    %topk_output_tensor = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_output, [%type_fp16, %type_fp32]) : index
    %mask_output_tensor = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_input, [%type_fp16, %type_fp32]) : index

    // Internal Tensors 
    %current_max = ddl.internal_tensor(%topk_output_tensor, [%type_fp16, %type_fp32]) : index
    %current_idx = ddl.internal_tensor(%topk_output_tensor, [%type_fp16, %type_fp32]) : index
    %state_reg = ddl.internal_tensor(%topk_output_tensor, [%type_bool]) : index
    %tmp_val = ddl.internal_tensor(%topk_output_tensor, [%type_fp16, %type_fp32]) : index 
    %tmp_idx = ddl.internal_tensor(%topk_output_tensor, [%type_fp16, %type_fp32]) : index 
 
    
    // Op
    %topk_idx_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1_tensor], [%topk_output_tensor], [%current_max, %current_idx, %state_reg, %tmp_val, %tmp_idx]) {opFuncName="topkindex", required=false}
    %topk_val_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1_tensor], [%topk_output_tensor], [%current_max, %current_idx, %state_reg, %tmp_val, %tmp_idx]) {opFuncName="topkvalue", required=false}
    %mask_idx_op = ddl.operation_bind([%type_fp16, %type_fp32], [%inp1_tensor, %topk_output_tensor], [%mask_output_tensor], [%state_reg]) {opFuncName="maskbyindex", required=false}
    %psum_op = ddl.operation_bind([], [%mask_output_tensor], [%mask_output_tensor]) {opFuncName="genericpartialreduction", required=false}
    
    // Constant 
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}
    %tmp_const = ddl.operand_constant {name="1.0"}
    %counter_const = ddl.define_constant(%type_fp16) {value=[0], name="counter"}
    %tmp_count_const = ddl.define_constant(%type_fp16) {value=[0], name="tmp_counter"}
    %zero_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %one_const_reg = ddl.define_constant(%type_fp16) {value=[0x3E00], name="one"}
    %negInf_const_reg = ddl.define_constant(%type_fp16){value=[0xFFFE], name="negInf"}
    %negOne_const_reg = ddl.define_constant(%type_fp16){value=[0xBE00], name="negOne"}


    // Allocation
    %inp1_lx_allocation = ddl.get_external_data_transfer_allocation (%inp1_tensor) {memory="lx", data_connect="l3_lx_input1"} 
    %topk_output_lx_allocation = ddl.get_external_data_transfer_allocation (%topk_output_tensor) {memory="lx", data_connect="lx_topk_output"} 
    %mask_output_lx_allocation = ddl.get_external_data_transfer_allocation (%mask_output_tensor) {memory="lx", data_connect="lx_mask_output"} 
    
    // Constraints
    ddl.constraint(%topk_idx_op, %topk_val_op, %mask_idx_op) {min_num_valid = 1, max_num_valid = 1}
    ddl.constraint() {min_num_cores = 1}
        
   
    // Dataflow
    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage{property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %above_interleave = ddl.datastage {strategy="maximize", allow_epilogue=true}
        %below_interleave = ddl.datastage {strategy="minimize"}
        ddl.datastage_constraint(%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {values=["1", "2", "4"]}
        
        // core-to-core communication for k worksplit 
        %start_core, %end_core, %next_core, %prev_core = ddl.core_to_core_communication(%k)
        %not_end_core = ddl.condition_not(%end_core)
        %topk_op = ddl.condition_or(%topk_idx_op, %topk_val_op)
        %is_end_mask = ddl.condition_and(%end_core, %mask_idx_op) 
   	

        	
        // Constants
	// Statically save -inf, -1 and 1 in SFP_reg[8], SFP_reg[9], and SFP_reg[10]
        %negInf_sfp_allocation = ddl.allocate(%negInf_const_reg) {memory="sfplrf"} 
        %src_sfp_negInf = ddl.unit(%negInf_const_reg) {unit="constant", data_connect="negInf_connect"} 
        %dst_sfp_negInf = ddl.unit(%negInf_const_reg, %negInf_sfp_allocation) {unit="sfp", data_connect="negInf_sfp_lrf"}
        ddl.data_transfer(%src_sfp_negInf, [%dst_sfp_negInf]) {}

	%negOne_sfp_allocation = ddl.allocate(%negOne_const_reg) {memory="sfplrf"}
        %src_sfp_negOne = ddl.unit(%negOne_const_reg) {unit="constant", data_connect="negOne_connect"}
        %dst_sfp_negOne = ddl.unit(%negOne_const_reg, %negOne_sfp_allocation) {unit="sfp", data_connect="negOne_sfp_lrf"}
        ddl.data_transfer(%src_sfp_negOne, [%dst_sfp_negOne]) {}

	%one_sfp_allocation = ddl.allocate(%one_const_reg) {memory="sfplrf"}
        %src_sfp_one = ddl.unit(%one_const_reg) {unit="constant", data_connect="one_connect"}
        %dst_sfp_one = ddl.unit(%one_const_reg, %one_sfp_allocation) {unit="sfp", data_connect="one_sfp_lrf"}
        ddl.data_transfer(%src_sfp_one, [%dst_sfp_one]) {}

        %sfp_negInf_lrf = ddl.unit(%negInf_const_reg, %negInf_sfp_allocation) {unit="sfp", data_connect="negInf_sfp_lrf"}
        %sfp_negOne_lrf = ddl.unit(%negOne_const_reg, %negOne_sfp_allocation) {unit="sfp", data_connect="negOne_sfp_lrf"}
        %sfp_one_lrf = ddl.unit(%one_const_reg, %one_sfp_allocation) {unit="sfp", data_connect="one_sfp_lrf"}

        ddl.loop (%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2){label="chunk_loop"} {
                ddl.loop (%b_datastage, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4){label="bottom_loop"} { 
                        // Conditions to only init on first loop
                        %first_loop_dim0 = ddl.condition(%reduce_dim#0){loop_label="chunk_loop", condition="eq", value_expr="first"}
                        %first_loop_dim1 = ddl.condition(%reduce_dim#1){loop_label="chunk_loop", condition="eq", value_expr="first"}
                        %first_loop_dim2 = ddl.condition(%reduce_dim#2){loop_label="chunk_loop", condition="eq", value_expr="first"}
                        %is_first_loop = ddl.condition_and(%first_loop_dim0, %first_loop_dim1, %first_loop_dim2)
                        
                        %sfp_val_lrf_allocation = ddl.allocate(%current_max) {memory="sfplrf"}
                        %sfp_idx_lrf_allocation = ddl.allocate(%current_idx) {memory="sfplrf"}
                        %counter_sfp_allocation = ddl.allocate(%counter_const) {memory="sfplrf"} 
              
                        %sfp_idx_lrf = ddl.unit(%current_idx, %sfp_idx_lrf_allocation) {unit="sfp", data_connect="idx_sfp_lrf"}
                        %sfp_val_lrf = ddl.unit(%current_max, %sfp_val_lrf_allocation) {unit="sfp", data_connect="val_sfp_lrf"}
                        %sfp_count_lrf = ddl.unit(%counter_const, %counter_sfp_allocation) {unit="sfp", data_connect="count_sfp_lrf"}

			ddl.if (%is_first_loop) {
                                // Initialize running values, indices, and counter
                                ddl.if (%topk_op) {
                                        ddl.loop (%b_datastage, %below_interleave, %k) {} {
                                                ddl.compute([%sfp_negOne_lrf, %one_const, %zero_const], [%sfp_idx_lrf]) {computetype="MACC", unit="sfp"}
                                                ddl.compute([%sfp_negInf_lrf, %one_const, %zero_const], [%sfp_val_lrf]) {computetype="MACC", unit="sfp"}
                                        }
                                }
                                %src_sfp_count = ddl.unit(%counter_const) {unit="constant", data_connect="count_connect"}
                                ddl.data_transfer(%src_sfp_count, [%sfp_count_lrf]) {} 
			}
                        ddl.loop (%b_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {} {
                                %sfp_current_lrf_allocation = ddl.allocate(%inp1_tensor) {memory="sfplrf"}
                                %sfp_cur_lrf = ddl.unit(%inp1_tensor, %sfp_current_lrf_allocation) {unit="sfp", data_connect="sfp_lx_input"}    
                                   
                                %sfp_mask_index_lrf_allocation = ddl.allocate(%topk_output_tensor) {memory="sfplrf"}  
                                %sfp_mask_index_lrf = ddl.unit(%topk_output_tensor, %sfp_mask_index_lrf_allocation) {unit="sfp", data_connect="mask_idx_sfp_lrf"}

                                %tmp_counter_sfp_allocation = ddl.allocate(%tmp_count_const) {memory="sfplrf"}
                                %sfp_tmp_count_lrf = ddl.unit(%tmp_count_const, %tmp_counter_sfp_allocation) {unit="sfp", data_connect="tmp_count_sfp_lrf"}
                                
                                
                                ddl.if(%mask_idx_op) {
                                        // maskbyindex: each core gets their own indices.                         
                                        %src_idx_lx = ddl.unit(%topk_output_tensor, %topk_output_lx_allocation) {unit="lxlu", data_connect="lx_topk_output"} 
                                        ddl.data_transfer(%src_idx_lx, [%sfp_mask_index_lrf]) {}   
                                }

                                // topk: copy of running counter
                                ddl.compute([%sfp_count_lrf, %one_const, %zero_const], [%sfp_tmp_count_lrf]) {computetype="MACC", unit="sfp"}
                        
                                %sfp_psum_src = ddl.unit(%inp1_tensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
                                %sfp_psum_dst_sfpring = ddl.unit(%inp1_tensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
                        
                                %sfp_psum_idx = ddl.unit(%topk_output_tensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
                                %sfp_psum_idx_sfpring = ddl.unit(%topk_output_tensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
                                
                                ddl.if (%start_core) {
                                        // start core gets input from lxlu
                                        %src_inp_lx = ddl.unit(%inp1_tensor, %inp1_lx_allocation) {unit="lxlu", data_connect="l3_lx_input1"}
                                        ddl.data_transfer(%src_inp_lx, [%sfp_cur_lrf]) {}   
                                } else { // middle and end cores gets input from sfpring 
                                        ddl.compute([%sfp_psum_src, %one_const, %zero_const], [%sfp_cur_lrf]) {computetype="MACC", unit="sfp"}
                                        ddl.if (%topk_op) {
                                                ddl.compute([%sfp_psum_idx, %one_const, %zero_const], [%sfp_count_lrf]) {computetype="MACC", unit="sfp"} 
                                        }
                                }

                                ddl.loop (%b_datastage, %below_interleave, %k) {label="k_loop"} {               
                                        %sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}
                                        %sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
                                                
                                        ddl.if (%topk_op) { // topk 
                                                // Allocations for temporary idx and value
                                                %sfp_tmp_val_allocation = ddl.allocate(%tmp_val) {memory="sfplrf"}
                                                %sfp_tmp_idx_allocation = ddl.allocate(%tmp_idx) {memory="sfplrf"}

                                                %sfp_tmp_val_lrf = ddl.unit(%tmp_val, %sfp_tmp_val_allocation) {unit="sfp", data_connect="tmp_val_lrf"}
                                                %sfp_tmp_idx_lrf = ddl.unit(%tmp_idx, %sfp_tmp_idx_allocation) {unit="sfp", data_connect="tmp_idx_lrf"}

                                                // Keep copy of running value and index
                                                ddl.compute([%sfp_val_lrf, %one_const, %zero_const], [%sfp_tmp_val_lrf]) {computetype="MACC", unit="sfp"}
                                                ddl.compute([%sfp_idx_lrf, %one_const, %zero_const], [%sfp_tmp_idx_lrf]) {computetype="MACC", unit="sfp"}
                                                
						ddl.compute([%sfp_val_lrf, %sfp_cur_lrf], [%sfp_state]) {computetype="GREATERTHAN", unit="sfp"}
                                                ddl.compute([%sfp_state, %sfp_val_lrf, %sfp_cur_lrf], [%sfp_val_lrf]) {computetype="SELECT", unit="sfp"}
                                                
						// SFP_SELECT tmp_val = SELECT([state_reg, tmp_val, lxlu],[tmp_val])
                                                ddl.compute([%sfp_state, %sfp_cur_lrf, %sfp_tmp_val_lrf], [%sfp_cur_lrf]) {computetype="SELECT", unit="sfp"}
                                                        
                                                // SFP_SELECT sfp_idx= SELECT([state_reg, counter, idx],[idx])
                                                ddl.compute([%sfp_state, %sfp_idx_lrf, %sfp_count_lrf], [%sfp_idx_lrf]) {computetype="SELECT", unit="sfp"}
                                                        
                                                // SFP_SELECT sfp_count = SELECT([state_reg, tmp_counter, counter],[sfp_count])
                                                ddl.compute([%sfp_state,  %sfp_count_lrf, %sfp_tmp_idx_lrf], [%sfp_count_lrf]) {computetype="SELECT", unit="sfp"}
                                        } 
                                        ddl.if (%mask_idx_op) { // mask by index
                                                ddl.compute([%sfp_mask_index_lrf, %sfp_count_lrf], [%sfp_state]) {computetype="EQUAL", unit="sfp"}
                                                ddl.compute([%sfp_state, %sfp_negInf_lrf, %sfp_cur_lrf], [%sfp_cur_lrf]) {computetype="SELECT", unit="sfp"}
                                                
                                        }    
                                                        
                                } // k loop 
                                // Share inputs and indicies across cores
                                ddl.if (%not_end_core) {
                                        ddl.compute([%sfp_cur_lrf, %one_const, %zero_const], [%sfp_psum_dst_sfpring]) {computetype="MACC", unit="sfp"}
                                        ddl.if (%topk_op) {
                                                ddl.compute([%sfp_count_lrf, %one_const, %zero_const], [%sfp_psum_idx_sfpring]) {computetype="MACC", unit="sfp"}
                                        }
                                } 
                                // Increment counter 
                                ddl.if (%topk_op) {
                                        // topk updates counter using copy in temp register
                                        ddl.compute([%sfp_tmp_count_lrf, %sfp_one_lrf, %sfp_one_lrf], [%sfp_count_lrf]) {computetype="MACC", unit="sfp"}
                                }
                                ddl.if (%mask_idx_op) {
                                        // mask by index increments running counter
                                        ddl.compute([%sfp_count_lrf, %sfp_one_lrf, %sfp_one_lrf], [%sfp_count_lrf]) {computetype="MACC", unit="sfp"}
                                }   
                                // SFP->LX mask by index
                                ddl.if (%is_end_mask) {
                                        %dst_out_sfp_lx = ddl.unit(%mask_output_tensor, %mask_output_lx_allocation) {unit="lxsu", data_connect="lx_mask_output"}
                                        ddl.data_transfer(%sfp_cur_lrf, [%dst_out_sfp_lx]) {}
                                }
                                 
                        }  // n loop
                        // SFP->LX topk
                        ddl.loop (%b_datastage, %below_interleave, %k) {} {
                                // output index to lx
                                ddl.if (%topk_idx_op) {
                                        %dst_out_sfp_lx = ddl.unit(%topk_output_tensor, %topk_output_lx_allocation) {unit="lxsu", data_connect="lx_topk_output"}
                                        ddl.data_transfer(%sfp_idx_lrf, [%dst_out_sfp_lx]) {}
                                }
                                // output value to lx
                                ddl.if (%topk_val_op) {
                                        %dst_out_sfp_lx = ddl.unit(%topk_output_tensor, %topk_output_lx_allocation) {unit="lxsu", data_connect="lx_topk_output"}
                                        ddl.data_transfer(%sfp_val_lrf, [%dst_out_sfp_lx]) {}
                                }
                        }
                } // chunk/1
	} // d/chunk
    } // dataflow
} // module
