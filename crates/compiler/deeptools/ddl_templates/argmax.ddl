//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    // Dimension
    %outer_dim:5 = ddl.dimension{} : index, index, index, index,index // X, Y, I, J, MB dimension
    %reduce_dim:3 = ddl.dimension{} : index, index, index // OUT dimension

    // Layout
    %slice_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}
    %stick_layout_nonstick = ddl.layout(%outer_dim#0) {is_order_fixed=true}

    %global_layout_input = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {is_order_fixed=false}
    %global_layout_output = ddl.layout (%outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {is_order_fixed=false}

    // DataType 
    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
    %type_fp32 = ddl.type {data_type="IEEE_FP32"}
    %type_bool = ddl.type {data_type="BOOL", bit_width=8}

    // Tensor
    %input_tensor = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_input, [%type_fp16, %type_fp32]) : index
    %output_tensor = ddl.tensor(%slice_layout_nonstick, %stick_layout_nonstick, %global_layout_output,[ %type_fp16, %type_fp32]) : index

    // Internal Tensors 
    %current_max = ddl.internal_tensor(%output_tensor, [%type_fp16, %type_fp32]) : index
    %counter = ddl.internal_tensor(%output_tensor, [%type_fp16, %type_fp32]) : index
    %state_reg = ddl.internal_tensor(%output_tensor, [%type_bool]) : index 
    
    // Op
    %argmax_op = ddl.operation_bind([%type_fp16, %type_fp32], [%input_tensor, %output_tensor], [%output_tensor], [%counter, %current_max, %state_reg]) {opFuncName="sumnonstick", required=false}

    // Constant 
    %zero_const = ddl.operand_constant {name="0.0"}
    %one_const = ddl.operand_constant {name="1.0"}
    %tmp_const = ddl.operand_constant {name="1.0"}
    %zero_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="zero"}
    %counter_const_reg = ddl.define_constant(%type_fp16) {value=[0], name="count"}
    %one_const_reg = ddl.define_constant(%type_fp16) {value=[0x3E00], name="one"}
    %negInf_const_reg = ddl.define_constant(%type_fp16) {value=[0xFFFE], name="negInf"}
    %negOne_const_reg = ddl.define_constant(%type_fp16) {value=[0xBE00], name="negOne"}

    // Allocation
    %input_lx_allocation = ddl.get_external_data_transfer_allocation (%input_tensor) {memory="lx", data_connect="l3_lx_input"} 
    %output_lx_allocation = ddl.get_external_data_transfer_allocation (%output_tensor) {memory="lx", data_connect="lxsu_output"}

    // Constraints
    ddl.constraint() {min_num_cores = 1}

    // Dataflow
    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage{property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %above_interleave = ddl.datastage {strategy="maximize", allow_epilogue=true}
        %below_interleave = ddl.datastage {strategy="minimize"}

        ddl.datastage_constraint(%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {values=["1", "2", "4"]}

        // Constants
        %negInf_sfp_allocation = ddl.allocate(%negInf_const_reg) {memory="sfplrf"} 
        %src_sfp_negInf = ddl.unit(%negInf_const_reg) {unit="constant", data_connect="negInf_connect"} 
        %dst_sfp_negInf = ddl.unit(%negInf_const_reg, %negInf_sfp_allocation) {unit="sfp", data_connect="negInf_sfp_lrf"}
        ddl.data_transfer(%src_sfp_negInf, [%dst_sfp_negInf]) {}

	%negOne_sfp_allocation = ddl.allocate(%negOne_const_reg) {memory="sfplrf"}
        %src_sfp_negOne = ddl.unit(%negOne_const_reg) {unit="constant", data_connect="negOne_connect"}
        %dst_sfp_negOne = ddl.unit(%negOne_const_reg, %negOne_sfp_allocation) {unit="sfp", data_connect="negOne_sfp_lrf"}
        ddl.data_transfer(%src_sfp_negOne, [%dst_sfp_negOne]) {}

        %zero_sfp_allocation = ddl.allocate(%zero_const_reg) {memory="sfplrf"}
        %src_sfp_zero = ddl.unit(%zero_const_reg) {unit="constant", data_connect="zero_connect"}
        %dst_sfp_zero = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect="zero_sfp_lrf"}
        ddl.data_transfer(%src_sfp_zero, [%dst_sfp_zero]) {}

	%one_sfp_allocation = ddl.allocate(%one_const_reg) {memory="sfplrf"}
        %src_sfp_one = ddl.unit(%one_const_reg) {unit="constant", data_connect="one_connect"}
        %dst_sfp_one = ddl.unit(%one_const_reg, %one_sfp_allocation) {unit="sfp", data_connect="one_sfp_lrf"}
        ddl.data_transfer(%src_sfp_one, [%dst_sfp_one]) {}
       
	ddl.loop (%d_datastage, %b_datastage, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2){} {
            ddl.loop (%b_datastage, %above_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2,%outer_dim#3, %outer_dim#4){} {
		%sfp_val_lrf_allocation = ddl.allocate(%current_max) {memory="sfplrf"}
                %sfp_idx_lrf_allocation = ddl.allocate(%output_tensor) {memory="sfplrf"}
                %sfp_count_lrf_allocation = ddl.allocate(%counter) {memory="sfplrf"}
		%sfp_state_allocation = ddl.allocate(%state_reg) {memory="sfpstate"}

		%sfp_count_lrf = ddl.unit(%counter, %sfp_count_lrf_allocation) {unit="sfp", data_connect="count_sfp_lrf"}
		%sfp_idx_lrf = ddl.unit(%output_tensor, %sfp_idx_lrf_allocation) {unit="sfp", data_connect="idx_sfp_lrf"}
		%sfp_val_lrf = ddl.unit(%current_max, %sfp_val_lrf_allocation) {unit="sfp", data_connect="val_sfp_lrf"}
		%sfp_state = ddl.unit(%state_reg, %sfp_state_allocation) {unit="sfp", data_connect="sfp_state_reg"}
		%sfp_negInf_lrf = ddl.unit(%negInf_const_reg, %negInf_sfp_allocation) {unit="sfp", data_connect="negInf_sfp_lrf"}
                %sfp_negOne_lrf = ddl.unit(%negOne_const_reg, %negOne_sfp_allocation) {unit="sfp", data_connect="negOne_sfp_lrf"}
                %sfp_zero_lrf = ddl.unit(%zero_const_reg, %zero_sfp_allocation) {unit="sfp", data_connect="zero_sfp_lrf"}
                %sfp_one_lrf = ddl.unit(%one_const_reg, %one_sfp_allocation) {unit="sfp", data_connect="one_sfp_lrf"}
		
		// Initialize current max + index
		ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {} {
		    ddl.compute([%sfp_negInf_lrf, %one_const, %zero_const], [%sfp_val_lrf]) {computetype="MACC", unit="sfp"}
		    ddl.compute([%sfp_negOne_lrf, %one_const, %zero_const], [%sfp_idx_lrf]) {computetype="MACC", unit="sfp"}
		}
		// Initalize counter 
		ddl.compute([%sfp_zero_lrf, %one_const, %zero_const], [%sfp_count_lrf]) {computetype="MACC", unit="sfp"}
               
		ddl.loop (%b_datastage, %below_interleave, %reduce_dim#0, %reduce_dim#1, %reduce_dim#2) {} {
			ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {} {
				%src_inp_lx = ddl.unit(%input_tensor, %input_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
                		%dst_inp_lx = ddl.unit(%input_tensor) {unit="sfp", data_connect="sfp_lx_input"}
                		%lxlu_inp = ddl.unit(%input_tensor) {unit="lxlu", data_connect="sfp_lx_input"}

                		ddl.data_transfer(%src_inp_lx, [%dst_inp_lx]) {}
			
				ddl.compute([%lxlu_inp, %sfp_val_lrf], [%sfp_state, %sfp_val_lrf]) {computetype="FMAX", unit="sfp"}	
				ddl.compute([%sfp_state,  %sfp_count_lrf,  %sfp_idx_lrf], [%sfp_idx_lrf]) {computetype="SELECT", unit="sfp"}

				//ddl.opaque(%output_tensor, %sfp_val_lrf_allocation, %sfp_count_lrf_allocation, %sfp_idx_lrf_allocation)	     
                                //{unit="sfp", op="ARGMAX", input_output_registers=["val", "counter", "idx"], internal_registers=[],
				// max_unroll_factor=1, params={"in0"="lxlu"},
				// input_data_connects=["val_sfp_lrf", "count_sfp_lrf", "idx_sfp_lrf","sfp_lx_input"],
				// output_data_connects=["val_sfp_lrf", "idx_sfp_lrf"]}
			}
			ddl.compute([%sfp_count_lrf, %sfp_one_lrf, %sfp_one_lrf], [%sfp_count_lrf]) {computetype="MACC", unit="sfp"}
               } 
                // SFP->LX transfer
                %dst_out_sfp_lx = ddl.unit(%output_tensor, %output_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                ddl.loop (%above_interleave, %below_interleave, %outer_dim#0, %outer_dim#1, %outer_dim#2, %outer_dim#3, %outer_dim#4) {} {
                    ddl.data_transfer(%sfp_idx_lrf, [%dst_out_sfp_lx]) {}
		}
            }
	} // D/B
    } // dataflow
} // module
