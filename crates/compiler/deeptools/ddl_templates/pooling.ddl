//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}

//   Uses different layout for input and output tensors.
%wrd:2 = ddl.dimension {} : index, index  // Expected to be I and J 
%d:2 = ddl.dimension {} : index, index    // Expected to be mb and out
// These dimensions can be used in a window operation context
%w:2 = ddl.dimension { dim_property = "window" } : index, index
// These dimensions can be used in a stride-related context
%s:2 = ddl.dimension { dim_property = "stride" } : index, index
// These dimensions can be used in a dilation-related context
%dl:2 = ddl.dimension { dim_property = "dilation" } : index, index
// These dimensions can be used in a front-padding related context
%pf:2 = ddl.dimension { dim_property = "pad_front" } : index, index
// These dimensions can be used in a back-padding related context
%pb:2 = ddl.dimension { dim_property = "pad_back" } : index, index
// These dimensions can be used in a context that requires access to
// the valid part of a padded dimension
%pv:2 = ddl.dimension { dim_property = "pad_valid" } : index, index
%wrd0_pad = ddl.padded_dimension(primary=%wrd#0, padding=[%pf#0, %pb#0, %pv#0], window=[%w#0, %s#0, %dl#0] )
%wrd1_pad = ddl.padded_dimension(primary=%wrd#1, padding=[%pf#1, %pb#1, %pv#1], window=[%w#1, %s#1, %dl#1] )

%slice_layout = ddl.layout(%d#0, %d#1) {is_order_fixed=false}  
%stick_layout = ddl.layout(%d#0, %d#1) {is_order_fixed=false}
%global_layout_input = ddl.layout(%wrd0_pad, %wrd1_pad, %d#0, %d#1) {}
%global_layout_output = ddl.layout(%wrd#0, %wrd#1, %d#0, %d#1) {}
%type_fp16 = ddl.type { data_type="SEN169_FP16", bit_width=16 }
%inptensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_input, [%type_fp16]) : index
%outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_output, [%type_fp16]) : index
// The nmap tensor for avgpool has only I and J dimensions. The tensor is
// broadcast along other dimensions.
// Stick has NOut, scale of -2, values are already replicated along I and J
%global_layout_nmap = ddl.layout(%wrd#0, %wrd#1) {}
%nmap_tensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_nmap, [%type_fp16]) : index

%maxpool_op = ddl.operation_bind([%type_fp16], [ %inptensor ], [%outtensor]) {opFuncName="maxpoolfwd", required=false}
%avgpool_op = ddl.operation_bind([%type_fp16], [ %inptensor ], [%outtensor]) {opFuncName="avgpoolfwd", required=false}
%avgpoolnmap_op = ddl.operation_bind([%type_fp16], [ %inptensor, %nmap_tensor ], [%outtensor]) {opFuncName="avgpoolnmapfwd", required=false}

ddl.constraint(%inptensor, %outtensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor, %outtensor) {property = "stick", cmp = "equal"}

ddl.constraint(%maxpool_op, %avgpool_op, %avgpoolnmap_op) {min_num_valid = 1, max_num_valid = 1}

// use nmap constant for avgpooling.
%nmap_const = ddl.get_external_constant(%type_fp16){name="nmap", num_elements=1}

// Get reference to the allocate spaces for input and output tensors in LX
%allocate_handler_input_lx = ddl.get_external_data_transfer_allocation (%inptensor) {
  memory="lx", data_connect="l3_lx_input"}
%allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) {
   memory="lx", data_connect="lx_output"}
// For average-nmap-pool, get reference to the allocate space for the nmap
// tensor in LX.
%allocate_handler_nmap_lx = ddl.get_external_data_transfer_allocation (%nmap_tensor) {
   memory="lx", data_connect="l3_lx_nmap"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %d#0, %d#1, %wrd#0, %wrd#1) {values = ["1","2","4"]}
  // for mb in (0, Nmb)
  //   for i in (0, Ni)
  //     for j in (0, Nj)
  //       for out in (0, Nout)
  //         for ki in (0, Nki)
  //           for kj in (0, Nkj)
  //             Output[out, mb, i, j] = MAX(Output[out, mb, i, j], Input[out, mb, i + ki, j + kj])
  
  %sfp_nmap_allocation = ddl.allocate(%nmap_const){memory = "sfplrf"}
  %sfp_nmap_const = ddl.unit(%nmap_const, %sfp_nmap_allocation){unit="sfp", data_connect="nmap_const_sfp_lrf"}

  ddl.if(%avgpool_op) {
    %src_nmap_const = ddl.unit(%nmap_const){unit="constant", data_connect="nmap_const_external"}
    ddl.data_transfer(%src_nmap_const, [%sfp_nmap_const]){}
  }

  ddl.loop (%d_datastage, %b_datastage, %d#0, %wrd#0, %wrd#1, %d#1){} {
    ddl.loop (%b_datastage, %interleave_datastage, %wrd#0, %wrd#1){} {
      // TO DO: Consider placing the transfer of nmap tensor inside the interleave
      // loop below and keeping a single loop for
      //   ddl.loop (%b_datastage, %interleave_datastage,%d#0, %d#1, %wrd#0, %wrd#1){}
      // Then rely on DDC transformation to move the transfer up by splitting the
      // interleave loop.
      // This way, we do not enforce a loop order in the DDL.

      // Allocate memory in SFP LRF for storing the nmap tensor.
      %nmap_tensor_sfp_allocation = ddl.allocate(%nmap_tensor) {memory="sfplrf"}
      %sfp_nmap_tensor = ddl.unit(%nmap_tensor, %nmap_tensor_sfp_allocation) {
        unit="sfp", data_connect="sfp_nmap_tensor"}
      ddl.if(%avgpoolnmap_op) {
        // The nmap tensor for avgpool has only I and J dimensions.
        // The tensor is broadcast along other dimensions.
        %nmap_tensor_lx = ddl.unit(%nmap_tensor, %allocate_handler_nmap_lx) {unit="lxlu", data_connect="l3_lx_nmap"}
        ddl.data_transfer(%nmap_tensor_lx, [%sfp_nmap_tensor]){}
      }

      ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1){} {

        // Allocate memory in SFP LRF for storing the accumulated result
        %outtensor_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
        %sfp_accumulator = ddl.unit(%outtensor, %outtensor_sfp_allocation) {
          unit="sfp", data_connect="sfp_outtensor"}
        %sfp_fma_dst00 = ddl.unit(%outtensor) {
          unit="lxsu", data_connect="lxsu_result_tensor"}

        // Perform compute for one input window (i.e. one OUT channel).
        //   Note: The loop requires window dimensions.
        ddl.loop(%b_datastage, %bottom_datastage, %w#0, %w#1 ){label="window_loop"} {

          %is_first_iteration_w0 = ddl.condition(%w#0) {loop_label="window_loop", condition="eq", value_expr="first"}
          %is_first_iteration_w1 = ddl.condition(%w#1) {loop_label="window_loop", condition="eq", value_expr="first"}
          %is_first_iteration = ddl.condition_and(%is_first_iteration_w0, %is_first_iteration_w1)

          %is_last_iteration_w0 = ddl.condition(%w#0) {loop_label="window_loop", condition="eq", value_expr="last"}
          %is_last_iteration_w1 = ddl.condition(%w#1) {loop_label="window_loop", condition="eq", value_expr="last"}
          %is_last_iteration = ddl.condition_and(%is_last_iteration_w0, %is_last_iteration_w1)

          ddl.loop(%interleave_datastage, %bottom_datastage, %d#0, %d#1, %wrd#0, %wrd#1){} {
            // Read input from LX into SFP FIFO
            %src_inp_lxsfp = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
            %src_lowered_inp_sfp_fifo_tr = ddl.unit(%inptensor) {unit="sfp", data_connect="sfp_fifo_lowered_input"}
            ddl.data_transfer(%src_inp_lxsfp, [%src_lowered_inp_sfp_fifo_tr], [%wrd0_pad, %wrd1_pad]) {
              access_pattern_style=["padded_wzeropad-to-lowered_padded","padded_wzeropad-to-lowered_padded"]}

            %src_lowered_inp_sfp_fifo = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_fifo_lowered_input"}

            ddl.if(%is_first_iteration) {
              // Initialize accumulator %sfp_accumulator with the input (direct transfer).
              ddl.if (%is_last_iteration) {
                ddl.if (%avgpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_const, %zero_const], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if (%avgpoolnmap_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_tensor, %zero_const], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if(%maxpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %one_const, %zero_const], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
              } else {
                ddl.if (%avgpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_const, %zero_const], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if (%avgpoolnmap_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_tensor, %zero_const], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if(%maxpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %one_const, %zero_const], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
              }
            } else {
              // Read next input and compute according to the opFunction.
              // compute in sfp
              ddl.if (%is_last_iteration) {
                // Place result directly in lxsu.
                ddl.if (%avgpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_const, %sfp_accumulator], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if (%avgpoolnmap_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_tensor, %sfp_accumulator], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if(%maxpool_op) {
                  // Maxpooling
                  // Read next input and perform FMAX operation.
                  // mode = 0: (SrcX > SrcZ) ? SrcX : SrcZ
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_accumulator], [%sfp_fma_dst00]) {computetype="FMAX", unit="sfp"}
                }
              } else {
                // Accumulate result.
                ddl.if (%avgpool_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_const, %sfp_accumulator], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if (%avgpoolnmap_op) {
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_nmap_tensor, %sfp_accumulator], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
                ddl.if(%maxpool_op) {
                  // Maxpooling
                  // Read next input and perform FMAX operation.
                  // mode = 0: (SrcX > SrcZ) ? SrcX : SrcZ
                  ddl.compute([%src_lowered_inp_sfp_fifo, %sfp_accumulator], [%sfp_accumulator]) {computetype="FMAX", unit="sfp"}
                }
              }
            }
          }
        }

        // Write the accumulated result to LX
        %dst_out_lxsu = ddl.unit(%outtensor) {unit="sfp", data_connect="lxsu_result_tensor"}
        %dst_out_lx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="lx_output"}
        ddl.data_transfer(%dst_out_lxsu, [%dst_out_lx]) {}
      }
    }
  }
}

ddl.transformations {
}

}

