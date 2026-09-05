//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}

//   Uses different layout for input and output tensors.
%in_i, %in_j = ddl.dimension {} : index, index  // Expected to be I and J 
%mb, %out = ddl.dimension {} : index, index    // Expected to be mb and out
// These dimensions can be used in a window operation context
%ki, %kj = ddl.dimension { dim_property = "window" } : index, index
%in_ch = ddl.dimension {} : index
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
%wrd0_pad = ddl.padded_dimension(primary=%in_i, padding=[%pf#0, %pb#0, %pv#0], window=[%ki, %s#0, %dl#0] )
%wrd1_pad = ddl.padded_dimension(primary=%in_j, padding=[%pf#1, %pb#1, %pv#1], window=[%kj, %s#1, %dl#1] )

%slice_layout = ddl.layout() {is_order_fixed=false}  
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout_input = ddl.layout(%wrd0_pad, %wrd1_pad, %mb, %out) {}
%global_layout_output = ddl.layout(%in_i, %in_j, %mb, %out) {}
%global_layout_kernel = ddl.layout(%ki, %kj, %in_ch, %out) {} //kernel

%type_fp16 = ddl.type { data_type="SEN169_FP16", bit_width=16 }
%inptensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_input, [%type_fp16]) : index
%outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_output, [%type_fp16]) : index
%kertensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_kernel, [%type_fp16]) : index
%bnA, %bnB = ddl.tensor(%slice_layout, %stick_layout, %global_layout_output, [%type_fp16]) : index, index
%depthwise_conv_op = ddl.operation_bind([%type_fp16], [%inptensor, %kertensor, %outtensor], [%outtensor]) {opFuncName="depthwiseconv2dnative", required=true}
%bn_op = ddl.operation_bind([%type_fp16], [%outtensor, %bnA, %bnB], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%relu_op = ddl.operation_bind([%type_fp16], [%outtensor], [%outtensor]) {opFuncName="relufwd", required=false}


ddl.constraint(%inptensor, %outtensor, %kertensor) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor, %outtensor, %kertensor) {property = "stick", cmp = "equal"}

// Get reference to the allocate spaces for input and output tensors in LX
%allocate_handler_input_lx = ddl.get_external_data_transfer_allocation (%inptensor) {
  memory="lx", data_connect="l3_lx_input"}
%allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) {
   memory="lx", data_connect="l3_lx_output"}
%allocate_handler_kernel_lx = ddl.get_external_data_transfer_allocation (%kertensor) {
   memory="lx", data_connect="l3_lx_kernel"}
%bnA_lx_allocation = ddl.get_external_data_transfer_allocation (%bnA) {memory="lx", data_connect="l3_lx_bnA"}
%bnB_lx_allocation = ddl.get_external_data_transfer_allocation (%bnB) {memory="lx", data_connect="l3_lx_bnB"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage = ddl.datastage {strategy="minimize"}
  %kernel_block_load = ddl.datastage {strategy="maximize"}

  ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j) {values = ["1","2","4"]}
  %is_any_aux_op = ddl.condition_or(%bn_op, %relu_op)
  %not_any_aux_op = ddl.condition_not(%is_any_aux_op)

  %is_bn_not_last = ddl.condition_or(%relu_op)


  // for out in (0, Nout) // output channel or number of kernels
  //   for mb in (0, Nmb) // mini batch size
  //     for i in (0, Ni) // i, j -> 2d input
  //       for j in (0, Nj/Intj)
  //           for ki in (0, Nki) // ki, kj ->  2d kernel
  //             for kj in (0, Nkj) 
  //               for j in (0, Intj)
  //                 Output[out, mb, i, j] = FMA(Input[out, mb, i + ki, j + kj], Kernel[out, ki, kj], Output[out, mb, i, j])

  ddl.loop (%d_datastage, %b_datastage, %mb, %in_i, %in_j, %ki, %kj, %out){label = "chunk_loop"} {
    %cond_last_chunk_ki_loop = ddl.condition(%ki){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %cond_last_chunk_kj_loop = ddl.condition(%kj){loop_label="chunk_loop", condition="eq", value_expr="last"}
    %is_block_load_bn = ddl.condition_and(%cond_last_chunk_ki_loop, %cond_last_chunk_kj_loop, %bn_op)
    ddl.loop(%b_datastage, %interleave_datastage, %out) {} {

      // block load bn / biasAdd param to sfp
      %bnA_sfp_allocation = ddl.allocate(%bnA) {memory="sfplrf"}
      %bnB_sfp_allocation = ddl.allocate(%bnB) {memory="sfplrf"}
      %sfp_bnA_lrf = ddl.unit(%bnA, %bnA_sfp_allocation) {unit="sfp", data_connect="bnA_sfp_lrf"}
      %sfp_bnB_lrf = ddl.unit(%bnB, %bnB_sfp_allocation) {unit="sfp", data_connect="bnB_sfp_lrf"}
      ddl.if (%is_block_load_bn) {
        %bnA_inp_lxsfp = ddl.unit(%bnA, %bnA_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA"}
        ddl.data_transfer(%bnA_inp_lxsfp, [%sfp_bnA_lrf])

        %bnB_inp_lxsfp = ddl.unit(%bnB, %bnB_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB"}
        ddl.data_transfer(%bnB_inp_lxsfp, [%sfp_bnB_lrf])
      }

      ddl.loop (%b_datastage, %kernel_block_load, %ki, %kj){label = "kernel_block_load" } {
        %cond_last_blk_ki_loop = ddl.condition(%ki){loop_label="kernel_block_load", condition="eq", value_expr="last"}
        %cond_last_blk_kj_loop = ddl.condition(%kj){loop_label="kernel_block_load", condition="eq", value_expr="last"}

        %kernel_inp_lxpe = ddl.unit(%kertensor, %allocate_handler_kernel_lx) {unit="lxlu", data_connect="l3_lx_kernel"}
        %kernel_pe_allocation = ddl.allocate(%kertensor) {memory="pelrf"}
        %pe_kernel_dst00 = ddl.unit(%kertensor, %kernel_pe_allocation) {unit="pe", data_connect="kernel_pe_lrf"}
        ddl.data_transfer(%kernel_inp_lxpe, [%pe_kernel_dst00])
        %cond_first_blockload0 = ddl.condition(%ki) {loop_label="kernel_block_load", condition="eq", value_expr="first"}
        %cond_first_blockload1 = ddl.condition(%kj) {loop_label="kernel_block_load", condition="eq", value_expr="first"}
        %cond_first_blockload_and= ddl.condition_and(%cond_first_blockload0, %cond_first_blockload1)
        %cond_not_first_blockload = ddl.condition_not(%cond_first_blockload_and)
        ddl.if(%cond_not_first_blockload) {
          ddl.sync {units=["lxsu"], is_receive=false, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
          ddl.sync {units=["lxlu"], is_receive=true, signal_name="input-lxsu-lxlu-sync", separate_corelets=true} 
        }  
        ddl.loop (%b_datastage, %interleave_datastage, %mb, %in_i, %in_j){} {
          ddl.if(%cond_not_first_blockload) {
            // load output from lx and send to sfp for accumulation
            %src_out_lxsfp = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxlu", data_connect="l3_lx_output"}
            %src_lowered_out_sfp_fifo_tr = ddl.unit(%outtensor) {unit="sfp", data_connect="outtensor_to_sfp"}
            ddl.data_transfer(%src_out_lxsfp, [%src_lowered_out_sfp_fifo_tr])
          }
          // Allocate memory in SFP LRF for accumulating result
          %outtensor_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
          %sfp_accumulator = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor"}
          %partial_output = ddl.unit(%outtensor) {unit="lxlu", data_connect = "outtensor_to_sfp"}
          //   Note: The loop requires window dimensions.
          %is_last_iter_of_all_outer_ki_kj_loop = ddl.condition_and(%cond_last_chunk_ki_loop, %cond_last_chunk_kj_loop, %cond_last_blk_ki_loop, %cond_last_blk_kj_loop, %is_any_aux_op)
          ddl.loop(%kernel_block_load, %bottom_datastage, %ki, %kj) {label="conv_loop"} {
            %is_first_iteration_w0 = ddl.condition(%ki) {loop_label="conv_loop", condition="eq", value_expr="first"}
            %is_first_iteration_w1 = ddl.condition(%kj) {loop_label="conv_loop", condition="eq", value_expr="first"}
            %is_first_iteration = ddl.condition_and(%is_first_iteration_w0, %is_first_iteration_w1)
    
            %is_last_iteration_w0 = ddl.condition(%ki) {loop_label="conv_loop", condition="eq", value_expr="last"}
            %is_last_iteration_w1 = ddl.condition(%kj) {loop_label="conv_loop", condition="eq", value_expr="last"}

            %is_not_last_iteration_w0 = ddl.condition(%ki) {loop_label="conv_loop", condition="ne", value_expr="last"}
            %is_not_last_iteration_w1 = ddl.condition(%kj) {loop_label="conv_loop", condition="ne", value_expr="last"}
            %is_write_to_reg_cond = ddl.condition_or(%is_not_last_iteration_w0, %is_not_last_iteration_w1, %is_last_iter_of_all_outer_ki_kj_loop)

            %is_not_first_iteration = ddl.condition_not(%is_first_iteration)

            ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
              // Read input from LX into PE FIFO
              %src_inp_lxpe = ddl.unit(%inptensor, %allocate_handler_input_lx) {unit="lxlu", data_connect="l3_lx_input"}
              %src_lowered_inp_pe_fifo_tr = ddl.unit(%inptensor) {unit="pe", data_connect="lxpe_fifo"}
              ddl.data_transfer(%src_inp_lxpe, [%src_lowered_inp_pe_fifo_tr])
              // PE FMA result to SFP
              %input_fifo = ddl.unit(%inptensor) {unit="lxlu", data_connect="lxpe_fifo"}
              %pe_fma = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_outtensor"}
              ddl.compute([%input_fifo, %pe_kernel_dst00, %zero_const], [%pe_fma]) {computetype="FMA16", unit="pe"}
            }
            ddl.if(%is_not_first_iteration) {
              ddl.if (%is_write_to_reg_cond) {
                ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                  // for accumulating output
                  %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                  ddl.compute([%sfp_fma_inp, %one_const, %sfp_accumulator], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                }
              } else {
                // Place result directly in lxsu.
                ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                  // for accumulating output
                  %sfp_fma_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
                  %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                  ddl.compute([%sfp_fma_inp, %one_const, %sfp_accumulator], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                }
              }
            } else {
              // It is first iteration of the window accumulation
              ddl.if(%cond_not_first_blockload) {
                ddl.if (%is_write_to_reg_cond) {
                  ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                    // for accumulating output
                    %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                    ddl.compute([%sfp_fma_inp, %one_const, %partial_output], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                  }
                } else {
                  ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                    // for accumulating output
                    %sfp_fma_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
                    %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                    // Place result directly in lxsu.
                    ddl.compute([%sfp_fma_inp, %one_const, %partial_output], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                  }
                }
              } else {
                ddl.if (%is_write_to_reg_cond) {
                  ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                    // for accumulating output
                    %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                    ddl.compute([%sfp_fma_inp, %one_const, %zero_const], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                  }
                } else {
                  ddl.loop(%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j){} {
                    // for accumulating output
                    %sfp_fma_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
                    %sfp_fma_inp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                    // Place result directly in lxsu.
                    ddl.compute([%sfp_fma_inp, %one_const, %zero_const], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                  }
                }
              }
            }
          }
          ddl.if (%is_last_iter_of_all_outer_ki_kj_loop) {
            // bn
            ddl.if (%bn_op) {
              ddl.loop (%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j) {} {
                %sfp_bn_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
                ddl.if(%is_bn_not_last) {
                  // bn: lrf * lrf + lrf -> lxsu
                  ddl.compute([%sfp_accumulator, %sfp_bnA_lrf, %sfp_bnB_lrf], [%sfp_accumulator]) {computetype="FMA16", unit="sfp"}
                } else {
                  // bn: lrf * lrf + lrf -> lrf
                  ddl.compute([%sfp_accumulator, %sfp_bnA_lrf, %sfp_bnB_lrf], [%sfp_bn_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
                }
              }
            }

            // relu
            ddl.if (%relu_op) {
              ddl.loop (%interleave_datastage, %bottom_datastage, %mb, %out, %in_i, %in_j) {} {
                %sfp_relu_dst00_lxsu = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
                // always last
                // relu: lrf -> lxsu
                ddl.compute([%sfp_accumulator, %zero_const], [%sfp_relu_dst00_lxsu]) {computetype="FMAX", unit="sfp"}
              }
            }
          }
          // Write the accumulated result to LX
          %dst_out_lxsu = ddl.unit(%outtensor) {unit="sfp", data_connect="lxsu_result_tensor"}
          %dst_out_lx = ddl.unit(%outtensor, %allocate_handler_output_lx) {unit="lxsu", data_connect="l3_lx_output"}
          ddl.data_transfer(%dst_out_lxsu, [%dst_out_lx]) {}
        }
      }
    }
  }
}

ddl.transformations {
}

}

