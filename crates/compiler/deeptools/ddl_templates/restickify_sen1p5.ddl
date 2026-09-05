//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// dimensions..
%d:3 = ddl.dimension{} : index, index, index // 3 dimensions of the input and output tensors
%asdin = ddl.dimension{} : index//, index stick dimension in input
%asdout = ddl.dimension {}: index//, index stick dimension in output

// layouts..
%slice_layout_input = ddl.layout(%asdin){is_order_fixed=false}   
%stick_layout_input = ddl.layout(%asdin){is_order_fixed=false} 
%global_layout_input = ddl.layout(%d#0, %d#1, %d#2, %asdin, %asdout) {}  //layout constraints, add all dimensions

%slice_layout_intermediate = ddl.layout(%asdout){is_order_fixed=false}
%stick_layout_intermediate = ddl.layout(%asdin) {is_order_fixed=false} //to make sure asdout#0 gets mapped to innermost
%global_layout_intermediate = ddl.layout(%d#0, %d#1, %d#2, %asdin, %asdout) {}

%slice_layout_output = ddl.layout(%asdout){is_order_fixed=false}
%stick_layout_output = ddl.layout(%asdout) {is_order_fixed=false} //to make sure asdout#0 gets mapped to innermost
%global_layout_output = ddl.layout(%d#0, %d#1, %d#2, %asdin, %asdout) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_bfp16 = ddl.type {data_type="BFLOAT16"}

// tensors.. 
%inptensor_fp16 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%intermediatetensor_fp16 = ddl.tensor(%slice_layout_intermediate, %stick_layout_intermediate, %global_layout_intermediate, [%type_fp16]) : index
%outtensor_fp16 = ddl.tensor(%slice_layout_output, %stick_layout_output, %global_layout_output, [%type_fp16]) : index
%internal_outtensor = ddl.internal_tensor(%outtensor_fp16, [%type_fp16]) : index
%internal_outtensor_bf16 = ddl.internal_tensor(%outtensor_fp16, [%type_bfp16]) : index

// operations
%rst_hbm_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %intermediatetensor_fp16], [%outtensor_fp16], [%internal_outtensor, %internal_outtensor_bf16]) {opFuncName="ReStickifyOpHBM", required=false}
%rst_lx_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %intermediatetensor_fp16], [%outtensor_fp16], [%internal_outtensor, %internal_outtensor_bf16]) {opFuncName="ReStickifyOpLx", required=false}
%rst_pthbm_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %intermediatetensor_fp16], [%outtensor_fp16], [%internal_outtensor]) {opFuncName="ReStickifyOpWithPTHBM", required=false}
%rst_ptlx_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %intermediatetensor_fp16], [%outtensor_fp16], [%internal_outtensor]) {opFuncName="ReStickifyOpWithPTLx", required=false}

//constraint
ddl.constraint(%rst_hbm_fp16_op, %rst_lx_fp16_op, %rst_pthbm_fp16_op, %rst_ptlx_fp16_op){min_num_valid = 1, max_num_valid = 1}
ddl.constraint() {min_num_cores = 1}

// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}

// alias input, kernel, ptsum tensor
%inptensor = ddl.alias_one_tensor_of(%inptensor_fp16) 
%internaltensor = ddl.alias_one_tensor_of(%intermediatetensor_fp16)
%outtensor = ddl.alias_one_tensor_of(%outtensor_fp16) 

// lx space allocation --// allocate space: lx
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="lxlu_input"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) {memory="lx", data_connect="lxsu_input"}

ddl.dataflow {
  // datastages
   %core_datastage = ddl.get_external_datastage{property = "core"}
   %chunk_datastage = ddl.get_external_datastage {property = "chunk"}
   %onexrf_datastage = ddl.datastage {strategy="minimize"}
   %restickifyblock_datastage = ddl.datastage {strategy="minimize"}
   %intrasliceblock_datastage = ddl.datastage {strategy="minimize"}
   %rowcombine_datastage = ddl.datastage {strategy = "minimize"}
   %singlesliceblock_datastage = ddl.datastage {strategy = "minimize"}
   %peforward_datastage = ddl.datastage {strategy = "minimize"}
   
   ddl.datastage_constraint(%intrasliceblock_datastage, %internaltensor, %asdout) {values=["1"]}
   ddl.datastage_constraint(%singlesliceblock_datastage, %internaltensor, %asdout) {values=["1"]}
   ddl.datastage_constraint(%singlesliceblock_datastage, %internaltensor, %asdin) {values=["0.125"]}
   ddl.datastage_constraint(%rowcombine_datastage, %singlesliceblock_datastage, %asdin) {values=["8"]}

  ddl.loop (%core_datastage, %restickifyblock_datastage, %d#0, %d#1, %d#2, %asdin, %asdout){label="allxrf_loop"} {
    %outtensor_xrf_allocation = ddl.allocate(%internal_outtensor) {memory="ptxrf", replication=8:si64}
    %cond_first_d0 = ddl.condition(%d#0){loop_label="allxrf_loop", condition="eq", value_expr="first"}
    %cond_first_d1 = ddl.condition(%d#1){loop_label="allxrf_loop", condition="eq", value_expr="first"}
    %cond_first_d2 = ddl.condition(%d#2){loop_label="allxrf_loop", condition="eq", value_expr="first"}
    %cond_first_asdin = ddl.condition(%asdin){loop_label="allxrf_loop", condition="eq", value_expr="first"}
    %cond_first_asdout = ddl.condition(%asdout){loop_label="allxrf_loop", condition="eq", value_expr="first"}
    %cond_first_dims = ddl.condition_and(%cond_first_d0, %cond_first_d1, %cond_first_d2, %cond_first_asdin, %cond_first_asdout)
    ddl.if (%cond_first_dims) {
      ddl.loop (%restickifyblock_datastage, %onexrf_datastage, %d#0, %d#1, %d#2, %asdin, %asdout){label="onexrf_loop"} {
        %dst_xrf_initial = ddl.unit(%internal_outtensor, %outtensor_xrf_allocation) {unit="pt", data_connect="xrf_initial"} 
        ddl.compute([%zero_const, %one_const, %zero_const], [%dst_xrf_initial]) {computetype="MACC", unit="pt"}
      }
    }
  }

  ddl.loop (%core_datastage, %chunk_datastage, %d#0, %d#1, %d#2, %asdin, %asdout){label="chunk_loop"} { 
        ddl.loop (%chunk_datastage, %restickifyblock_datastage, %d#0, %d#1, %d#2, %asdin, %asdout){label="subchunk_loop"} {
            %outtensor_xrf_allocation = ddl.allocate(%outtensor) {memory="ptxrf"}
            ddl.loop (%restickifyblock_datastage, %intrasliceblock_datastage, %asdout){label="intraslice_loop"} {
                %src_inp_lxsfp = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="lxlu_input"}
                %inptensor_sfplrf_allocation = ddl.allocate(%inptensor) {memory="sfplrf"}
                %dst_inp_lxsfp = ddl.unit(%inptensor, %inptensor_sfplrf_allocation) {unit="sfp", data_connect="sfp_input"}
                ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}

                %internaltensor_sfplrf_allocation = ddl.allocate(%internaltensor) {memory="sfplrf"}
                %dst_internal_lxsfp = ddl.unit(%internaltensor, %internaltensor_sfplrf_allocation) {unit="sfp", data_connect="sfp_internal"}
                ddl.compute([%dst_inp_lxsfp], [%dst_internal_lxsfp]) {computetype="assign", unit="sfp", repetition=8}

                %internaltensor_l0_allocation = ddl.allocate(%internaltensor) {memory="l0",num_buffers=-1:si64}
                %dst_out_sfp = ddl.unit(%internaltensor, %internaltensor_l0_allocation) {unit="l0su", data_connect="sfp_output"}
                ddl.implicit_sync(%internaltensor_l0_allocation)
                ddl.data_transfer(%dst_internal_lxsfp, [%dst_out_sfp]) {}

                ddl.loop (%restickifyblock_datastage, %rowcombine_datastage, %asdin){label="combinerow_loop"} {
                    ddl.loop (%rowcombine_datastage, %singlesliceblock_datastage, %asdin){label="singleslice_loop"} {
                        %src_internal_l0fifo = ddl.unit(%internaltensor, %internaltensor_l0_allocation) {unit="l0lu", data_connect="sfp_output"}
                        %dst_internal_l0fifo = ddl.unit(%internaltensor) {unit="pt", data_connect="pt_fifo"} 
                        ddl.data_transfer(%src_internal_l0fifo, [%dst_internal_l0fifo]) {}

                        %dst_out_compute = ddl.unit(%outtensor, %outtensor_xrf_allocation) {unit="pt", data_connect="compute_out"} 
                        %dst_internal_fifo = ddl.unit(%internaltensor) {unit="l0lu", data_connect="pt_fifo"} 
                        ddl.compute([%zero_const, %one_const, %dst_internal_fifo], [%dst_out_compute]) {computetype="MACC", unit="pt"}
                    }
                }
            }
            %src_out_xrflx = ddl.unit(%outtensor, %outtensor_xrf_allocation) {unit="pt", data_connect="compute_out"}
            %dst_out_xrfpe = ddl.unit(%outtensor) {unit="pe", data_connect="pe_input"} 
            ddl.data_transfer(%src_out_xrflx, [%dst_out_xrfpe]) {}

            ddl.loop (%restickifyblock_datastage, %peforward_datastage, %asdin){label="pe_loop"} {
              %src_out_xrfpt = ddl.unit(%outtensor) {unit="pt", data_connect="pe_input"} 
              %dst_out_xrfpelx = ddl.unit(%internal_outtensor_bf16) {unit="lxsu", data_connect="pe_output"} 
              ddl.compute([%src_out_xrfpt, %one_const, %zero_const], [%dst_out_xrfpelx]) {computetype="FMA32", unit="pe"}
            }

            %src_out_xrfpelx = ddl.unit(%outtensor) {unit="pe", data_connect="pe_output"} 
            %dst_out_xrflx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_input"} 
            ddl.data_transfer(%src_out_xrfpelx, [%dst_out_xrflx]) {}
        }
  }
}

ddl.transformations {
}
}

