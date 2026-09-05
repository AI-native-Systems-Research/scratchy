//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// const
%zero_const = ddl.operand_constant{name="0.0"}

%dqkv, %r:3, %token, %mb = ddl.dimension {} : index, index, index, index, index, index

%slice_layout  = ddl.layout(%dqkv) {is_order_fixed=false}  
%stick_layout  = ddl.layout(%dqkv) {is_order_fixed=false}
%global_layout = ddl.layout(%dqkv, %r#0, %r#1, %r#2, %mb, %token) {}
%m_layout      = ddl.layout(%dqkv, %token, %mb) {}
%const_layout  = ddl.layout(%dqkv) {}

%type_fp16 = ddl.type { data_type="SEN169_FP16", bit_width=16 }

%constensor = ddl.tensor(%slice_layout, %stick_layout, %const_layout, [%type_fp16]) : index
%m1tensor   = ddl.tensor(%slice_layout, %stick_layout, %m_layout, [%type_fp16]) : index
%m2tensor   = ddl.tensor(%slice_layout, %stick_layout, %m_layout, [%type_fp16]) : index
%xtensor    = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index
%outtensor  = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index
// Intermediate activation tensor
%iatensor   = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index

%rope_p1_op = ddl.operation_bind([%type_fp16], [ %xtensor, %m2tensor, %constensor ], [%iatensor]) {opFuncName="rope64p1", required=false}
%rope_p2_op = ddl.operation_bind([%type_fp16], [ %xtensor, %iatensor, %m1tensor ], [%outtensor]) {opFuncName="rope64p2", required=false}
ddl.constraint(%rope_p1_op, %rope_p2_op) {min_num_valid = 1, max_num_valid = 1}

ddl.constraint(%xtensor, %iatensor) {property = "slice", cmp = "equal"}
ddl.constraint(%xtensor, %iatensor) {property = "stick", cmp = "equal"}

ddl.if(%rope_p2_op) {
  ddl.constraint(%xtensor, %outtensor) {property = "slice", cmp = "equal"}
  ddl.constraint(%xtensor, %outtensor) {property = "stick", cmp = "equal"}
}

// Get reference to the allocated spaces for input and output tensors in LX
%allocate_handler_x_lx = ddl.get_external_data_transfer_allocation (%xtensor) {
  memory="lx", data_connect="l3_lx_x"}
%allocate_handler_m1_lx = ddl.get_external_data_transfer_allocation (%m1tensor) {
  memory="lx", data_connect="l3_lx_m1"}
%allocate_handler_m2_lx = ddl.get_external_data_transfer_allocation (%m2tensor) {
  memory="lx", data_connect="l3_lx_m2"}
%allocate_handler_const_lx = ddl.get_external_data_transfer_allocation (%constensor) {
  memory="lx", data_connect="l3_lx_const"}  
%allocate_handler_output_lx = ddl.get_external_data_transfer_allocation (%outtensor) {
   memory="lx", data_connect="l3_lx_output"}
%allocate_handler_ia_lx = ddl.get_external_data_transfer_allocation (%iatensor) {
   memory="lx", data_connect="l3_lx_ia"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  ddl.loop (%d_datastage, %b_datastage, %dqkv, %r#0, %r#1, %r#2, %mb, %token){} {
    ddl.if(%rope_p1_op) {
      ddl.loop (%b_datastage, %bottom_datastage, %dqkv){} {
        // Load constant tensor C1
        // Broadcast along MB and X dimension.
        %const_tensor_lxpe = ddl.unit(%constensor, %allocate_handler_const_lx) {unit="lxlu", data_connect="l3_lx_const"}
        %const_tensor_pe_lrf_allocation = ddl.allocate(%constensor) {memory="pelrf"}
        %pe_lrf_const_tensor = ddl.unit(%constensor, %const_tensor_pe_lrf_allocation) {
          unit="pe", data_connect="pe_lrf_const_tensor"}
        ddl.data_transfer(%const_tensor_lxpe, [%pe_lrf_const_tensor]) {}

        ddl.loop (%b_datastage, %bottom_datastage, %token, %mb){} {
          // Load tensor M2 from LXLU to SFP LRF
          // Broadcast along X dimensions
          %m2tensor_lxsfp = ddl.unit(%m2tensor, %allocate_handler_m2_lx) {unit="lxlu", data_connect="l3_lx_m2"}
          %m2_sfp_lrf_allocation = ddl.allocate(%m2tensor) {memory="sfplrf"}
          %m2tensor_tr_sfp_lrf = ddl.unit(%m2tensor, %m2_sfp_lrf_allocation) {
            unit="sfp", data_connect="lx_sfp_lrf_m2_tensor"}
          ddl.data_transfer(%m2tensor_lxsfp, [%m2tensor_tr_sfp_lrf]){rotate_num_elements=32}

          ddl.loop (%b_datastage, %bottom_datastage, %r#0, %r#1, %r#2){} {
            // Load tensor X from LX into PE FIFO
            %xtensor_lxpe = ddl.unit(%xtensor, %allocate_handler_x_lx) {unit="lxlu", data_connect="l3_lx_x"}
            %xtensor_tr_pe_fifo = ddl.unit(%xtensor) {unit="pe", data_connect="lx_pe_fifo_x_tensor"}
            ddl.data_transfer(%xtensor_lxpe, [%xtensor_tr_pe_fifo]) {}

            // PE computation: X * C1
            //   - Result is sent to SFP FIFO
            %xtensor_compute_lx_fifo = ddl.unit(%xtensor) {unit="lxlu", data_connect="lx_pe_fifo_x_tensor"}
            %pe_lrf_compute_const_tensor = ddl.unit(%constensor, %const_tensor_pe_lrf_allocation) {
              unit="pe", data_connect="pe_lrf_const_tensor"}
            %result_sfp_fifo = ddl.unit(%iatensor) {unit="sfp", data_connect="sfp_result_tensor"}
            ddl.compute([%xtensor_compute_lx_fifo, %pe_lrf_compute_const_tensor, %zero_const], [%result_sfp_fifo]) {
              computetype="FMA16", unit="pe"}

            // SFP Computation: (X * C1) * M2 
            %prod_compute_pe_fifo = ddl.unit(%iatensor) {unit="pe", data_connect="sfp_result_tensor"}
            %m2tensor_compute_sfp_lrf = ddl.unit(%m2tensor, %m2_sfp_lrf_allocation) {
              unit="sfp", data_connect="lx_sfp_lrf_m2_tensor"}
            %result_lxsu_fifo = ddl.unit(%iatensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
            ddl.compute([%prod_compute_pe_fifo, %m2tensor_compute_sfp_lrf, %zero_const], [%result_lxsu_fifo]) {
              computetype="FMA16", unit="sfp"}

            // Write the result to LX
            %dst_out_lxsu = ddl.unit(%iatensor) {unit="sfp", data_connect="lxsu_result_tensor"}
            %dst_out_lx = ddl.unit(%iatensor, %allocate_handler_ia_lx) {unit="lxsu", data_connect="l3_lx_ia"}
            ddl.data_transfer(%dst_out_lxsu, [%dst_out_lx]) {}
          }
        }
      }
    }
    ddl.if(%rope_p2_op) {
      ddl.loop (%b_datastage, %bottom_datastage, %dqkv, %token, %mb){} {
        // The computation is performed on SFP instead of PE. The computation
        // requires three inputs - M1, X, and IA.
        //   - M1 has to be loaded into LRF for broadcasting.
        //   - IA can be loaded on LXLU-SFP FIFO.
        //   - X can either be loaded into LRF (causes bubble) or
        //     loaded on PE-SFP FIFO (no bubble in steady-state).
        //  For performance reason associated with the loading of X,
        //  the computation is done on SFP.

        // Load tensor M1 from LXLU to PE LRF
        // Broadcast along X dimensions
        %m1tensor_lxsfp = ddl.unit(%m1tensor, %allocate_handler_m1_lx) {unit="lxlu", data_connect="l3_lx_m1"}
        %m1_sfp_lrf_allocation = ddl.allocate(%m1tensor) {memory="sfplrf"}
        %m1tensor_tr_sfp_lrf = ddl.unit(%m1tensor, %m1_sfp_lrf_allocation) {
          unit="sfp", data_connect="lx_sfp_lrf_m1_tensor"}
        ddl.data_transfer(%m1tensor_lxsfp, [%m1tensor_tr_sfp_lrf]){}

        // Future work: Consider loading multiple tokens (chunking along token)
        ddl.loop (%b_datastage, %bottom_datastage, %r#0, %r#1, %r#2){} {
          // Load tensor X from LX.
          // Option 1: Load X into PE LRF.
          //   This option could cause a bubble in each iteration. When DDC
          //   adds a transformation to allow latch reuse, the LRF option
          //   can be enabled.
          //   
          // %xtensor_lxpe = ddl.unit(%xtensor, %allocate_handler_x_lx) {unit="lxlu", data_connect="l3_lx_x"}
          // %x_pe_lrf_allocation = ddl.allocate(%xtensor) {memory="pelrf"}
          // %xtensor_tr_pe_lrf = ddl.unit(%xtensor, %x_pe_lrf_allocation) {unit="pe", data_connect="lx_pe_lrf_x_tensor"}
          // ddl.data_transfer(%xtensor_lxpe, [%xtensor_tr_pe_lrf]) {}
          //
          // Option 2: Load X from LX into SFP FIFO via PE.
          //   Load X into SFP FIFO that gets input from PE. This option avoids
          //   writing to LRF. In steady state, no bubble is expected.
          %xtensor_lxsfp = ddl.unit(%xtensor, %allocate_handler_x_lx) {unit="lxlu", data_connect="l3_lx_x"}
          %xtensor_tr_sfp_fifo = ddl.unit(%xtensor) {unit="sfp", vias=["pe"], data_connect="lx_pe_sfp_fifo_x_tensor"}
          ddl.data_transfer(%xtensor_lxsfp, [%xtensor_tr_sfp_fifo]) {}

          // Load tensor IA from LX into SFP FIFO
          %iatensor_lxsfp = ddl.unit(%iatensor, %allocate_handler_ia_lx) {unit="lxlu", data_connect="l3_lx_ia"}
          %iatensor_tr_sfp_fifo = ddl.unit(%iatensor) {unit="sfp", data_connect="lx_sfp_fifo_ia_tensor"}
          ddl.data_transfer(%iatensor_lxsfp, [%iatensor_tr_sfp_fifo]) {rotate_num_elements=32}

          // SFP Computation: (X * M1) + IA 
          %ia_compute_sfp_fifo = ddl.unit(%iatensor) {unit="lxlu", data_connect="lx_sfp_fifo_ia_tensor"}
          %m1tensor_compute_sfp_lrf = ddl.unit(%m1tensor, %m1_sfp_lrf_allocation) {
            unit="sfp", data_connect="lx_sfp_lrf_m1_tensor"}
          %xtensor_compute_sfp_fifo = ddl.unit(%xtensor) {unit="pe", data_connect="lx_pe_sfp_fifo_x_tensor"}
          %result_lxsu_fifo = ddl.unit(%outtensor) {unit="lxsu", data_connect="lxsu_result_tensor"}
          ddl.compute([%xtensor_compute_sfp_fifo, %m1tensor_compute_sfp_lrf, %ia_compute_sfp_fifo], [%result_lxsu_fifo]) {
            computetype="FMA16", unit="sfp"}

          // Write the result to LX
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

