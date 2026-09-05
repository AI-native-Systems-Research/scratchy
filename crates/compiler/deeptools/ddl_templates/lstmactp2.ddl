//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%d:6 = ddl.dimension {} : index, index, index, index, index, index
%slice_layout = ddl.layout() {is_order_fixed=false} 
%stick_layout = ddl.layout() {is_order_fixed=false}
%global_layout = ddl.layout(%d#0, %d#1, %d#2, %d#3, %d#4, %d#5) {}
%type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}
%inptensor_i, %inptensor_z, %inptensor_o, %inptensor_f, %inptensor_ct_1, %outtensor_ht = ddl.tensor(%slice_layout, %stick_layout, %global_layout, [%type_fp16]) : index, index, index, index, index, index

%temptensor3 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index
%temptensor6 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index
%temptensor7 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index
%temptensor8 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index
%temptensor9 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index
%temptensor10 = ddl.internal_tensor(%outtensor_ht, [%type_fp16]) : index

%lstmactp2_op = ddl.operation_bind([%type_fp16], [%inptensor_i, %inptensor_z, %inptensor_f, %inptensor_o, %inptensor_ct_1], [%outtensor_ht, %inptensor_ct_1], [%temptensor3, %temptensor6, %temptensor7, %temptensor8, %temptensor9, %temptensor10]) {opFuncName="lstmactp2", required=false}
// 0 ->i , 1->c/z, 2-> f, 3 -> o, 4 -> cell state

ddl.constraint(%inptensor_i, %inptensor_z, %inptensor_o, %inptensor_f, %inptensor_ct_1, %outtensor_ht) {property = "slice", cmp = "equal"}
ddl.constraint(%inptensor_i, %inptensor_z, %inptensor_o, %inptensor_f, %inptensor_ct_1, %outtensor_ht) {property = "stick", cmp = "equal"}
ddl.constraint() {min_num_cores = 1}
ddl.constraint(%lstmactp2_op) {min_num_valid = 1, max_num_valid = 1}
%zero_const = ddl.operand_constant {name="0.0"}
%one_const = ddl.operand_constant {name="1.0"}


// allocate space: lx
%allocate_handler_input_lx_i = ddl.get_external_data_transfer_allocation (%inptensor_i) {memory="lx", data_connect="l3_lx_input_i"}
%allocate_handler_input_lx_z = ddl.get_external_data_transfer_allocation (%inptensor_z) {memory="lx", data_connect="l3_lx_input_z"}
%allocate_handler_input_lx_o = ddl.get_external_data_transfer_allocation (%inptensor_o) {memory="lx", data_connect="l3_lx_input_o"} 
%allocate_handler_input_lx_f = ddl.get_external_data_transfer_allocation (%inptensor_f) {memory="lx", data_connect="l3_lx_input_f"} 
%allocate_handler_input_lx_ct_1 = ddl.get_external_data_transfer_allocation (%inptensor_ct_1) {memory="lx", data_connect="l3_lx_input_ct_1"} 
%allocate_handler_output_ht = ddl.get_external_data_transfer_allocation (%outtensor_ht) { memory="lx", data_connect="lxsu_input_ht"}

ddl.dataflow {
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %bottom_datastage = ddl.datastage {strategy="minimize"}
  ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5) {values=["1", "2", "4"]}
  // main dataflow
  ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {  
    ddl.loop (%b_datastage, %interleave_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
      // tmp1 = i * z
      %src_inp_lx_i = ddl.unit(%inptensor_i, %allocate_handler_input_lx_i) {unit="lxlu", data_connect="l3_lx_input_i"}
      %src_inp_lx_z = ddl.unit(%inptensor_z, %allocate_handler_input_lx_z) {unit="lxlu", data_connect="l3_lx_input_z"}
      %pe_lrf_allocation_i = ddl.allocate(%inptensor_i) {memory="pelrf"}
      %dst_inp_pe_i = ddl.unit(%inptensor_i, %pe_lrf_allocation_i) {unit="pe", data_connect="pe_lx_input_i"}
      %dst_inp_pe_z = ddl.unit(%inptensor_z) {unit="pe", data_connect="pe_lx_input_z"}
      ddl.data_transfer(%src_inp_lx_i, [%dst_inp_pe_i]) {}
      ddl.data_transfer(%src_inp_lx_z, [%dst_inp_pe_z]) {}
      %pe_lrf_allocation_temptensor3 = ddl.allocate(%temptensor3) {memory="pelrf"}
      %pe_dst_iz = ddl.unit(%temptensor3, %pe_lrf_allocation_temptensor3) {unit="pe", data_connect="pe_interim_output_iz"}
      %dst_inp_pe_z_fifo = ddl.unit(%inptensor_z) {unit="lxlu", data_connect="pe_lx_input_z"}
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} { 
        ddl.compute([%dst_inp_pe_i, %dst_inp_pe_z_fifo], [%pe_dst_iz]) {computetype = "FMUL", unit = "pe", mode = 10}
      }

      // tmp2 = f * ct_1
      %src_inp_lx_f = ddl.unit(%inptensor_f, %allocate_handler_input_lx_f) {unit="lxlu", data_connect="l3_lx_input_f"}
      %src_inp_lx_ct_1 = ddl.unit(%inptensor_ct_1, %allocate_handler_input_lx_ct_1) {unit="lxlu", data_connect="l3_lx_input_ct_1"}
      %pe_lrf_allocation_f = ddl.allocate(%inptensor_f) {memory="pelrf"}
      %dst_inp_pe_f = ddl.unit(%inptensor_f, %pe_lrf_allocation_f) {unit="pe", data_connect="pe_lx_input_f"}
      %dst_inp_pe_ct_1 = ddl.unit(%inptensor_ct_1) {unit="pe", data_connect="pe_lx_input_ct_1"}
      ddl.data_transfer(%src_inp_lx_f, [%dst_inp_pe_f]) {}
      ddl.data_transfer(%src_inp_lx_ct_1, [%dst_inp_pe_ct_1]) {}
      %pe_lrf_allocation_temptensor6 = ddl.allocate(%temptensor6) {memory="pelrf"}
      %pe_dst_fct_1 = ddl.unit(%temptensor6, %pe_lrf_allocation_temptensor6) {unit="pe", data_connect="pe_output"}
      %dst_inp_pe_ct_1_fifo = ddl.unit(%inptensor_ct_1) {unit="lxlu", data_connect="pe_lx_input_ct_1"}
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%dst_inp_pe_f, %dst_inp_pe_ct_1_fifo], [%pe_dst_fct_1]) {computetype = "FMUL", unit = "pe", mode = 10}
      }

      // tmp3 = tmp1 + tmp2

      %pe_lrf_allocation_temptensor7 = ddl.allocate(%temptensor7) {memory="pelrf"}
      %pe_dst22 = ddl.unit(%temptensor7) {unit="lxsu", data_connect="pe_output"} // intermediate result neeed to send to lxsu
      %pe_dst23 = ddl.unit(%temptensor7, %pe_lrf_allocation_temptensor7) {unit="pe", data_connect="pe_output"} // intermediate result neeed to send to lxsu
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%pe_dst_iz, %one_const, %pe_dst_fct_1], [%pe_dst22, %pe_dst23]) {computetype = "FMA16", unit = "pe", mode = 0} // c(t)
      }
      %src_out_pe_lx_out_inter = ddl.unit(%inptensor_ct_1) {unit="pe", data_connect="pe_output"}
      %dst_out_pe_lx_inter = ddl.unit(%inptensor_ct_1, %allocate_handler_input_lx_ct_1) {unit="lxsu", data_connect="l3_lx_input_ct_1"}
      ddl.data_transfer(%src_out_pe_lx_out_inter, [%dst_out_pe_lx_inter]) {}

      // tmp4 = tanh(tmp3)
      %pe_allocation00 = ddl.allocate(%temptensor8) {memory="pelrf"}
      %pe_allocation11 = ddl.allocate(%temptensor9) {memory="pelrf"}
      %pe_allocation10 = ddl.allocate(%temptensor10) {memory="pelrf"}
      %pe_tan_out00 = ddl.unit(%temptensor8, %pe_allocation00) {unit="pe", data_connect="pe_outtensor_lrf"}
      %pe_tan_out11 = ddl.unit(%temptensor9, %pe_allocation11) {unit="pe", data_connect="pe_outtensor_lrf"}
      %pe_tan_dst = ddl.unit(%temptensor10, %pe_allocation10) {unit="pe", data_connect="pe_tan_output"}
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%pe_dst23], [%pe_tan_out00]) {computetype="FEST", unit="pe", mode=8}
      }
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%pe_dst23], [%pe_tan_out11]) {computetype="FEST", unit="pe", mode=9}
      }
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%pe_tan_out00, %pe_dst23, %pe_tan_out11], [%pe_tan_dst]) {computetype="FMA16", unit="pe"}
      }

      // ht = o * tmp4
      %src_inp_lx_o = ddl.unit(%inptensor_o, %allocate_handler_input_lx_o) {unit="lxlu", data_connect="l3_lx_input_o"}
      %pe_lrf_allocation_o = ddl.allocate(%inptensor_o) {memory="pelrf"}
      %dst_inp_pe_o = ddl.unit(%inptensor_o) {unit="pe", data_connect="pe_lx_input_o"}
      ddl.data_transfer(%src_inp_lx_o, [%dst_inp_pe_o]) {}
      %dst_inp_pe_o_fifo = ddl.unit(%inptensor_o) {unit = "lxlu", data_connect="pe_lx_input_o"}
      %pe_ht_out = ddl.unit(%outtensor_ht) {unit = "lxsu", data_connect = "pe_ht_out" }
      ddl.loop (%interleave_datastage, %bottom_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5){} {
        ddl.compute([%pe_tan_dst, %dst_inp_pe_o_fifo], [%pe_ht_out]) {computetype = "FMUL" ,  unit = "pe", mode = 10} // h(t)
      }
      // pe-lx
      %src_out_pe_lx_out = ddl.unit(%outtensor_ht) {unit="pe", data_connect="pe_output"}
      %dst_out_pe_lx = ddl.unit(%outtensor_ht, %allocate_handler_output_ht) {unit="lxsu", data_connect="lxsu_input_ht"}
      ddl.data_transfer(%src_out_pe_lx_out, [%dst_out_pe_lx]) {}
    }  
  }
}

ddl.transformations {
}

}
