//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// dimensions..
%d:6 = ddl.dimension{} : index, index, index, index, index, index// 6 dimensions of the input and output tensors
%asdin:1 = ddl.dimension{} : index//, index
%asdout:1 = ddl.dimension {}: index//, index 

// layouts..
%slice_layout = ddl.layout(%asdin) {is_order_fixed=false}   
%stick_layout_input = ddl.layout(%asdin) {is_order_fixed=false} 
%global_layout_input = ddl.layout(%d#0, %d#1, %d#2,%d#3, %d#4, %d#5, %asdin, %asdout) {}  //layout constraints, add all dimensions


 
%stick_layout_output = ddl.layout(%asdout) {is_order_fixed=false} //to make sure asdout#0 gets mapped to innermost
%global_layout_output = ddl.layout(%d#0, %d#1, %d#2,%d#3, %d#4, %d#5, %asdin, %asdout) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_fp8 = ddl.type {data_type="SEN143_FP8"}

// tensors.. 
%inptensor_fp16 = ddl.tensor(%slice_layout, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%outtensor_fp16 = ddl.tensor(%slice_layout, %stick_layout_output, %global_layout_output, [%type_fp16]) : index
%inptensor_fp8 = ddl.tensor(%slice_layout, %stick_layout_input, %global_layout_input, [%type_fp8]) : index
%outtensor_fp8 = ddl.tensor(%slice_layout, %stick_layout_output, %global_layout_output, [%type_fp8]) : index

// operations

%ist_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16], [%outtensor_fp16]) {opFuncName="interslicetranspose_fp16", required=false}
%ist_fp8_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8], [%outtensor_fp8]) {opFuncName="interslicetranspose_fp8", required=false}


//constraint
ddl.constraint(%ist_fp16_op,%ist_fp8_op){min_num_valid = 1, max_num_valid = 1}
ddl.constraint() {min_num_cores = 1}

// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}

// alias input, kernel, ptsum tensor
%inptensor = ddl.alias_one_tensor_of(%inptensor_fp16, %inptensor_fp8) 
%outtensor = ddl.alias_one_tensor_of(%outtensor_fp16, %outtensor_fp8) 

// lx space allocation --// allocate space: lx
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="lxlu_input"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) {memory="lx", data_connect="lxsu_input"}




ddl.dataflow {
  // datastages
   %d_datastage = ddl.get_external_datastage{property = "core"}
   %b_datastage = ddl.get_external_datastage {property = "chunk"}
   
   %l0subchunk_datastage = ddl.datastage {strategy="minimize"} 
   %bottom_datastage = ddl.datastage{strategy="minimize"} // to send or broadcast one stick at a time
   

   ddl.datastage_constraint(%l0subchunk_datastage, %outtensor, %asdout) {values=["1"]} 
   ddl.datastage_constraint(%bottom_datastage, %outtensor, %asdout) {values=["0.125"]} 
   
   
   
   ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5, %asdin, %asdout){label="chunk_loop"} { 
        // ddl.loop (%b_datastage, %_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5, %asdin, %asdout){label="subchunk1_loop"} {
         ddl.loop (%b_datastage, %l0subchunk_datastage, %d#0, %d#1, %d#2,%d#3, %d#4, %d#5, %asdin, %asdout){label="subchunk_loop"} {
            
            
            //lx to l0 //transferring the subchunk corresponding l0subchunk_datastage to l0
            %inptensor_l0_allocation = ddl.allocate(%inptensor) {memory="l0", num_buffers=-1:si64}
            ddl.implicit_sync(%inptensor_l0_allocation) 
            %src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="lxlu_input"}
            %dst_inp_lxl0 = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0su", data_connect="l0_input"} 
            ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0]) {}
            
            //l0 to ptarf 
            
            %outtensor_arf_allocation = ddl.allocate(%outtensor) {memory="ptarf"}
            // %dst_inp_l0arf = ddl.unit(%inptensor, %inptensor_arf_allocation) {unit="pt", data_connect="arf_pt"} 
            ddl.loop(%l0subchunk_datastage, %bottom_datastage,%asdout){}{
                //ddl.loop(%l0subchunk_datastage, %bottom_datastage,%asdout#0){}{
                    %inptensor_arf_allocation = ddl.allocate(%inptensor) {memory="ptarf"}
                    %src_inp_l0arf = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0lu", data_connect="l0_input"}
                    %dst_inp_l0arf = ddl.unit(%inptensor, %inptensor_arf_allocation) {unit="pt", data_connect="arf_pt"} 
                    %src_out_compute = ddl.unit(%outtensor, %outtensor_arf_allocation) {unit="pt", data_connect="compute_out"} 
                    ddl.data_transfer(%src_inp_l0arf, [%dst_inp_l0arf]) {}
                    ddl.compute([%dst_inp_l0arf, %one_const, %zero_const], [%src_out_compute]) {computetype="MACC", unit="pt"}
                //}
            }
            //ptarf to lx via pe input tensor is written to the lx allocated for the slice wise transposed output
            %src_out_arflx = ddl.unit(%outtensor, %outtensor_arf_allocation) {unit="pt", data_connect="compute_out"}
            %dst_out_arflx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", vias=["pe"], data_connect="lxsu_input"} 
            ddl.data_transfer(%src_out_arflx, [%dst_out_arflx]) {} 

                  
        }
   }
}
// }


           

ddl.transformations {
}


}

