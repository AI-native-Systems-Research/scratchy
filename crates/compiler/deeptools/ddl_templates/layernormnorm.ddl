//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
%nd = ddl.dimension {} : index // normalization dimension
%od:4 = ddl.dimension {} : index, index, index, index  // other dimensions
%slice_layout = ddl.layout(%nd) {is_order_fixed=true} 
%stick_layout = ddl.layout(%nd) {is_order_fixed=true} 
%global_layout_all = ddl.layout(%nd, %od#0, %od#1, %od#2, %od#3) {} 
%global_layout_nd = ddl.layout(%nd) {}
%type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16} 

%inptensor, %outtensor, %exx2out, %lnsout = ddl.tensor(%slice_layout, %stick_layout, %global_layout_all, [%type_fp16]) : index, index, index, index
%lnA, %lnB =  ddl.tensor(%slice_layout, %stick_layout, %global_layout_nd, [%type_fp16]) : index, index

%layernormnorm_op = ddl.operation_bind([%type_fp16], [%inptensor, %exx2out, %lnsout, %lnA, %lnB], [%outtensor]) {opFuncName="layernormnorm", required=true} 

ddl.constraint(%outtensor, %inptensor, %exx2out, %lnsout, %lnA, %lnB) {property = "slice", cmp = "equal"}
ddl.constraint(%outtensor, %inptensor, %exx2out, %lnsout, %lnA, %lnB) {property = "stick", cmp = "equal"}
ddl.constraint() {min_num_cores = 1}

// allocate space: lx
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"} 
%exx2out_lx_allocation = ddl.get_external_data_transfer_allocation (%exx2out) {memory="lx", data_connect="l3_lx_exx2out"}
%lnsout_lx_allocation = ddl.get_external_data_transfer_allocation (%lnsout) {memory="lx", data_connect="l3_lx_lnsout"}
%lnA_lx_allocation = ddl.get_external_data_transfer_allocation (%lnA) {memory="lx", data_connect="l3_lx_lnA"} 
%lnB_lx_allocation = ddl.get_external_data_transfer_allocation (%lnB) {memory="lx", data_connect="l3_lx_lnB"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_output"}

%zero_const = ddl.operand_constant{name="0.0"}

ddl.dataflow {
    %d_datastage = ddl.get_external_datastage{property = "core"}
    %b_datastage = ddl.get_external_datastage {property = "chunk"}
    %nd_subchunk_datastage = ddl.datastage {strategy="maximize"}
    %bottom_datastage = ddl.datastage {strategy="minimize"}

    ddl.loop (%d_datastage, %b_datastage, %nd, %od#0, %od#1, %od#2, %od#3){} {  
        ddl.loop (%b_datastage, %nd_subchunk_datastage, %nd){} {
            // load lnA and lnB into SFP
            %lnA_sfp_allocation = ddl.allocate(%lnA) {memory="sfplrf"} 
            %src_lnA_lxsfp = ddl.unit(%lnA, %lnA_lx_allocation) {unit="lxlu", data_connect="l3_lx_lnA"}
            %dst_lnA_lxsfp = ddl.unit(%lnA, %lnA_sfp_allocation) {unit="sfp", data_connect="sfp_lnA"} 
            ddl.data_transfer(%src_lnA_lxsfp, [%dst_lnA_lxsfp]) {}

            %lnB_sfp_allocation = ddl.allocate(%lnB) {memory="sfplrf"}
            %src_lnB_lxsfp = ddl.unit(%lnB, %lnB_lx_allocation) {unit="lxlu", data_connect="l3_lx_lnB"} 
            %dst_lnB_lxsfp = ddl.unit(%lnB, %lnB_sfp_allocation) {unit="sfp", data_connect="sfp_lnB"}
            ddl.data_transfer(%src_lnB_lxsfp, [%dst_lnB_lxsfp]) {}

            ddl.loop (%b_datastage, %bottom_datastage, %od#0, %od#1, %od#2, %od#3){} {
                // load exx2out and lnsout into PE
                %lnsout_pe_allocation = ddl.allocate(%lnsout) {memory="pelrf"} 
                %src_lnsout_lxpe = ddl.unit(%lnsout, %lnsout_lx_allocation) {unit="lxlu", data_connect="l3_lx_lnsout"}
                %dst_lnsout_lxpe = ddl.unit(%lnsout, %lnsout_pe_allocation) {unit="pe", data_connect="pe_lnsout"}
                ddl.data_transfer(%src_lnsout_lxpe, [%dst_lnsout_lxpe]) {}

                %src_exx2out_lxpe = ddl.unit(%exx2out, %exx2out_lx_allocation) {unit="lxlu", data_connect="l3_lx_exx2out"}
                %dst_exx2out_lxpe = ddl.unit(%exx2out) {unit="pe", data_connect="pe_tmp_exx2out"} 
                ddl.data_transfer(%src_exx2out_lxpe, [%dst_exx2out_lxpe]) {}

                // pe-fnms 
                %exx2out_pe_allocation = ddl.allocate(%exx2out) {memory="pelrf"}
                %pe_fnms_src00 = ddl.unit(%lnsout, %lnsout_pe_allocation) {unit="pe", data_connect="pe_lnsout"}
                %pe_fnms_src01 = ddl.unit(%exx2out) {unit="lxlu", data_connect="pe_tmp_exx2out"} 
                %pe_fnms_dst00 = ddl.unit(%exx2out, %exx2out_pe_allocation) {unit="pe", data_connect="pe_exx2out"} 
                ddl.compute([%pe_fnms_src00, %pe_fnms_src01, %zero_const], [%pe_fnms_dst00]) {computetype="FNMS", unit="pe"}

                ddl.loop (%nd_subchunk_datastage, %bottom_datastage, %nd){} {
                    // load input tensor lx-pe
                    %src_inp_lxpe = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
                    %dst_inp_lxpe = ddl.unit(%inptensor) {unit="pe", data_connect="pe_lx_input"}  // no need to allocate input in pe-register as its done on the fly
                    ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe]) {}

                    // pe-FMA
                    %pe_fma_src00 = ddl.unit(%inptensor) {unit="lxlu", data_connect="pe_lx_input"}
                    %pe_fma_src01 = ddl.unit(%lnsout, %lnsout_pe_allocation) {unit="pe", data_connect="pe_lnsout"}
                    %pe_fma_src02 = ddl.unit(%exx2out, %exx2out_pe_allocation) {unit="pe", data_connect="pe_exx2out"}
                    %pe_fma_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_outtensor"}
                    ddl.compute([%pe_fma_src00, %pe_fma_src01, %pe_fma_src02], [%pe_fma_dst00]) {computetype="FMA16", unit="pe"}

                    // pe-sfp
                    %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_outtensor"}
                    %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_outtensor"}
                    ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}

                    // sfp-FMA
                    %sfp_fma_src00 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
                    %sfp_fma_src01 = ddl.unit(%lnA, %lnA_sfp_allocation) {unit="sfp", data_connect="sfp_lnA"}
                    %sfp_fma_src02 = ddl.unit(%lnB, %lnB_sfp_allocation) {unit="sfp", data_connect="sfp_lnB"}
                    %sfp_fma_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_lxsu_outtensor"}
                    ddl.compute([%sfp_fma_src00, %sfp_fma_src01, %sfp_fma_src02], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}

                    // store output tensor sfp-lx
                    %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_lxsu_outtensor"}
                    %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                    ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
                }
            } 
        }   
    }
}


ddl.transformations {
}

}
