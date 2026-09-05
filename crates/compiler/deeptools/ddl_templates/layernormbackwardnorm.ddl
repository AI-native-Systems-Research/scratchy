//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
    %od:4 = ddl.dimension {} : index, index, index, index // i j mb x
    %out  = ddl.dimension {} : index // out

    %slice_layout     = ddl.layout(%out) {is_order_fixed=true}
    %stick_layout     = ddl.layout(%out) {is_order_fixed=true}
    %global_layout    = ddl.layout(%out, %od#0, %od#1, %od#2, %od#3) {}
    %global_layout_od = ddl.layout(%od#0, %od#1, %od#2, %od#3) {}

    %type_fp16 = ddl.type {data_type="SEN169_FP16", bit_width=16}

    %dnorm, %norm, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout,    [%type_fp16]) : index, index, index
    %e1, %e2, %rstd           = ddl.tensor(%slice_layout, %stick_layout, %global_layout_od, [%type_fp16]) : index, index, index

    %layernormbackwardnorm_op = ddl.operation_bind([%type_fp16], [%dnorm, %e1, %norm, %e2, %rstd], [%outtensor]) {opFuncName="layernormbackwardnorm", required=true}

    ddl.constraint(%outtensor, %dnorm, %e1, %norm, %e2, %rstd) {property = "slice", cmp = "equal"}
    ddl.constraint(%outtensor, %dnorm, %e1, %norm, %e2, %rstd) {property = "stick", cmp = "equal"}
    ddl.constraint() {min_num_cores = 1}

    // allocate space: lx
    %dnorm_lx_allocation = ddl.get_external_data_transfer_allocation (%dnorm) {memory="lx", data_connect="l3_lx_dnorm"}
    %e1_lx_allocation    = ddl.get_external_data_transfer_allocation (%e1)    {memory="lx", data_connect="l3_lx_e1"}
    %norm_lx_allocation  = ddl.get_external_data_transfer_allocation (%norm)  {memory="lx", data_connect="l3_lx_norm"}
    %e2_lx_allocation = ddl.get_external_data_transfer_allocation (%e2) {memory="lx", data_connect="l3_lx_e2"}
    %rstd_lx_allocation = ddl.get_external_data_transfer_allocation (%rstd) {memory="lx", data_connect="l3_lx_rstd"}
    %outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) {memory="lx", data_connect="lxsu_output"}

    %zero_const = ddl.operand_constant{name="0.0"}
    %one_const  = ddl.operand_constant{name="1.0"}

    ddl.dataflow {
        %d_datastage = ddl.get_external_datastage {property = "core"}
        %b_datastage = ddl.get_external_datastage {property = "chunk"}
        %interleave_datastage = ddl.datastage {strategy="maximize"}
        %bottom_datastage = ddl.datastage {strategy="minimize"}

        ddl.loop (%d_datastage, %b_datastage, %out, %od#0, %od#1, %od#2, %od#3){} {
            ddl.loop (%b_datastage, %interleave_datastage, %od#0, %od#1, %od#2, %od#3){} {
                // PE: e1
                %e1_pe_allocation = ddl.allocate(%e1) {memory="pelrf"}
                %src_e1_lxpe = ddl.unit(%e1, %e1_lx_allocation) {unit="lxlu", data_connect="l3_lx_e1"}
                %dst_e1_lxpe = ddl.unit(%e1, %e1_pe_allocation) {unit="pe", data_connect="pe_e1"}
                ddl.data_transfer(%src_e1_lxpe, [%dst_e1_lxpe]) {}

                // SFP: e2
                %e2_sfp_allocation = ddl.allocate(%e2) {memory="sfplrf"}
                %src_e2_lxsfp = ddl.unit(%e2, %e2_lx_allocation) {unit="lxlu", data_connect="l3_lx_e2"}
                %dst_e2_lxsfp = ddl.unit(%e2, %e2_sfp_allocation) {unit="sfp", data_connect="sfp_e2"}
                ddl.data_transfer(%src_e2_lxsfp, [%dst_e2_lxsfp]) {}

                // SFP: rstd
                %rstd_sfp_allocation = ddl.allocate(%rstd) {memory="sfplrf"}
                %src_rstd_lxpe = ddl.unit(%rstd, %rstd_lx_allocation) {unit="lxlu", data_connect="l3_lx_rstd"}
                %dst_rstd_lxpe = ddl.unit(%rstd, %rstd_sfp_allocation) {unit="sfp", data_connect="sfp_rstd"}
                ddl.data_transfer(%src_rstd_lxpe, [%dst_rstd_lxpe]) {}

                ddl.loop (%b_datastage, %bottom_datastage, %out){} {
                    // PE: dnorm
                    %src_dnorm_lxpe = ddl.unit(%dnorm, %dnorm_lx_allocation) {unit="lxlu", data_connect="l3_lx_dnorm"}
                    %dst_dnorm_lxpe = ddl.unit(%dnorm) {unit="pe", data_connect="pe_dnorm"}
                    ddl.data_transfer(%src_dnorm_lxpe, [%dst_dnorm_lxpe]) {}

                    %pe_fnms_src00 = ddl.unit(%e1, %e1_pe_allocation) {unit="pe", data_connect="pe_e1"}
                    %pe_fnms_src01 = ddl.unit(%dnorm) {unit="lxlu", data_connect="pe_dnorm"}
                    %pe_fnms_dst00 = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_output"}
                    // PE: -(e1 * 1 - dnorm) = dnorm - e1
                    ddl.loop (%interleave_datastage, %bottom_datastage, %od#0, %od#1, %od#2, %od#3){} {
                        ddl.compute([%pe_fnms_src00, %one_const, %pe_fnms_src01], [%pe_fnms_dst00]) {computetype="FNMS", unit="pe"}
                    }

                    // PE-SFP
                    %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="pe_output"}
                    %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_input"}
                    ddl.data_transfer(%src_out_pesfp, [%dst_out_pesfp]) {}

                    // SFP: norm
                    %src_norm_lxsfp = ddl.unit(%norm, %norm_lx_allocation) {unit="lxlu", data_connect="l3_lx_norm"}
                    %dst_norm_lxsfp = ddl.unit(%norm) {unit="sfp", data_connect="sfp_norm"}
                    ddl.data_transfer(%src_norm_lxsfp, [%dst_norm_lxsfp]) {}

                    %out_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
                    %sfp_fnms_src00 = ddl.unit(%e2, %e2_sfp_allocation) {unit="sfp", data_connect="sfp_e2"}
                    %sfp_fnms_src01 = ddl.unit(%norm) {unit="lxlu", data_connect="sfp_norm"}
                    %sfp_fnms_src02 = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_input"}
                    %sfp_fnms_dst00 = ddl.unit(%outtensor, %out_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor2"}
                    // SFP: -(e2 * norm - out) = -(e2 * norm - dnorm + e1) = dnorm - e1 - e2 * norm
                    ddl.loop (%interleave_datastage, %bottom_datastage, %od#0, %od#1, %od#2, %od#3){} {
                        ddl.compute([%sfp_fnms_src00, %sfp_fnms_src01, %sfp_fnms_src02], [%sfp_fnms_dst00]) {computetype="FNMS", unit="sfp"}
                    }

                    %sfp_fma_src00 = ddl.unit(%outtensor, %out_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor2"}
                    %sfp_fma_src01 = ddl.unit(%rstd, %rstd_sfp_allocation) {unit="sfp", data_connect="sfp_rstd"}
                    %sfp_fma_dst00 = ddl.unit(%outtensor) {unit="lxsu", data_connect="sfp_outtensor3"}
                    // SFP: out * rstd + 0
                    ddl.loop (%interleave_datastage, %bottom_datastage, %od#0, %od#1, %od#2, %od#3){} {
                        ddl.compute([%sfp_fma_src00, %sfp_fma_src01, %zero_const], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
                    }

                    // SFP-LX
                    %src_out_sfplx = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_outtensor3"}
                    %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
                    ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
                }
            }
        }
    }
}
