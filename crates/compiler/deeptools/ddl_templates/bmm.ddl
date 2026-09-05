//RUN: ddl_standalone -d %s 2>&1 | tee %t.ddl
//RUN: ddl_standalone -d %t.ddl

module {
// dimensions..
%wrd:4 = ddl.dimension{} : index, index, index, index // weight reuse dimension -- i, j mb, y
%wrdd:2 = ddl.dimension{} : index, index
%zf = ddl.dimension{dim_property="pad_front"} : index
%zb = ddl.dimension{dim_property="pad_back"} : index
%padvalid = ddl.dimension{dim_property="pad_valid"} : index
%wrdpd = ddl.padded_dimension(primary=%wrdd#0, padding=[%zf, %zb, %padvalid], window=[])
%nrd:3 = ddl.dimension{} : index, index, index  // no reuse dimension -- x, x1
%in, %out = ddl.dimension{} : index, index // channels 
%ki, %kj = ddl.dimension{} : index, index // kernel window

// layouts..
%slice_layout_input = ddl.layout(%in, %wrd#0, %wrdpd) {is_order_fixed=false} 
%slice_layout_input_16bit = ddl.layout(%in) {is_order_fixed=true} 
%stick_layout_input = ddl.layout(%in) {is_order_fixed=true} 
%global_layout_input = ddl.layout(%in, %wrd#0, %wrd#1, %wrd#2, %wrd#3, %nrd#0, %nrd#1, %nrd#2, %wrdpd, %wrdd#1) {}

%slice_layout_kernel = ddl.layout(%in,%out) {is_order_fixed=true} 
%slice_layout_kernel_16bit = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_kernel = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_kernel = ddl.layout(%ki, %kj, %in, %out, %nrd#0, %nrd#1, %nrd#2) {}

%slice_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%stick_layout_output = ddl.layout(%out) {is_order_fixed=true} 
%global_layout_output = ddl.layout(%out, %wrd#0, %wrd#1, %wrd#2, %wrd#3, %nrd#0, %nrd#1, %nrd#2) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%type_int8 = ddl.type {data_type="SENINT8"}
%type_fp8 = ddl.type {data_type="SEN143_FP8"}
%type_int4 = ddl.type {data_type="SENINT4"}
%type_int24 = ddl.type {data_type="SENINT24"}

// tensors.. 
%inptensor_int8 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_int8]) : index
%kertensor_int8 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_int8]) : index
%inptensor_fp16 = ddl.tensor(%slice_layout_input_16bit, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%kertensor_fp16 = ddl.tensor(%slice_layout_kernel_16bit, %stick_layout_kernel, %global_layout_kernel, [%type_fp16]) : index
%inptensor_fp8 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_fp8]) : index
%kertensor_fp8 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_fp8]) : index
%inptensor_int4 = ddl.tensor(%slice_layout_input, %stick_layout_input, %global_layout_input, [%type_int4]) : index
%kertensor_int4 = ddl.tensor(%slice_layout_kernel, %stick_layout_kernel, %global_layout_kernel, [%type_int4]) : index
%outtensor, %bnA, %bnB, %bias, %resadd, %resadd2 = ddl.tensor(%slice_layout_output, %stick_layout_output, %global_layout_output, [%type_fp16]) : index, index, index, index, index, index
%ptsum_fp = ddl.internal_tensor(%outtensor, [%type_fp16]) : index
%ptsum_int = ddl.internal_tensor(%outtensor, [%type_int24]) : index
%pesum = ddl.internal_tensor(%outtensor, [%type_fp16]) : index

// operation..
%bmm_int8_mbkg3_op = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="batchmatmulint8mbkg3", required=false}
%bmm_int8_op = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="batchmatmulint8", required=false}
%bmm_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %kertensor_fp16], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="batchmatmul", required=false}
%bmm_fp8_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8, %kertensor_fp8], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="batchmatmulfp8", required=false}
%bmm_fp8_mb_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8, %kertensor_fp8], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="batchmatmulfp8mb", required=false}
%bmm_int4_op = ddl.operation_bind([%type_int4], [%inptensor_int4, %kertensor_int4], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="batchmatmulint4", required=false}
%mm_int8_op = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="matmulint8", required=false}
%mm_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %kertensor_fp16], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="matmul", required=false}
%mm_fp8_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8, %kertensor_fp8], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="matmulfp8", required=false}
%mm_int4_op = ddl.operation_bind([%type_int4], [%inptensor_int4, %kertensor_int4], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="matmulint4", required=false}
%bmmxrf_int8_op = ddl.operation_bind([%type_int8], [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="batchmatmulxrfint8", required=false}
%bmmxrf_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16, %kertensor_fp16], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="batchmatmulxrf", required=false}
%bmmxrf_fp8_op = ddl.operation_bind([%type_fp8], [%inptensor_fp8, %kertensor_fp8], [%outtensor], [%ptsum_fp, %pesum]) {opFuncName="batchmatmulxrffp8", required=false}
%bmmxrf_int4_op = ddl.operation_bind([%type_int4], [%inptensor_int4, %kertensor_int4], [%outtensor], [%ptsum_int, %pesum]) {opFuncName="batchmatmulxrfint4", required=false}

%fp16_bmm = ddl.condition_or(%bmm_fp16_op, %mm_fp16_op, %bmmxrf_fp16_op) 
%fp8_bmm = ddl.condition_or(%bmm_fp8_op, %bmm_fp8_mb_op, %mm_fp8_op, %bmmxrf_fp8_op) 
%int8_bmm = ddl.condition_or(%bmm_int8_mbkg3_op, %bmm_int8_op, %mm_int8_op, %bmmxrf_int8_op) 
%int4_bmm = ddl.condition_or(%bmm_int4_op, %mm_int4_op, %bmmxrf_int4_op) 

%fp_bmm = ddl.condition_or(%fp16_bmm, %fp8_bmm)
%int_bmm = ddl.condition_or(%int8_bmm, %int4_bmm)

%psum_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="genericpartialreduction", required=false}
%bn_op = ddl.operation_bind([], [%outtensor, %bnA, %bnB], [%outtensor]) {opFuncName="batchnormfwd", required=false}
%bias_op = ddl.operation_bind([], [%outtensor, %bias], [%outtensor]) {opFuncName="biasadd", required=false}
%relu_op = ddl.operation_bind([], [%outtensor], [%outtensor]) {opFuncName="relufwd", required=false}
%stradd_op = ddl.operation_bind([], [%outtensor, %resadd], [%outtensor]) {opFuncName="stridedadd", required=false}
%stradd2_op = ddl.operation_bind([], [%outtensor, %resadd2], [%outtensor]) {opFuncName="stridedadd", required=false}
%bn_or_bias_op = ddl.condition_or(%bn_op, %bias_op)

// constraints.. 
// no need to add stick/slice "size" constraint when there is one  dimension in stick
ddl.constraint(%inptensor_int8) {property = "slice", dim_idx = 0, cmp = "equal", value = 8}
ddl.constraint(%inptensor_int4) {property = "slice", dim_idx = 0, cmp = "equal", value = 16}
ddl.constraint(%inptensor_int8, %inptensor_int4) {property = "slice", dim_idx = 1, cmp = "equal", value = 2}
ddl.constraint(%kertensor_int8, %kertensor_fp8) {property = "slice", dim_idx = 0, cmp = "equal", value = 2}
ddl.constraint(%kertensor_int4) {property = "slice", dim_idx = 0, cmp = "equal", value = 4}
ddl.constraint(%kertensor_int8, %kertensor_fp8, %kertensor_int4) {property = "slice", dim_idx = 1, cmp = "equal", value = 8}
ddl.constraint(%bmm_int8_mbkg3_op, %bmm_int8_op, %bmm_fp16_op, %bmm_fp8_op, %bmm_fp8_mb_op, %bmm_int4_op, 
                %bmmxrf_int8_op, %bmmxrf_fp16_op, %bmmxrf_fp8_op, %bmmxrf_int4_op, 
                %mm_int8_op, %mm_fp16_op, %mm_fp8_op, %mm_int4_op) {min_num_valid = 1, max_num_valid = 1}
ddl.constraint(%bmm_int8_mbkg3_op, %bmm_int8_op, %bmm_fp16_op, %bmm_fp8_op, %bmm_fp8_mb_op, %bmm_int4_op,
                %bmmxrf_int8_op, %bmmxrf_fp16_op, %bmmxrf_fp8_op, %bmmxrf_int4_op, 
                %mm_int8_op, %mm_fp16_op, %mm_fp8_op, %mm_int4_op,
                %psum_op, %stradd_op, %stradd2_op, %bn_op, %bias_op, %relu_op) {relative_op_order=true}
ddl.constraint() {min_num_cores = 1}

// const
%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}

// alias input, kernel, ptsum tensor
%inptensor = ddl.alias_one_tensor_of(%inptensor_int8, %inptensor_fp16, %inptensor_fp8, %inptensor_int4)
%kertensor = ddl.alias_one_tensor_of(%kertensor_int8, %kertensor_fp16, %kertensor_fp8, %kertensor_int4)
%ptsum = ddl.alias_one_tensor_of(%ptsum_fp, %ptsum_int)

// lx space allocation -- psum and ptsum does not enter lx..
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"}
%kertensor_lx_allocation = ddl.get_external_data_transfer_allocation (%kertensor) {memory="lx", data_connect="l3_lx_kernel"}
%bnA_lx_allocation = ddl.get_external_data_transfer_allocation (%bnA) {memory="lx", data_connect="l3_lx_bnA"}
%bnB_lx_allocation = ddl.get_external_data_transfer_allocation (%bnB) {memory="lx", data_connect="l3_lx_bnB"}
%bias_lx_allocation = ddl.get_external_data_transfer_allocation (%bias) {memory="lx", data_connect="l3_lx_bias"}
%resadd_lx_allocation = ddl.get_external_data_transfer_allocation (%resadd) {memory="lx", data_connect="l3_lx_resadd"}
%resadd2_lx_allocation = ddl.get_external_data_transfer_allocation (%resadd2) {memory="lx", data_connect="l3_lx_resadd2"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) { memory="lx", data_connect="lxsu_output"}
%kertensor_xrf_ext_allocation = ddl.get_external_data_transfer_allocation (%kertensor) {memory="ptxrf", data_connect="xrf_kernel"}

// ddl.constraint(%kertensor_lx_allocation, %kertensor_xrf_ext_allocation) {min_num_valid = 1, max_num_valid = 1}  // re-enable after above-lx scheduler is implemented

ddl.dataflow {
  // datastages
  %d_datastage = ddl.get_external_datastage{property = "core"}
  %b_datastage = ddl.get_external_datastage {property = "chunk"}
  %blk_datastage = ddl.datastage {strategy="maximize"}
  %blk_rows_iter_datastage = ddl.datastage {strategy="maximize"}
  %accum_in_stick_datastage = ddl.datastage {strategy="minimize"}
  %interleave_datastage = ddl.datastage {strategy="maximize", allow_epilogue=true}
  %accum2_datastage = ddl.datastage {strategy="minimize"}
  %bottom_datastage = ddl.datastage {strategy="minimize"}

  // datastage constraints
  ddl.datastage_constraint(%bottom_datastage, %outtensor, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {values=["1"]}
  ddl.datastage_constraint(%accum_in_stick_datastage, %inptensor, %in) {values=["0.125"]}  // 1/8 (one row)
  ddl.if(%fp_bmm) {
    ddl.if(%fp16_bmm) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["8"]}
    } 
    ddl.if(%fp8_bmm) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["4"]}
    }
    ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {values = ["1","2","4"]}
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["1"]}
  }
  ddl.if(%int_bmm) {
    ddl.if(%int8_bmm) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["4"]}
    } 
    ddl.if(%int4_bmm) {
      ddl.datastage_constraint(%blk_rows_iter_datastage, %kertensor, %in) {values=["2"]}
    }
    ddl.datastage_constraint(%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {max="2"}
    ddl.datastage_constraint(%bottom_datastage, %kertensor, %in) {values=["2"]}
  }
  ddl.if(%fp8_bmm) {
    ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %in) {values=["4"]}  // "accum4" in fp8 because of break of IN coordinates within slice
  } else {
    ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %in) {values=["2"]}
  }


  ddl.loop (%d_datastage, %b_datastage, %in, %out, %nrd#0, %nrd#1, %nrd#2, %wrd#0, %wrd#1, %wrd#2, %wrd#3){label="chunk_loop"} {  
    %cond_first_chunkinloop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="first"}
    %cond_last_chunkinloop = ddl.condition(%in){loop_label="chunk_loop", condition="eq", value_expr="last"}
    ddl.loop (%b_datastage, %bottom_datastage, %out, %nrd#0, %nrd#1, %nrd#2) {} {

      ddl.loop (%b_datastage, %blk_datastage, %in) {label="blkload_loop"} {
        %cond_first_blkloadloop = ddl.condition(%in){loop_label="blkload_loop", condition="eq", value_expr="first"}
        %cond_first_inloop = ddl.condition_and(%cond_first_chunkinloop, %cond_first_blkloadloop)
        %cond_not_first_inloop = ddl.condition_not(%cond_first_inloop)
        %psum_start, %psum_end, %next_core, %prev_core = ddl.core_to_core_communication(%in)
        // block load into lx-XRF
        %kertensor_xrf_allocation = ddl.allocate(%kertensor, %kertensor_xrf_ext_allocation) {memory="ptxrf"}
        ddl.if (%kertensor_lx_allocation) {
          ddl.loop (%blk_datastage, %blk_rows_iter_datastage, %in) {} {
            %src_ker_lxpt = ddl.unit(%kertensor, %kertensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_kernel"}
            %dst_ker_lxpt = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", data_connect="xrf_kernel"} 
            ddl.data_transfer(%src_ker_lxpt, [%dst_ker_lxpt]) {}
          }
        }
        ddl.if(%psum_end) {
          ddl.if(%cond_not_first_inloop) {
            ddl.sync {units=["lxsu"], is_receive=false, signal_name="input-lxsu-lxlu-sync", separate_corelets=true}
            ddl.sync {units=["lxlu"], is_receive=true, signal_name="input-lxsu-lxlu-sync", separate_corelets=true} 
          }               
        }               
        ddl.loop (%b_datastage, %interleave_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
          // block load bn / biasAdd param to sfp
          %bnA_sfp_allocation = ddl.allocate(%bnA) {memory="sfplrf"}
          %bnB_sfp_allocation = ddl.allocate(%bnB) {memory="sfplrf"}
          %bias_sfp_allocation = ddl.allocate(%bias) {memory="sfplrf"}
          ddl.if(%psum_end) {
            ddl.if(%cond_last_chunkinloop) {
              ddl.if(%bn_op) {
                %src_bnA_lxsfp = ddl.unit(%bnA, %bnA_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnA"}
                %dst_bnA_lxsfp = ddl.unit(%bnA, %bnA_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnA"}
                ddl.data_transfer(%src_bnA_lxsfp, [%dst_bnA_lxsfp]) {}
                %src_bnB_lxsfp = ddl.unit(%bnB, %bnB_lx_allocation) {unit="lxlu", data_connect="l3_lx_bnB"}
                %dst_bnB_lxsfp = ddl.unit(%bnB, %bnB_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bnB"} 
                ddl.data_transfer(%src_bnB_lxsfp, [%dst_bnB_lxsfp]) {}
              }
              ddl.if(%bias_op) {
                %src_bias_lxsfp = ddl.unit(%bias, %bias_lx_allocation) {unit="lxlu", data_connect="l3_lx_bias"}
                %dst_bias_lxsfp = ddl.unit(%bias, %bias_sfp_allocation) {unit="sfp", vias=["pe"], data_connect="sfp_bias"}
                ddl.data_transfer(%src_bias_lxsfp, [%dst_bias_lxsfp]) {}
              }
            }
          }
          // load input to lx-l0
          %inptensor_l0_allocation = ddl.allocate(%inptensor) {memory="l0", num_buffers=-1:si64}
          ddl.implicit_sync(%inptensor_l0_allocation)
          
          %src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
          %dst_inp_lxl0 = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0su", data_connect="l0_input"} 
          ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0]) {}

          // load output to lx-pe-fifo
          %outtensor_pe_allocation = ddl.allocate(%outtensor) {memory="pelrf"}
          %pe_out_lrf = ddl.unit(%outtensor, %outtensor_pe_allocation) {unit="pe", data_connect="lxpe_output"}
          ddl.if(%psum_end) {
            ddl.if(%cond_not_first_inloop) {
              %src_out_lxpe = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxlu", data_connect="lxsu_output"}
              ddl.data_transfer(%src_out_lxpe, [%pe_out_lrf]) {}
            }
          }
          %pesum_allocation = ddl.allocate(%pesum) {memory="pelrf"}
          %pe_fma_lrf = ddl.unit(%pesum, %pesum_allocation) {unit="pe", data_connect="pe_outtensor"}
          %pe_fma_dst00_sfp = ddl.unit(%outtensor) {unit="sfp", data_connect="pe_outtensor"}
          ddl.loop (%blk_datastage, %accum_in_stick_datastage, %in) {label="blkaccumstick_loop"} {
            ddl.loop (%accum_in_stick_datastage, %accum2_datastage, %in) {label="accumstickaccum2_loop"} {
              %cond_first_blkaccumstickloop = ddl.condition(%in){loop_label="blkaccumstick_loop", condition="eq", value_expr="first"}
              %cond_first_accumstickaccum2loop = ddl.condition(%in){loop_label="accumstickaccum2_loop", condition="eq", value_expr="first"}
              %cond_first_blkaccum2loop = ddl.condition_and(%cond_first_blkaccumstickloop, %cond_first_accumstickaccum2loop)
              %cond_last_blkaccumstickloop = ddl.condition(%in){loop_label="blkaccumstick_loop", condition="eq", value_expr="last"}
              %cond_last_accumstickaccum2loop = ddl.condition(%in){loop_label="accumstickaccum2_loop", condition="eq", value_expr="last"}
              %cond_last_blkaccum2loop = ddl.condition_and(%cond_last_blkaccumstickloop, %cond_last_accumstickaccum2loop)
              ddl.loop (%accum2_datastage, %bottom_datastage, %in) {label="accum2bottom_loop"} {
                %cond_first_accum2bottomloop = ddl.condition(%in){loop_label="accum2bottom_loop", condition="eq", value_expr="first"}
                %cond_last_accum2bottomloop = ddl.condition(%in){loop_label="accum2bottom_loop", condition="eq", value_expr="last"}
                %ptsum_arf_allocation = ddl.allocate(%ptsum) {memory="ptarf"}
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  // load l0-pt
                  %src_inp_l0pt = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0lu", data_connect="l0_input"}
                  %dst_inp_l0pt = ddl.unit(%inptensor) {unit="pt", data_connect="pt_input"} 
                  ddl.data_transfer(%src_inp_l0pt, [%dst_inp_l0pt]) {}

                  // do IMA/FMA 
                  %pt_src00 = ddl.unit(%inptensor) {unit="l0lu", data_connect="pt_input"}
                  %pt_src01 = ddl.unit(%kertensor, %kertensor_xrf_allocation) {unit="pt", data_connect="xrf_kernel"}
                  %pt_src02_arf = ddl.unit(%ptsum, %ptsum_arf_allocation) {unit="pt", data_connect="arf_ptsum"}
                  %pt_src02_north = ddl.unit(%ptsum) {unit="ptnorth", data_connect="arf_ptsum"}
                  %pt_dst00_arf = ddl.unit(%ptsum, %ptsum_arf_allocation) {unit="pt", data_connect="arf_ptsum"}
                  %pt_dst00_south = ddl.unit(%ptsum) {unit="ptsouth", data_connect="arf_ptsum"}
                  
                  ddl.if(%cond_first_accum2bottomloop) {
                    // flavor1: 0 + store to arf..
                    ddl.compute([%pt_src00, %pt_src01, %zero_const], [%pt_dst00_arf]) {computetype="MACC", unit="ptrow0"}
                    // flavor2: north + store to arf..
                    ddl.compute([%pt_src00, %pt_src01, %pt_src02_north], [%pt_dst00_arf]) {computetype="MACC", unit="ptrow1-7"}
                  } else {
                    ddl.if(%cond_last_accum2bottomloop) {
                      // flavor4: arf + send to south
                      ddl.compute([%pt_src00, %pt_src01, %pt_src02_arf], [%pt_dst00_south]) {computetype="MACC", unit="pt"}
                    } else {
                      // flavor3: arf + store to arf 
                      ddl.compute([%pt_src00, %pt_src01, %pt_src02_arf], [%pt_dst00_arf]) {computetype="MACC", unit="pt"}
                    }
                  }
                  // pt-pe-ptsum
                  ddl.if(%cond_last_accum2bottomloop) {
                    %src_ptsum_ptpe = ddl.unit(%ptsum) {unit="ptrow7", data_connect="arf_ptsum"}
                    %dst_ptsum_ptpe = ddl.unit(%ptsum) {unit="pe", data_connect="pe_ptsum"}
                    ddl.data_transfer(%src_ptsum_ptpe, [%dst_ptsum_ptpe]) {}                            
                  }
                }
              }
              ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                // chunk accumulation in PE
                %pe_fma_src00 = ddl.unit(%ptsum) {unit="pt", data_connect="pe_ptsum"}
                ddl.if(%cond_first_blkaccum2loop) {
                  // chunk accumulation in PE: 0 + ptsum => lrf
                  ddl.compute([%pe_fma_src00, %one_const, %zero_const], [%pe_fma_lrf]) {computetype="FMA16", unit="pe"}
                } else {
                  // chunk accumulation in PE: lrf + ptsum => lrf
                  ddl.compute([%pe_fma_src00, %one_const, %pe_fma_lrf], [%pe_fma_lrf]) {computetype="FMA16", unit="pe"}                                        
                }
              }
            }
          }
          ddl.if(%psum_end) {
            ddl.if(%cond_not_first_inloop) {
              ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                // chunk accumulation in PE: lrf + lrf => lrf
                ddl.compute([%pe_out_lrf, %one_const, %pe_fma_lrf], [%pe_fma_lrf]) {computetype="FMA16", unit="pe"}
              }
            }                       
          }                       
          // pe-sfp-output
          %dst_out_pesfp = ddl.unit(%outtensor) {unit="sfp", data_connect="sfp_outtensor"}
          ddl.data_transfer(%pe_fma_lrf, [%dst_out_pesfp]) {}     

          // Aux ops in SFP 
          %outtensor_sfp_allocation = ddl.allocate(%outtensor) {memory="sfplrf"}
          %cond_last_blkloadinloop = ddl.condition(%in){loop_label="blkload_loop", condition="eq", value_expr="last"}
          %cond_last_inaccum = ddl.condition_and(%cond_last_blkloadinloop, %cond_last_chunkinloop)
          %is_any_aux_op = ddl.condition_or(%stradd_op, %stradd2_op, %bn_op, %bias_op, %relu_op)
          %cond_aux_ops_iter = ddl.condition_and(%cond_last_inaccum, %is_any_aux_op)
          %cond_psum_or_aux_ops_iter = ddl.condition_or(%psum_op, %cond_aux_ops_iter)
          %not_psum = ddl.condition_not(%psum_op)
          
          %src_out_pesfp = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
          %sfp_output_lrf = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
          ddl.if (%psum_op) {
            ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
              %sfp_psum_src02 = ddl.unit(%outtensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
              %sfp_psum_dst00_sfpring = ddl.unit(%outtensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
              ddl.if (%psum_start) {
                // psum: 0 + pe -> sfpring
                ddl.compute([%src_out_pesfp, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
              } else {
                ddl.if (%psum_end) {
                  // psum: sfp-ring + pe -> lrf
                  ddl.compute([%src_out_pesfp, %one_const, %sfp_psum_src02], [%sfp_output_lrf]) {computetype="FMA16", unit="sfp"}                                     
                } else { // middle cores
                  // psum: sfp-ring + pe -> sfpring
                  ddl.compute([%src_out_pesfp, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
                }
              }
            }
          }
          ddl.if(%psum_end) {
            ddl.if (%cond_aux_ops_iter) {
              ddl.if (%not_psum) {
                ddl.data_transfer(%src_out_pesfp, [%sfp_output_lrf]) {}     
              }
              // strided add
              ddl.if (%stradd_op) {
                // load stridedAdd input from lxlu to pe-sfp-fifo
                %src_resadd_lxsfp = ddl.unit(%resadd, %resadd_lx_allocation) {unit="lxlu", data_connect="l3_lx_resadd"}
                %dst_resadd_lxsfp = ddl.unit(%resadd) {unit="sfp", vias=["pe"], data_connect="sfp_resadd"} 
                ddl.data_transfer(%src_resadd_lxsfp, [%dst_resadd_lxsfp]) {}
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  %sfp_stradd_src02_pe = ddl.unit(%resadd) {unit="pe", data_connect="sfp_resadd"}
                  // stridedadd: pe + lrf -> lrf
                  ddl.compute([%sfp_output_lrf, %one_const, %sfp_stradd_src02_pe], [%sfp_output_lrf]) {computetype="FMA16", unit="sfp"}
                }
              }
              ddl.if (%stradd2_op) {
                // load stridedAdd input from lxlu to pe-sfp-fifo
                %src_resadd2_lxsfp = ddl.unit(%resadd2, %resadd2_lx_allocation) {unit="lxlu", data_connect="l3_lx_resadd2"}
                %dst_resadd2_lxsfp = ddl.unit(%resadd2) {unit="sfp", vias=["pe"], data_connect="sfp_resadd2"} 
                ddl.data_transfer(%src_resadd2_lxsfp, [%dst_resadd2_lxsfp]) {}
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  %sfp_stradd2_src02_pe = ddl.unit(%resadd2) {unit="pe", data_connect="sfp_resadd2"}
                  // stridedadd: pe + lrf -> lrf
                  ddl.compute([%sfp_output_lrf, %one_const, %sfp_stradd2_src02_pe], [%sfp_output_lrf]) {computetype="FMA16", unit="sfp"}
                }
              }
              // bn
              ddl.if (%bn_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  %sfp_bn_src01 = ddl.unit(%bnA, %bnA_sfp_allocation) {unit="sfp", data_connect="sfp_bnA"}
                  %sfp_bn_src02 = ddl.unit(%bnB, %bnB_sfp_allocation) {unit="sfp", data_connect="sfp_bnB"}
                  // bn: lrf * lrf + lrf -> lrf
                  ddl.compute([%sfp_output_lrf, %sfp_bn_src01, %sfp_bn_src02], [%sfp_output_lrf]) {computetype="FMA16", unit="sfp"}
                }
              }
              // biasadd
              ddl.if (%bias_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  %sfp_biasadd_src02 = ddl.unit(%bias, %bias_sfp_allocation) {unit="sfp", data_connect="sfp_bias"}
                  // biasadd: 1.0 * lrf + lrf -> lrf
                  ddl.compute([%sfp_output_lrf, %one_const, %sfp_biasadd_src02], [%sfp_output_lrf]) {computetype="FMA16", unit="sfp"}
                }
              }
              // relu
              ddl.if (%relu_op) {
                ddl.loop (%interleave_datastage, %bottom_datastage, %wrd#0, %wrd#1, %wrd#2, %wrd#3) {} {
                  // relu: lrf -> lrf
                  ddl.compute([%sfp_output_lrf, %zero_const], [%sfp_output_lrf]) {computetype="FMAX", unit="sfp"}
                }
              }
            }
            ddl.if(%cond_psum_or_aux_ops_iter) {
              // store output tensor sfp-lx
              %src_out_sfplx = ddl.unit(%outtensor, %outtensor_sfp_allocation) {unit="sfp", data_connect="sfp_outtensor_lrf"}
              %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output"}
              ddl.data_transfer(%src_out_sfplx, [%dst_out_sfplx]) {}
            } else {
              // store output tensor pe-sfp-lx
              %noaux_src00_pe = ddl.unit(%outtensor) {unit="pe", data_connect="sfp_outtensor"}
              %dst_out_sfplx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", vias=["sfp"], data_connect="lxsu_output"}
              ddl.data_transfer(%noaux_src00_pe, [%dst_out_sfplx]) {}
            }
          }
        }
      }
    }
  }
}

ddl.transformations {
}

}
