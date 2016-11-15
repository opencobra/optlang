

Maximize
 obj: R_Biomass_Ecoli_core_w_GAM

Subject To
 M_13dpg_c:  R_PGK + R_GAPD  = 0
 M_2pg_c:    - R_PGM - R_ENO  = 0
 M_3pg_c:    - 1.496 R_Biomass_Ecoli_core_w_GAM - R_PGK + R_PGM  = 0
 M_6pgc_c:   R_PGL - R_GND  = 0
 M_6pgl_c:   - R_PGL + R_G6PDH2r  = 0
 M_ac_c:     R_ACt2r - R_ACKr  = 0
 M_ac_e:     - R_ACt2r - R_EX_ac_e  = 0
 M_acald_c:  R_ALCD2x + R_ACALDt - R_ACALD  = 0
 M_acald_e:  - R_ACALDt - R_EX_acald_e  = 0
 M_accoa_c:  - 3.7478 R_Biomass_Ecoli_core_w_GAM + R_ACALD - R_PTAr + R_PFL
             + R_PDH - R_MALS - R_CS  = 0
 M_acon_C_c: - R_ACONTb + R_ACONTa  = 0
 M_actp_c:   R_ACKr + R_PTAr  = 0
 M_adp_c:    59.81 R_Biomass_Ecoli_core_w_GAM + R_PGK + R_ACKr + R_SUCOAS
             - R_PYK + R_PPCK + R_PFK + R_GLNabc + R_GLNS - R_ATPS4r + R_ATPM
             + 2 R_ADK1  = 0
 M_akg_c:    4.1182 R_Biomass_Ecoli_core_w_GAM + R_ICDHyr - R_GLUSy + R_GLUDy
             + R_AKGt2r - R_AKGDH  = 0
 M_akg_e:    - R_AKGt2r - R_EX_akg_e  = 0
 M_amp_c:    - R_ADK1 + R_PPS  = 0
 M_atp_c:    - 59.81 R_Biomass_Ecoli_core_w_GAM - R_PGK - R_ACKr - R_SUCOAS
             + R_PYK - R_PPCK - R_PFK - R_GLNabc - R_GLNS + R_ATPS4r - R_ATPM
             - R_ADK1 - R_PPS  = 0
 M_cit_c:    R_CS - R_ACONTa  = 0
 M_co2_c:    R_GND + R_PDH + R_PPCK + R_ICDHyr + R_AKGDH - R_PPC + R_ME2
             + R_ME1 + R_CO2t  = 0
 M_co2_e:    - R_CO2t - R_EX_co2_e  = 0
 M_coa_c:    3.7478 R_Biomass_Ecoli_core_w_GAM - R_ACALD + R_PTAr - R_PFL
             - R_PDH + R_MALS + R_CS - R_SUCOAS - R_AKGDH  = 0
 M_dhap_c:   - R_TPI + R_FBA  = 0
 M_e4p_c:    - 0.361 R_Biomass_Ecoli_core_w_GAM - R_TKT2 + R_TALA  = 0
 M_etoh_c:   - R_ALCD2x + R_ETOHt2r  = 0
 M_etoh_e:   - R_ETOHt2r - R_EX_etoh_e  = 0
 M_f6p_c:    - 0.0709 R_Biomass_Ecoli_core_w_GAM - R_PFK + R_TKT2 + R_TALA
             + R_PGI + R_FRUpts2 + R_FBP  = 0
 M_fdp_c:    R_PFK - R_FBA - R_FBP  = 0
 M_for_c:    R_PFL - R_FORti + R_FORt2  = 0
 M_for_e:    R_FORti - R_FORt2 - R_EX_for_e  = 0
 M_fru_e:    - R_FRUpts2 - R_EX_fru_e  = 0
 M_fum_c:    R_SUCDi + R_FUMt2_2 - R_FUM - R_FRD7  = 0
 M_fum_e:    - R_FUMt2_2 - R_EX_fum_e  = 0
 M_g3p_c:    - 0.129 R_Biomass_Ecoli_core_w_GAM - R_GAPD + R_TPI + R_FBA
             + R_TKT2 - R_TALA + R_TKT1  = 0
 M_g6p_c:    - 0.205 R_Biomass_Ecoli_core_w_GAM - R_G6PDH2r - R_PGI + R_GLCpts
              = 0
 M_glc_D_e:  - R_GLCpts - R_EX_glc_e  = 0
 M_gln_L_c:  - 0.2557 R_Biomass_Ecoli_core_w_GAM + R_GLNabc + R_GLNS - R_GLUSy
             - R_GLUN  = 0
 M_gln_L_e:  - R_GLNabc - R_EX_gln_L_e  = 0
 M_glu_L_c:  - 4.9414 R_Biomass_Ecoli_core_w_GAM - R_GLNS + 2 R_GLUSy - R_GLUDy
             + R_GLUN + R_GLUt2r  = 0
 M_glu_L_e:  - R_GLUt2r - R_EX_glu_L_e  = 0
 M_glx_c:    - R_MALS + R_ICL  = 0
 M_h2o_c:    - 59.81 R_Biomass_Ecoli_core_w_GAM + R_ENO - R_PGL - R_MALS - R_CS
             - R_ACONTb + R_ACONTa - R_GLNabc + R_ATPS4r - R_ATPM - R_GLUDy
             - R_PPS - R_PPC - R_FBP - R_FUM - R_GLUN + R_H2Ot + R_CYTBD  = 0
 M_h2o_e:    - R_H2Ot - R_EX_h2o_e  = 0
 M_h_c:      59.81 R_Biomass_Ecoli_core_w_GAM + R_GAPD + R_PGL + R_G6PDH2r
             + R_ACt2r + R_ALCD2x + R_ACALD + R_MALS + R_CS - R_PYK + R_PFK
             + R_GLNabc + R_GLNS + 3 R_ATPS4r + R_ATPM - R_GLUSy + R_GLUDy
             + R_AKGt2r + 2 R_PPS + R_PPC + R_ETOHt2r + R_FORt2 + 2 R_FUMt2_2
             + R_GLUt2r - 2 R_CYTBD + 2 R_THD2 + R_SUCCt3 + 2 R_SUCCt2_2
             + R_PYRt2r + R_PIt2r - 4 R_NADH16 + R_MDH + 2 R_MALt2_2 + R_LDH_D
             + R_D_LACt2  = 0
 M_h_e:      - R_ACt2r - 4 R_ATPS4r - R_AKGt2r - R_ETOHt2r - R_FORt2
             - 2 R_FUMt2_2 - R_GLUt2r + 2 R_CYTBD - 2 R_THD2 - R_SUCCt3
             - 2 R_SUCCt2_2 - R_PYRt2r - R_PIt2r + 3 R_NADH16 - 2 R_MALt2_2
             - R_D_LACt2 - R_EX_h_e  = 0
 M_icit_c:   R_ACONTb - R_ICDHyr - R_ICL  = 0
 M_lac_D_c:  - R_LDH_D + R_D_LACt2  = 0
 M_lac_D_e:  - R_D_LACt2 - R_EX_lac_D_e  = 0
 M_mal_L_c:  R_MALS - R_ME2 - R_ME1 + R_FUM - R_MDH + R_MALt2_2  = 0
 M_mal_L_e:  - R_MALt2_2 - R_EX_mal_L_e  = 0
 M_nad_c:    - 3.547 R_Biomass_Ecoli_core_w_GAM - R_GAPD - R_ALCD2x - R_ACALD
             - R_PDH - R_AKGDH - R_ME1 + R_THD2 + R_NADH16 - R_MDH - R_LDH_D
             - R_NADTRHD  = 0
 M_nadh_c:   3.547 R_Biomass_Ecoli_core_w_GAM + R_GAPD + R_ALCD2x + R_ACALD
             + R_PDH + R_AKGDH + R_ME1 - R_THD2 - R_NADH16 + R_MDH + R_LDH_D
             + R_NADTRHD  = 0
 M_nadp_c:   13.0279 R_Biomass_Ecoli_core_w_GAM - R_GND - R_G6PDH2r - R_ICDHyr
             + R_GLUSy - R_GLUDy - R_ME2 - R_THD2 + R_NADTRHD  = 0
 M_nadph_c:  - 13.0279 R_Biomass_Ecoli_core_w_GAM + R_GND + R_G6PDH2r
             + R_ICDHyr - R_GLUSy + R_GLUDy + R_ME2 + R_THD2 - R_NADTRHD  = 0
 M_nh4_c:    - R_GLNS + R_GLUDy + R_GLUN + R_NH4t  = 0
 M_nh4_e:    - R_NH4t - R_EX_nh4_e  = 0
 M_o2_c:     - 0.5 R_CYTBD + R_O2t  = 0
 M_o2_e:     - R_O2t - R_EX_o2_e  = 0
 M_oaa_c:    - 1.7867 R_Biomass_Ecoli_core_w_GAM - R_CS - R_PPCK + R_PPC
             + R_MDH  = 0
 M_pep_c:    - 0.5191 R_Biomass_Ecoli_core_w_GAM + R_ENO - R_PYK + R_PPCK
             + R_PPS - R_PPC - R_FRUpts2 - R_GLCpts  = 0
 M_pi_c:     59.81 R_Biomass_Ecoli_core_w_GAM - R_GAPD - R_PTAr + R_SUCOAS
             + R_GLNabc + R_GLNS - R_ATPS4r + R_ATPM + R_PPS + R_PPC + R_FBP
             + R_PIt2r  = 0
 M_pi_e:     - R_PIt2r - R_EX_pi_e  = 0
 M_pyr_c:    - 2.8328 R_Biomass_Ecoli_core_w_GAM - R_PFL - R_PDH + R_PYK
             - R_PPS + R_ME2 + R_ME1 + R_FRUpts2 + R_GLCpts + R_PYRt2r
             + R_LDH_D  = 0
 M_pyr_e:    - R_PYRt2r - R_EX_pyr_e  = 0
 M_q8_c:     - R_SUCDi + R_FRD7 + R_CYTBD - R_NADH16  = 0
 M_q8h2_c:   R_SUCDi - R_FRD7 - R_CYTBD + R_NADH16  = 0
 M_r5p_c:    - 0.8977 R_Biomass_Ecoli_core_w_GAM - R_TKT1 - R_RPI  = 0
 M_ru5p_D_c: R_GND + R_RPI - R_RPE  = 0
 M_s7p_c:    - R_TALA + R_TKT1  = 0
 M_succ_c:   - R_SUCOAS - R_SUCDi + R_FRD7 + R_ICL - R_SUCCt3 + R_SUCCt2_2  = 0
 M_succ_e:   R_SUCCt3 - R_SUCCt2_2 - R_EX_succ_e  = 0
 M_succoa_c: R_SUCOAS + R_AKGDH  = 0
 M_xu5p_D_c: - R_TKT2 - R_TKT1 + R_RPE  = 0
 M_13dpg_c:  R_Biomass_Ecoli_core_w_GAM + R_PGK  = 0
 M_2pg_c:    - R_GAPD - R_PGM  = 0
 M_3pg_c:    - R_Biomass_Ecoli_core_w_GAM + R_GAPD - 1.496 R_EX_succ_e  = 0
 M_6pgc_c:   R_ENO - R_PGL  = 0
 M_6pgl_c:   - R_ENO + R_GND  = 0
 M_ac_c:     R_G6PDH2r - R_ACt2r  = 0
 M_ac_e:     - R_G6PDH2r - R_ACKr  = 0
 M_acald_c:  R_EX_ac_e + R_ALCD2x - R_ACALDt  = 0
 M_acald_e:  - R_ALCD2x - R_ACALD  = 0
 M_accoa_c:  R_ACALDt - R_EX_acald_e + R_PTAr + R_PFL - R_PDH - R_MALS
             - 3.7478 R_EX_succ_e  = 0
 M_acon_C_c: - R_CS + R_ACONTb  = 0
 M_actp_c:   R_ACt2r + R_EX_acald_e  = 0
 M_adp_c:    R_Biomass_Ecoli_core_w_GAM + R_ACt2r + R_ACONTa - R_SUCOAS + R_PYK
             + R_PPCK + R_PFK + R_GLNabc - R_GLNS + R_ATPS4r + 2 R_ATPM
             + 59.81 R_EX_succ_e  = 0
 M_akg_c:    R_ADK1 - R_ICDHyr + R_GLUSy + R_GLUDy - R_AKGt2r
             + 4.1182 R_EX_succ_e  = 0
 M_akg_e:    - R_GLUDy - R_AKGDH  = 0
 M_amp_c:    - R_ATPM + R_EX_akg_e  = 0
 M_atp_c:    - R_Biomass_Ecoli_core_w_GAM - R_ACt2r - R_ACONTa + R_SUCOAS
             - R_PYK - R_PPCK - R_PFK - R_GLNabc + R_GLNS - R_ATPS4r - R_ATPM
             - R_EX_akg_e - 59.81 R_EX_succ_e  = 0
 M_cit_c:    R_MALS - R_ACONTb  = 0
 M_co2_c:    R_PGL + R_PFL + R_PYK + R_ADK1 + R_AKGt2r - R_PPS + R_PPC + R_ME2
             + R_ME1  = 0
 M_co2_e:    - R_ME1 - R_CO2t  = 0
 M_coa_c:    - R_ACALDt + R_EX_acald_e - R_PTAr - R_PFL + R_PDH + R_MALS
             - R_ACONTa - R_AKGt2r + 3.7478 R_EX_succ_e  = 0
 M_dhap_c:   - R_EX_co2_e + R_TPI  = 0
 M_e4p_c:    - R_FBA + R_TKT2 - 0.361 R_EX_succ_e  = 0
 M_etoh_c:   - R_EX_ac_e + R_TALA  = 0
 M_etoh_e:   - R_TALA - R_ETOHt2r  = 0
 M_f6p_c:    - R_PPCK + R_FBA + R_TKT2 + R_EX_etoh_e + R_PGI + R_FRUpts2
             - 0.0709 R_EX_succ_e  = 0
 M_fdp_c:    R_PPCK - R_TPI - R_FRUpts2  = 0
 M_for_c:    R_PTAr - R_FBP + R_FORti  = 0
 M_for_e:    R_FBP - R_FORti - R_FORt2  = 0
 M_fru_e:    - R_PGI - R_EX_for_e  = 0
 M_fum_c:    R_EX_fru_e + R_SUCDi - R_FUMt2_2 - R_FUM  = 0
 M_fum_e:    - R_SUCDi - R_FRD7  = 0
 M_g3p_c:    - R_PGK + R_EX_co2_e + R_TPI + R_FBA - R_TKT2 + R_EX_fum_e
             - 0.129 R_EX_succ_e  = 0
 M_g6p_c:    - R_GND - R_EX_etoh_e + R_TKT1 - 0.205 R_EX_succ_e  = 0
 M_glc_D_e:  - R_TKT1 - R_GLCpts  = 0
 M_gln_L_c:  R_PFK + R_GLNabc - R_ICDHyr - R_EX_glc_e - 0.2557 R_EX_succ_e  = 0
 M_gln_L_e:  - R_PFK - R_GLUN  = 0
 M_glu_L_c:  - R_GLNabc + 2 R_ICDHyr - R_GLUSy + R_EX_glc_e + R_EX_gln_L_e
             - 4.9414 R_EX_succ_e  = 0
 M_glu_L_e:  - R_EX_gln_L_e - R_GLUt2r  = 0
 M_glx_c:    - R_PDH + R_EX_glu_L_e  = 0
 M_h2o_c:    R_PGM - R_ENO - R_PDH - R_MALS - R_CS + R_ACONTb - R_PFK + R_GLNS
             - R_ATPS4r - R_GLUSy - R_EX_akg_e - R_PPS - R_FRUpts2 - R_FUMt2_2
             - R_EX_glc_e + R_ICL + R_H2Ot - 59.81 R_EX_succ_e  = 0
 M_h2o_e:    - R_ICL - R_CYTBD  = 0
 M_h_c:      R_PGK + R_ENO + R_GND + R_G6PDH2r + R_EX_ac_e + R_ACALDt + R_PDH
             + R_MALS - R_SUCOAS + R_PPCK + R_PFK + R_GLNabc + 3 R_GLNS
             + R_ATPS4r - R_ICDHyr + R_GLUSy + R_GLUDy + 2 R_EX_akg_e + R_PPS
             + R_TALA + R_FORti + 2 R_SUCDi + R_EX_gln_L_e - 2 R_H2Ot
             + 2 R_EX_h2o_e + R_THD2 + 2 R_SUCCt3 + R_SUCCt2_2 + R_PYRt2r
             - 4 R_PIt2r + R_NADH16 + 2 R_MDH + R_MALt2_2 + R_LDH_D
             + 59.81 R_EX_succ_e  = 0
 M_h_e:      - R_G6PDH2r - 4 R_GLNS - R_GLUDy - R_TALA - R_FORti - 2 R_SUCDi
             - R_EX_gln_L_e + 2 R_H2Ot - 2 R_EX_h2o_e - R_THD2 - 2 R_SUCCt3
             - R_SUCCt2_2 - R_PYRt2r + 3 R_PIt2r - 2 R_MDH - R_LDH_D
             - R_D_LACt2  = 0
 M_icit_c:   R_CS - R_ADK1 - R_EX_glu_L_e  = 0
 M_lac_D_c:  - R_MALt2_2 + R_LDH_D  = 0
 M_lac_D_e:  - R_LDH_D - R_EX_h_e  = 0
 M_mal_L_c:  R_PDH - R_PPC - R_ME2 + R_FUMt2_2 - R_NADH16 + R_MDH  = 0
 M_mal_L_e:  - R_MDH - R_EX_lac_D_e  = 0
 M_nad_c:    - R_PGK - R_EX_ac_e - R_ACALDt - R_PFL - R_AKGt2r - R_ME2
             + R_EX_h2o_e + R_PIt2r - R_NADH16 - R_MALt2_2 - R_EX_mal_L_e
             - 3.547 R_EX_succ_e  = 0
 M_nadh_c:   R_PGK + R_EX_ac_e + R_ACALDt + R_PFL + R_AKGt2r + R_ME2
             - R_EX_h2o_e - R_PIt2r + R_NADH16 + R_MALt2_2 + R_EX_mal_L_e
             + 3.547 R_EX_succ_e  = 0
 M_nadp_c:   - R_PGL - R_GND - R_ADK1 + R_ICDHyr - R_GLUSy - R_PPC - R_EX_h2o_e
             + R_EX_mal_L_e + 13.0279 R_EX_succ_e  = 0
 M_nadph_c:  R_PGL + R_GND + R_ADK1 - R_ICDHyr + R_GLUSy + R_PPC + R_EX_h2o_e
             - R_EX_mal_L_e - 13.0279 R_EX_succ_e  = 0
 M_nh4_c:    - R_GLNabc + R_GLUSy + R_EX_glc_e + R_NADTRHD  = 0
 M_nh4_e:    - R_NADTRHD - R_NH4t  = 0
 M_o2_c:     - 0.5 R_H2Ot + R_EX_nh4_e  = 0
 M_o2_e:     - R_EX_nh4_e - R_O2t  = 0
 M_oaa_c:    - R_MALS - R_PYK + R_PPS + R_NADH16 - 1.7867 R_EX_succ_e  = 0
 M_pep_c:    R_PGM - R_SUCOAS + R_PYK + R_EX_akg_e - R_PPS - R_PGI - R_TKT1
             - 0.5191 R_EX_succ_e  = 0
 M_pi_c:     - R_PGK - R_EX_acald_e + R_ACONTa + R_PFK + R_GLNabc - R_GLNS
             + R_ATPS4r + R_EX_akg_e + R_PPS + R_FRUpts2 + R_PYRt2r
             + 59.81 R_EX_succ_e  = 0
 M_pi_e:     - R_PYRt2r - R_EX_o2_e  = 0
 M_pyr_c:    - R_PTAr - R_PFL + R_SUCOAS - R_EX_akg_e + R_PPC + R_ME2 + R_PGI
             + R_TKT1 + R_SUCCt2_2 + R_MALt2_2 - 2.8328 R_EX_succ_e  = 0
 M_pyr_e:    - R_SUCCt2_2 - R_EX_pi_e  = 0
 M_q8_c:     - R_EX_fru_e + R_FUM + R_H2Ot - R_PIt2r  = 0
 M_q8h2_c:   R_EX_fru_e - R_FUM - R_H2Ot + R_PIt2r  = 0
 M_r5p_c:    - R_EX_fum_e - R_EX_pyr_e - 0.8977 R_EX_succ_e  = 0
 M_ru5p_D_c: R_PGL + R_EX_pyr_e - R_RPI  = 0
 M_s7p_c:    - R_TKT2 + R_EX_fum_e  = 0
 M_succ_c:   - R_ACONTa - R_EX_fru_e + R_FUM + R_EX_glu_L_e - R_THD2 + R_SUCCt3
              = 0
 M_succ_e:   R_THD2 - R_SUCCt3 - R_RPE  = 0
 M_succoa_c: R_ACONTa + R_AKGt2r  = 0
 M_xu5p_D_c: - R_FBA - R_EX_fum_e + R_RPI  = 0
Bounds
 0 <= R_Biomass_Ecoli_core_w_GAM <= 10000000
-10000000 <= R_PGK <= 10000000
-10000000 <= R_GAPD <= 10000000
-10000000 <= R_PGM <= 10000000
-10000000 <= R_ENO <= 10000000
 0 <= R_PGL <= 10000000
 0 <= R_GND <= 10000000
-10000000 <= R_G6PDH2r <= 10000000
-10000000 <= R_ACt2r <= 10000000
-10000000 <= R_ACKr <= 10000000
 0 <= R_EX_ac_e <= 10000000
-10000000 <= R_ALCD2x <= 10000000
-10000000 <= R_ACALDt <= 10000000
-10000000 <= R_ACALD <= 10000000
 0 <= R_EX_acald_e <= 10000000
-10000000 <= R_PTAr <= 10000000
 0 <= R_PFL <= 10000000
 0 <= R_PDH <= 10000000
 0 <= R_MALS <= 10000000
 0 <= R_CS <= 10000000
-10000000 <= R_ACONTb <= 10000000
-10000000 <= R_ACONTa <= 10000000
-10000000 <= R_SUCOAS <= 10000000
 0 <= R_PYK <= 10000000
 0 <= R_PPCK <= 10000000
 0 <= R_PFK <= 10000000
 0 <= R_GLNabc <= 10000000
 0 <= R_GLNS <= 10000000
-10000000 <= R_ATPS4r <= 10000000
 8.39 <= R_ATPM <= 10000000
-10000000 <= R_ADK1 <= 10000000
-10000000 <= R_ICDHyr <= 10000000
 0 <= R_GLUSy <= 10000000
-10000000 <= R_GLUDy <= 10000000
-10000000 <= R_AKGt2r <= 10000000
 0 <= R_AKGDH <= 10000000
 0 <= R_EX_akg_e <= 10000000
 0 <= R_PPS <= 10000000
 0 <= R_PPC <= 10000000
 0 <= R_ME2 <= 10000000
 0 <= R_ME1 <= 10000000
-10000000 <= R_CO2t <= 10000000
-10000000 <= R_EX_co2_e <= 10000000
-10000000 <= R_TPI <= 10000000
-10000000 <= R_FBA <= 10000000
-10000000 <= R_TKT2 <= 10000000
-10000000 <= R_TALA <= 10000000
-10000000 <= R_ETOHt2r <= 10000000
 0 <= R_EX_etoh_e <= 10000000
-10000000 <= R_PGI <= 10000000
 0 <= R_FRUpts2 <= 10000000
 0 <= R_FBP <= 10000000
 0 <= R_FORti <= 10000000
 0 <= R_FORt2 <= 10000000
 0 <= R_EX_for_e <= 10000000
 0 <= R_EX_fru_e <= 10000000
 0 <= R_SUCDi <= 10000000
 0 <= R_FUMt2_2 <= 10000000
-10000000 <= R_FUM <= 10000000
 0 <= R_FRD7 <= 10000000
 0 <= R_EX_fum_e <= 10000000
-10000000 <= R_TKT1 <= 10000000
 0 <= R_GLCpts <= 10000000
-10 <= R_EX_glc_e <= 10000000
 0 <= R_GLUN <= 10000000
 0 <= R_EX_gln_L_e <= 10000000
-10000000 <= R_GLUt2r <= 10000000
 0 <= R_EX_glu_L_e <= 10000000
 0 <= R_ICL <= 10000000
-10000000 <= R_H2Ot <= 10000000
 0 <= R_CYTBD <= 10000000
-10000000 <= R_EX_h2o_e <= 10000000
 0 <= R_THD2 <= 10000000
 0 <= R_SUCCt3 <= 10000000
 0 <= R_SUCCt2_2 <= 10000000
-10000000 <= R_PYRt2r <= 10000000
-10000000 <= R_PIt2r <= 10000000
 0 <= R_NADH16 <= 10000000
-10000000 <= R_MDH <= 10000000
 0 <= R_MALt2_2 <= 10000000
-10000000 <= R_LDH_D <= 10000000
-10000000 <= R_D_LACt2 <= 10000000
-10000000 <= R_EX_h_e <= 10000000
 0 <= R_EX_lac_D_e <= 10000000
 0 <= R_EX_mal_L_e <= 10000000
 0 <= R_NADTRHD <= 10000000
-10000000 <= R_NH4t <= 10000000
-10000000 <= R_EX_nh4_e <= 10000000
-10000000 <= R_O2t <= 10000000
-10000000 <= R_EX_o2_e <= 10000000
-10000000 <= R_EX_pi_e <= 10000000
 0 <= R_EX_pyr_e <= 10000000
-10000000 <= R_RPI <= 10000000
-10000000 <= R_RPE <= 10000000
 0 <= R_EX_succ_e <= 10000000
End
