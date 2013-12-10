\* Problem: Unknown *\

Maximize
 obj: + R_Biomass_Ecoli_core_w_GAM

Subject To
 M_13dpg_c: + R_PGK + R_GAPD = 0
 M_2pg_c: - R_PGM - R_ENO = 0
 M_3pg_c: + R_PGM - R_PGK - 1.496 R_Biomass_Ecoli_core_w_GAM = 0
 M_6pgc_c: + R_PGL - R_GND = 0
 M_6pgl_c: - R_PGL + R_G6PDH2r = 0
 M_ac_c: + R_ACt2r - R_ACKr = 0
 M_ac_e: - R_EX_ac_e - R_ACt2r = 0
 M_acald_c: + R_ALCD2x + R_ACALDt - R_ACALD = 0
 M_acald_e: - R_EX_acald_e - R_ACALDt = 0
 M_accoa_c: - R_PTAr + R_PFL + R_PDH - R_MALS - R_CS
 - 3.7478 R_Biomass_Ecoli_core_w_GAM + R_ACALD = 0
 M_acon_C_c: - R_ACONTb + R_ACONTa = 0
 M_actp_c: + R_PTAr + R_ACKr = 0
 M_adp_c: + R_SUCOAS - R_PYK + R_PPCK + R_PGK + R_PFK + R_GLNabc
 + R_GLNS + 59.81 R_Biomass_Ecoli_core_w_GAM - R_ATPS4r + R_ATPM
 + 2 R_ADK1 + R_ACKr = 0
 M_akg_c: + R_ICDHyr - R_GLUSy + R_GLUDy
 + 4.1182 R_Biomass_Ecoli_core_w_GAM + R_AKGt2r - R_AKGDH = 0
 M_akg_e: - R_EX_akg_e - R_AKGt2r = 0
 M_amp_c: + R_PPS - R_ADK1 = 0
 M_atp_c: - R_SUCOAS + R_PYK - R_PPS - R_PPCK - R_PGK - R_PFK - R_GLNabc
 - R_GLNS - 59.81 R_Biomass_Ecoli_core_w_GAM + R_ATPS4r - R_ATPM
 - R_ADK1 - R_ACKr = 0
 M_cit_c: + R_CS - R_ACONTa = 0
 M_co2_c: + R_PPCK - R_PPC + R_PDH + R_ME2 + R_ME1 + R_ICDHyr + R_GND
 + R_CO2t + R_AKGDH = 0
 M_co2_e: - R_EX_co2_e - R_CO2t = 0
 M_coa_c: - R_SUCOAS + R_PTAr - R_PFL - R_PDH + R_MALS + R_CS
 + 3.7478 R_Biomass_Ecoli_core_w_GAM - R_AKGDH - R_ACALD = 0
 M_dhap_c: - R_TPI + R_FBA = 0
 M_e4p_c: - R_TKT2 + R_TALA - 0.361 R_Biomass_Ecoli_core_w_GAM = 0
 M_etoh_c: + R_ETOHt2r - R_ALCD2x = 0
 M_etoh_e: - R_EX_etoh_e - R_ETOHt2r = 0
 M_f6p_c: + R_TKT2 + R_TALA + R_PGI - R_PFK + R_FRUpts2 + R_FBP
 - 0.0709 R_Biomass_Ecoli_core_w_GAM = 0
 M_fdp_c: + R_PFK - R_FBP - R_FBA = 0
 M_for_c: + R_PFL - R_FORti + R_FORt2 = 0
 M_for_e: + R_FORti - R_FORt2 - R_EX_for_e = 0
 M_fru_e: - R_FRUpts2 - R_EX_fru_e = 0
 M_fum_c: + R_SUCDi + R_FUMt2_2 - R_FUM - R_FRD7 = 0
 M_fum_e: - R_FUMt2_2 - R_EX_fum_e = 0
 M_g3p_c: + R_TPI + R_TKT2 + R_TKT1 - R_TALA - R_GAPD + R_FBA
 - 0.129 R_Biomass_Ecoli_core_w_GAM = 0
 M_g6p_c: - R_PGI + R_GLCpts - R_G6PDH2r
 - 0.205 R_Biomass_Ecoli_core_w_GAM = 0
 M_glc_D_e: - R_GLCpts - R_EX_glc_e = 0
 M_gln_L_c: - R_GLUSy - R_GLUN + R_GLNabc + R_GLNS
 - 0.2557 R_Biomass_Ecoli_core_w_GAM = 0
 M_gln_L_e: - R_GLNabc - R_EX_gln_L_e = 0
 M_glu_L_c: + R_GLUt2r + 2 R_GLUSy + R_GLUN - R_GLUDy - R_GLNS
 - 4.9414 R_Biomass_Ecoli_core_w_GAM = 0
 M_glu_L_e: - R_GLUt2r - R_EX_glu_L_e = 0
 M_glx_c: - R_MALS + R_ICL = 0
 M_h2o_c: - R_PPS - R_PPC - R_PGL - R_MALS + R_H2Ot - R_GLUN - R_GLUDy
 - R_GLNabc - R_FUM - R_FBP + R_ENO + R_CYTBD - R_CS
 - 59.81 R_Biomass_Ecoli_core_w_GAM + R_ATPS4r - R_ATPM - R_ACONTb
 + R_ACONTa = 0
 M_h2o_e: - R_H2Ot - R_EX_h2o_e = 0
 M_h_c: + 2 R_THD2 + R_SUCCt3 + 2 R_SUCCt2_2 + R_PYRt2r - R_PYK
 + 2 R_PPS + R_PPC + R_PIt2r + R_PGL + R_PFK - 4 R_NADH16 + R_MDH
 + 2 R_MALt2_2 + R_MALS + R_LDH_D + R_GLUt2r - R_GLUSy + R_GLUDy
 + R_GLNabc + R_GLNS + R_GAPD + R_G6PDH2r + 2 R_FUMt2_2 + R_FORt2
 + R_ETOHt2r + R_D_LACt2 - 2 R_CYTBD + R_CS
 + 59.81 R_Biomass_Ecoli_core_w_GAM + 3 R_ATPS4r + R_ATPM + R_ALCD2x
 + R_AKGt2r + R_ACt2r + R_ACALD = 0
 M_h_e: - 2 R_THD2 - R_SUCCt3 - 2 R_SUCCt2_2 - R_PYRt2r - R_PIt2r
 + 3 R_NADH16 - 2 R_MALt2_2 - R_GLUt2r - 2 R_FUMt2_2 - R_FORt2
 - R_EX_h_e - R_ETOHt2r - R_D_LACt2 + 2 R_CYTBD - 4 R_ATPS4r - R_AKGt2r
 - R_ACt2r = 0
 M_icit_c: - R_ICL - R_ICDHyr + R_ACONTb = 0
 M_lac_D_c: - R_LDH_D + R_D_LACt2 = 0
 M_lac_D_e: - R_EX_lac_D_e - R_D_LACt2 = 0
 M_mal_L_c: - R_ME2 - R_ME1 - R_MDH + R_MALt2_2 + R_MALS + R_FUM = 0
 M_mal_L_e: - R_MALt2_2 - R_EX_mal_L_e = 0
 M_nad_c: + R_THD2 - R_PDH - R_NADTRHD + R_NADH16 - R_ME1 - R_MDH
 - R_LDH_D - R_GAPD - 3.547 R_Biomass_Ecoli_core_w_GAM - R_ALCD2x
 - R_AKGDH - R_ACALD = 0
 M_nadh_c: - R_THD2 + R_PDH + R_NADTRHD - R_NADH16 + R_ME1 + R_MDH
 + R_LDH_D + R_GAPD + 3.547 R_Biomass_Ecoli_core_w_GAM + R_ALCD2x
 + R_AKGDH + R_ACALD = 0
 M_nadp_c: - R_THD2 + R_NADTRHD - R_ME2 - R_ICDHyr - R_GND + R_GLUSy
 - R_GLUDy - R_G6PDH2r + 13.0279 R_Biomass_Ecoli_core_w_GAM = 0
 M_nadph_c: + R_THD2 - R_NADTRHD + R_ME2 + R_ICDHyr + R_GND - R_GLUSy
 + R_GLUDy + R_G6PDH2r - 13.0279 R_Biomass_Ecoli_core_w_GAM = 0
 M_nh4_c: + R_NH4t + R_GLUN + R_GLUDy - R_GLNS = 0
 M_nh4_e: - R_NH4t - R_EX_nh4_e = 0
 M_o2_c: + R_O2t - 0.5 R_CYTBD = 0
 M_o2_e: - R_O2t - R_EX_o2_e = 0
 M_oaa_c: - R_PPCK + R_PPC + R_MDH - R_CS
 - 1.7867 R_Biomass_Ecoli_core_w_GAM = 0
 M_pep_c: - R_PYK + R_PPS + R_PPCK - R_PPC - R_GLCpts - R_FRUpts2
 + R_ENO - 0.5191 R_Biomass_Ecoli_core_w_GAM = 0
 M_pi_c: + R_SUCOAS - R_PTAr + R_PPS + R_PPC + R_PIt2r + R_GLNabc
 + R_GLNS - R_GAPD + R_FBP + 59.81 R_Biomass_Ecoli_core_w_GAM - R_ATPS4r
 + R_ATPM = 0
 M_pi_e: - R_PIt2r - R_EX_pi_e = 0
 M_pyr_c: + R_PYRt2r + R_PYK - R_PPS - R_PFL - R_PDH + R_ME2 + R_ME1
 + R_LDH_D + R_GLCpts + R_FRUpts2 - 2.8328 R_Biomass_Ecoli_core_w_GAM
 = 0
 M_pyr_e: - R_PYRt2r - R_EX_pyr_e = 0
 M_q8_c: - R_SUCDi - R_NADH16 + R_FRD7 + R_CYTBD = 0
 M_q8h2_c: + R_SUCDi + R_NADH16 - R_FRD7 - R_CYTBD = 0
 M_r5p_c: - R_TKT1 - R_RPI - 0.8977 R_Biomass_Ecoli_core_w_GAM = 0
 M_ru5p_D_c: + R_RPI - R_RPE + R_GND = 0
 M_s7p_c: + R_TKT1 - R_TALA = 0
 M_succ_c: - R_SUCOAS - R_SUCDi - R_SUCCt3 + R_SUCCt2_2 + R_ICL + R_FRD7
 = 0
 M_succ_e: + R_SUCCt3 - R_SUCCt2_2 - R_EX_succ_e = 0
 M_succoa_c: + R_SUCOAS + R_AKGDH = 0
 M_xu5p_D_c: - R_TKT2 - R_TKT1 + R_RPE = 0

Bounds
 -10000000 <= R_ACALD <= 10000000
 -10000000 <= R_ACALDt <= 10000000
 -10000000 <= R_ACKr <= 10000000
 -10000000 <= R_ACONTa <= 10000000
 -10000000 <= R_ACONTb <= 10000000
 -10000000 <= R_ACt2r <= 10000000
 -10000000 <= R_ADK1 <= 10000000
 0 <= R_AKGDH <= 10000000
 -10000000 <= R_AKGt2r <= 10000000
 -10000000 <= R_ALCD2x <= 10000000
 8.39 <= R_ATPM <= 10000000
 -10000000 <= R_ATPS4r <= 10000000
 0 <= R_Biomass_Ecoli_core_w_GAM <= 10000000
 -10000000 <= R_CO2t <= 10000000
 0 <= R_CS <= 10000000
 0 <= R_CYTBD <= 10000000
 -10000000 <= R_D_LACt2 <= 10000000
 -10000000 <= R_ENO <= 10000000
 -10000000 <= R_ETOHt2r <= 10000000
 0 <= R_EX_ac_e <= 10000000
 0 <= R_EX_acald_e <= 10000000
 0 <= R_EX_akg_e <= 10000000
 -10000000 <= R_EX_co2_e <= 10000000
 0 <= R_EX_etoh_e <= 10000000
 0 <= R_EX_for_e <= 10000000
 0 <= R_EX_fru_e <= 10000000
 0 <= R_EX_fum_e <= 10000000
 -10 <= R_EX_glc_e <= 10000000
 0 <= R_EX_gln_L_e <= 10000000
 0 <= R_EX_glu_L_e <= 10000000
 -10000000 <= R_EX_h_e <= 10000000
 -10000000 <= R_EX_h2o_e <= 10000000
 0 <= R_EX_lac_D_e <= 10000000
 0 <= R_EX_mal_L_e <= 10000000
 -10000000 <= R_EX_nh4_e <= 10000000
 -10000000 <= R_EX_o2_e <= 10000000
 -10000000 <= R_EX_pi_e <= 10000000
 0 <= R_EX_pyr_e <= 10000000
 0 <= R_EX_succ_e <= 10000000
 -10000000 <= R_FBA <= 10000000
 0 <= R_FBP <= 10000000
 0 <= R_FORt2 <= 10000000
 0 <= R_FORti <= 10000000
 0 <= R_FRD7 <= 10000000
 0 <= R_FRUpts2 <= 10000000
 -10000000 <= R_FUM <= 10000000
 0 <= R_FUMt2_2 <= 10000000
 -10000000 <= R_G6PDH2r <= 10000000
 -10000000 <= R_GAPD <= 10000000
 0 <= R_GLCpts <= 10000000
 0 <= R_GLNS <= 10000000
 0 <= R_GLNabc <= 10000000
 -10000000 <= R_GLUDy <= 10000000
 0 <= R_GLUN <= 10000000
 0 <= R_GLUSy <= 10000000
 -10000000 <= R_GLUt2r <= 10000000
 0 <= R_GND <= 10000000
 -10000000 <= R_H2Ot <= 10000000
 -10000000 <= R_ICDHyr <= 10000000
 0 <= R_ICL <= 10000000
 -10000000 <= R_LDH_D <= 10000000
 0 <= R_MALS <= 10000000
 0 <= R_MALt2_2 <= 10000000
 -10000000 <= R_MDH <= 10000000
 0 <= R_ME1 <= 10000000
 0 <= R_ME2 <= 10000000
 0 <= R_NADH16 <= 10000000
 0 <= R_NADTRHD <= 10000000
 -10000000 <= R_NH4t <= 10000000
 -10000000 <= R_O2t <= 10000000
 0 <= R_PDH <= 10000000
 0 <= R_PFK <= 10000000
 0 <= R_PFL <= 10000000
 -10000000 <= R_PGI <= 10000000
 -10000000 <= R_PGK <= 10000000
 0 <= R_PGL <= 10000000
 -10000000 <= R_PGM <= 10000000
 -10000000 <= R_PIt2r <= 10000000
 0 <= R_PPC <= 10000000
 0 <= R_PPCK <= 10000000
 0 <= R_PPS <= 10000000
 -10000000 <= R_PTAr <= 10000000
 0 <= R_PYK <= 10000000
 -10000000 <= R_PYRt2r <= 10000000
 -10000000 <= R_RPE <= 10000000
 -10000000 <= R_RPI <= 10000000
 0 <= R_SUCCt2_2 <= 10000000
 0 <= R_SUCCt3 <= 10000000
 0 <= R_SUCDi <= 10000000
 -10000000 <= R_SUCOAS <= 10000000
 -10000000 <= R_TALA <= 10000000
 0 <= R_THD2 <= 10000000
 -10000000 <= R_TKT1 <= 10000000
 -10000000 <= R_TKT2 <= 10000000
 -10000000 <= R_TPI <= 10000000

End
